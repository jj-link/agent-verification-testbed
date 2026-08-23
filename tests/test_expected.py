"""Tests for the continuous Qwen verifier aggregation (plan 13.4)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from avt.config import load_config
from avt.expected import ContinuousVerifier, CoverageError
from avt.storage.ids import candidate_id, pair_id

CONFIG_YAML = """
experiment:
  name: s
  seed: 42
  task_file: __TASKFILE__
  candidates_per_task: __N__
upstream:
  terminal_bench_commit: abc
  harbor_commit: def
generator:
  harness: qwen-coder
  model: m
  endpoint: e
  temperature: 0.7
  max_tokens: 8192
verifier:
  model: m
  endpoint: http://host.docker.internal:8000/v1
  criteria: [specification]
  granularity: 5
  repetitions: 1
rendering:
  max_pair_context_tokens: 1000
ranking:
  method: round_robin_bradley_terry
storage:
  root: __ROOT__
  metadata_db: __ROOT__/experiment.sqlite
  ground_truth_db: __ROOT__/ground_truth.sqlite
"""


def _make(tmp_path: Path, n: int = 2) -> ContinuousVerifier:
    (tmp_path / "tasks.txt").write_text("task_x\n", encoding="utf-8")
    cfg = CONFIG_YAML.replace("__TASKFILE__", (tmp_path / "tasks.txt").as_posix())
    cfg = cfg.replace("__ROOT__", (tmp_path / "avt").as_posix())
    cfg = cfg.replace("__N__", str(n))
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    return ContinuousVerifier(load_config(cfg_path), tmp_path)


def _response(pa: dict[str, float], pb: dict[str, float]) -> dict[str, object]:
    def toks(p: dict[str, float]) -> list[dict[str, object]]:
        return [
            {"token": lab, "logprob": math.log(prob) if prob else float("-inf")}
            for lab, prob in p.items()
        ]

    return {
        "choices": [
            {
                "logprobs": {
                    "content": [
                        {"token": "A", "top_logprobs": toks(pa)},
                        {"token": "E", "top_logprobs": toks(pb)},
                    ]
                }
            }
        ]
    }


def _seed_and_pairs(verifier: ContinuousVerifier, scratch: Path, n: int) -> list[str]:
    cids = [candidate_id(verifier.exp, "task_x", i) for i in range(n)]
    with verifier.catalog.connect() as sc:
        sc.upsert_experiment_config(verifier.exp, {}, "VERIFIED")
        sc.record_task(verifier.exp, "task_x", "PUBLIC TASK")
        for i, cid in enumerate(cids):
            sc.record_candidate(cid, verifier.exp, "task_x", i, "SUCCEEDED", None, None)
    return cids


def _add_verification(
    verifier: ContinuousVerifier,
    scratch: Path,
    pid: str,
    ca: str,
    cb: str,
    pa: dict[str, float],
    pb: dict[str, float],
    tag: str,
) -> Path:
    resp = scratch / f"{tag}.json"
    resp.write_text(json.dumps(_response(pa, pb)), encoding="utf-8")
    with verifier.catalog.connect() as sc:
        sc.record_verification(
            f"v_{tag}",
            pid,
            "specification",
            0,
            f"{ca}+{cb}",
            "SUCCEEDED",
            None,
            str(resp),
            None,
        )
    return resp


def test_continuous_verifier_math_and_persistence(tmp_path: Path) -> None:
    verifier = _make(tmp_path, n=2)
    scratch = tmp_path / "responses"
    scratch.mkdir(parents=True)
    ca, cb = _seed_and_pairs(verifier, scratch, 2)
    pid = pair_id(verifier.exp, "task_x", (ca, cb))
    with verifier.catalog.connect() as sc:
        sc.record_pair(pid, verifier.exp, "task_x", ca, cb, "PAIRED")

    # rawA = 0.25*(1+2+3+4) = 2.5 (probs already sum to 1); rawB = E=1.0 -> 5.0
    pa = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25, "E": 0.0}
    pb = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 1.0}
    _add_verification(verifier, scratch, pid, ca, cb, pa, pb, "r1")

    results = verifier.compute()
    by_cand = {r.candidate_id: r for r in results}
    assert len(results) == 2
    assert abs(by_cand[ca].raw_expected_score - 2.5) < 1e-6
    assert abs(by_cand[ca].normalized_score - (2.5 - 1) / 4) < 1e-6
    assert abs(by_cand[cb].raw_expected_score - 5.0) < 1e-6
    assert abs(by_cand[cb].normalized_score - 1.0) < 1e-6
    assert by_cand[ca].observations == 1

    with verifier.catalog.connect() as sc:
        rows = sc.list_expected_scores(ca)
        assert len(rows) == 1
        val = rows[0]["raw_expected_score"]
        assert isinstance(val, (int, float))
        assert abs(val - 2.5) < 1e-6


def test_renormalizes_label_mass(tmp_path: Path) -> None:
    # raw exp masses sum to 0.7, not 1; must be renormalized -> rawA = 2.5
    verifier = _make(tmp_path, n=2)
    scratch = tmp_path / "responses"
    scratch.mkdir(parents=True)
    ca, cb = _seed_and_pairs(verifier, scratch, 2)
    pid = pair_id(verifier.exp, "task_x", (ca, cb))
    with verifier.catalog.connect() as sc:
        sc.record_pair(pid, verifier.exp, "task_x", ca, cb, "PAIRED")
    # weights: A..D = 0.15 each (sum 0.6) + E=0.1 -> total 0.7
    pa = {"A": 0.15, "B": 0.15, "C": 0.15, "D": 0.15, "E": 0.10}
    # normalized A..D = 0.15/0.7, E=0.10/0.7
    # rawA = (1+2)*0.15/0.7 + (3+4)*0.15/0.7 + 5*0.10/0.7
    #      = (0.15*10 + 0.5)/0.7 = 2.0/0.7? let's just assert the formula
    _add_verification(verifier, scratch, pid, ca, cb, pa, pa, "r1")
    (by,) = [r for r in verifier.compute() if r.candidate_id == ca]
    expected = (1 * 0.15 + 2 * 0.15 + 3 * 0.15 + 4 * 0.15 + 5 * 0.10) / 0.7
    assert abs(by.raw_expected_score - expected) < 1e-6


def test_g20_uses_all_labels_in_expectation(tmp_path: Path) -> None:
    """G>5 expectation must use the granularity labels, not the A-E default.

    Regression: the verifier previously computed expected scores with the
    default SCORE_LABELS (A-E) even at G=20, then normalized by (G-1)=19, so
    probability mass on any label above E was silently dropped. This test puts
    all mass on T (the 20th label) and asserts the full G=20 math.
    """
    cfg = CONFIG_YAML.replace("granularity: 5", "granularity: 20")
    (tmp_path / "tasks.txt").write_text("task_x\n", encoding="utf-8")
    cfg = cfg.replace("__TASKFILE__", (tmp_path / "tasks.txt").as_posix())
    cfg = cfg.replace("__ROOT__", (tmp_path / "avt").as_posix())
    cfg_path = tmp_path / "g20.yaml"
    cfg_path.write_text(cfg.replace("__N__", "2"), encoding="utf-8")
    verifier = ContinuousVerifier(load_config(cfg_path), tmp_path)
    scratch = tmp_path / "responses"
    scratch.mkdir(parents=True)
    ca, cb = _seed_and_pairs(verifier, scratch, 2)
    pid = pair_id(verifier.exp, "task_x", (ca, cb))
    with verifier.catalog.connect() as sc:
        sc.record_pair(pid, verifier.exp, "task_x", ca, cb, "PAIRED")
    pa = {chr(ord("A") + i): 0.0 for i in range(20)}
    pa["T"] = 1.0
    _add_verification(verifier, scratch, pid, ca, cb, pa, pa, "g20")
    by = {r.candidate_id: r for r in verifier.compute()}
    # All mass on T (value 20): raw = 20.0, normalized = (20-1)/(20-1) = 1.0.
    assert abs(by[ca].raw_expected_score - 20.0) < 1e-6
    assert abs(by[ca].normalized_score - 1.0) < 1e-6


def test_incomplete_coverage_raises(tmp_path: Path) -> None:
    verifier = _make(tmp_path, n=3)
    scratch = tmp_path / "responses"
    scratch.mkdir(parents=True)
    ca, cb, cc = _seed_and_pairs(verifier, scratch, 3)
    # 3 candidates -> each needs (3-1)*1 = 2 observations/criterion.
    # Only build one pair => coverage failure.
    pid = pair_id(verifier.exp, "task_x", (ca, cb))
    with verifier.catalog.connect() as sc:
        sc.record_pair(pid, verifier.exp, "task_x", ca, cb, "PAIRED")
    pa = {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0}
    _add_verification(verifier, scratch, pid, ca, cb, pa, pa, "r1")
    with pytest.raises(CoverageError):
        verifier.compute()


def test_malformed_response_raises(tmp_path: Path) -> None:
    verifier = _make(tmp_path, n=2)
    scratch = tmp_path / "responses"
    scratch.mkdir(parents=True)
    ca, cb = _seed_and_pairs(verifier, scratch, 2)
    pid = pair_id(verifier.exp, "task_x", (ca, cb))
    with verifier.catalog.connect() as sc:
        sc.record_pair(pid, verifier.exp, "task_x", ca, cb, "PAIRED")
    bad = scratch / "bad.json"
    bad.write_text(
        json.dumps({"choices": [{"logprobs": {"content": [{"token": "x"}]}}]}), encoding="utf-8"
    )
    with verifier.catalog.connect() as sc:
        sc.record_verification(
            "v_bad", pid, "specification", 0, f"{ca}+{cb}", "SUCCEEDED", None, str(bad), None
        )
    with pytest.raises(CoverageError):
        verifier.compute()
