"""Tests for round-robin Bradley-Terry ranking (plan 14)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avt.config import load_config
from avt.ranking import RoundRobinRanker, _sigmoid
from avt.storage.ids import candidate_id

CONFIG_YAML = """
experiment:
  name: s
  seed: 42
  task_file: __TASKFILE__
  candidates_per_task: 3
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
  minimum_pair_coverage: 1.0
storage:
  root: __ROOT__
  metadata_db: __ROOT__/experiment.sqlite
  ground_truth_db: __ROOT__/ground_truth.sqlite
"""


def _make(tmp_path: Path) -> RoundRobinRanker:
    (tmp_path / "tasks.txt").write_text("task_x\n", encoding="utf-8")
    cfg = CONFIG_YAML.replace("__TASKFILE__", (tmp_path / "tasks.txt").as_posix())
    cfg = cfg.replace("__ROOT__", (tmp_path / "avt").as_posix())
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    return RoundRobinRanker(load_config(cfg_path), tmp_path)


def _pair_id(a: str, b: str) -> str:
    lo, hi = (a, b) if a < b else (b, a)
    return f"pair-{lo[:8]}-{hi[:8]}"


def _seed(
    ranker: RoundRobinRanker,
    n: int,
    raw_scores: list[float],
    *,
    malformed: dict[str, int] | None = None,
    drop_verification: bool = False,
    wrong_criterion: bool = False,
    duplicate_verification: bool = False,
) -> list[str]:
    """Seed a full task pool: candidates, evaluations, pairs, verifications.

    Mirrors the Stage 8-10 pipeline: one SUCCEEDED pair per unordered
    combination and one SUCCEEDED verification per pair (specification, rep 0),
    with per-pair malformed counts shared by both members.
    """
    malformed = malformed or {}
    cids = [candidate_id(ranker.exp, "task_x", i) for i in range(n)]
    with ranker.catalog.connect() as sc:
        sc.upsert_experiment_config(ranker.exp, {}, "VERIFIED")
        sc.record_task(ranker.exp, "task_x", "PUBLIC TASK")
        for i, cid in enumerate(cids):
            sc.record_candidate(cid, ranker.exp, "task_x", i, "SUCCEEDED", None, None)
            raw = raw_scores[i]
            sc.record_evaluation(cid, raw, (raw - 1) / 4, "specification", 2)
        for a in range(n):
            for b in range(a + 1, n):
                pid = _pair_id(cids[a], cids[b])
                sc.record_pair(pid, ranker.exp, "task_x", cids[a], cids[b], "SUCCEEDED")
                if drop_verification:
                    continue
                m = malformed.get(cids[a], 0) + malformed.get(cids[b], 0)
                criterion = "output" if wrong_criterion else "specification"
                sc.record_verification(
                    f"ver-{pid}",
                    pid,
                    criterion,
                    0,
                    f"{cids[a]}+{cids[b]}",
                    "SUCCEEDED",
                    None,
                    None,
                    None,
                    m,
                )
                if duplicate_verification:
                    sc.record_verification(
                        f"ver-{pid}-dup",
                        pid,
                        "specification",
                        0,
                        f"{cids[a]}+{cids[b]}",
                        "SUCCEEDED",
                        None,
                        None,
                        None,
                        m,
                    )
    return cids


def test_sigmoid() -> None:
    assert abs(_sigmoid(0.0) - 0.5) < 1e-9
    assert _sigmoid(10) > 0.999


def test_highest_aggregate_ranks_first(tmp_path: Path) -> None:
    ranker = _make(tmp_path)
    cids = _seed(ranker, 3, [5.0, 4.0, 3.0])

    rec = ranker._rank_task("task_x")
    assert rec.ranking[0].candidate_id == cids[0]  # raw 5.0 wins
    by_id = {r.candidate_id: r for r in rec.ranking}
    assert by_id[cids[0]].utility > by_id[cids[1]].utility > by_id[cids[2]].utility
    # ranking persisted immutably
    with ranker.catalog.connect() as sc:
        assert sc._conn.execute("SELECT COUNT(*) FROM rankings").fetchone()[0] == 1


def test_rank_all_no_grader_consulted(tmp_path: Path) -> None:
    ranker = _make(tmp_path)
    cids = _seed(ranker, 2, [3.0, 4.0])
    rec = ranker.rank_all()
    assert len(rec) == 1
    assert len(rec[0].ranking) == 2
    # candidate with higher expected score wins
    assert rec[0].ranking[0].candidate_id == cids[1]


def test_tie_breaks_by_lower_candidate_id(tmp_path: Path) -> None:
    """Identical scores/utilities and equal malformed -> lower id ranks better."""
    ranker = _make(tmp_path)
    cids = _seed(ranker, 3, [4.0, 4.0, 4.0])

    rec = ranker._rank_task("task_x")
    assert [r.candidate_id for r in rec.ranking] == sorted(cids)


def test_tie_breaks_by_fewer_malformed(tmp_path: Path) -> None:
    """Identical scores/utilities -> the candidate in cleaner pairs ranks better.

    Per-candidate malformed A=2, B=2, C=0 give C a lower total malformed count
    than A and B, so C outranks them even though its id is not lowest (plan 14
    tie-break 2 beats tie-break 3).
    """
    ranker = _make(tmp_path)
    cids = [candidate_id(ranker.exp, "task_x", i) for i in range(3)]
    _seed(
        ranker,
        3,
        [4.0, 4.0, 4.0],
        malformed={cids[0]: 2, cids[1]: 2, cids[2]: 0},
    )

    rec = ranker._rank_task("task_x")
    assert rec.ranking[0].candidate_id == cids[2]  # fewest malformed wins


def test_ranking_fails_without_full_pair_coverage(tmp_path: Path) -> None:
    """A missing verification fails the ranking visibly (plan 14 coverage)."""
    ranker = _make(tmp_path)
    _seed(ranker, 3, [4.0, 4.0, 4.0], drop_verification=True)

    with pytest.raises(ValueError):
        ranker._rank_task("task_x")


def test_ranking_fails_on_wrong_criterion(tmp_path: Path) -> None:
    """A wrong criterion slot is not masked by a matching total count."""
    ranker = _make(tmp_path)
    _seed(ranker, 2, [3.0, 4.0], wrong_criterion=True)

    with pytest.raises(ValueError):
        ranker._rank_task("task_x")


def test_ranking_fails_on_duplicate_verification(tmp_path: Path) -> None:
    """Duplicate rows do not satisfy the exact (criterion, repetition) set."""
    ranker = _make(tmp_path)
    _seed(ranker, 2, [3.0, 4.0], duplicate_verification=True)

    with pytest.raises(ValueError):
        ranker._rank_task("task_x")


def test_ranking_conflicting_rerun_raises(tmp_path: Path) -> None:
    """A persisted ranking rejects a conflicting rewrite (immutable record)."""
    ranker = _make(tmp_path)
    _seed(ranker, 2, [3.0, 4.0])
    rec = ranker._rank_task("task_x")
    selector = json.dumps(rec.selector_config, sort_keys=True)
    with pytest.raises(ValueError), ranker.catalog.connect() as sc:
        sc.record_ranking(
            rec.ranking_id,
            "task_x",
            rec.pool_hash,
            selector,
            "DIFFERENT_RESULT",
            "SUCCEEDED",
        )


def test_no_ranking_without_aggregate(tmp_path: Path) -> None:
    """A candidate with no aggregate expected score fails the ranking visibly."""
    ranker = _make(tmp_path)
    cids = [candidate_id(ranker.exp, "task_x", i) for i in range(2)]
    with ranker.catalog.connect() as sc:
        sc.upsert_experiment_config(ranker.exp, {}, "VERIFIED")
        sc.record_task(ranker.exp, "task_x", "PUBLIC TASK")
        for cid in cids:
            sc.record_candidate(cid, ranker.exp, "task_x", 0, "SUCCEEDED", None, None)
        # no record_evaluation call -> aggregate missing

    with pytest.raises(ValueError):
        ranker._rank_task("task_x")
