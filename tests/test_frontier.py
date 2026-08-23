"""Tests for the frontier-assisted two-stage verifier (plan 24)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from avt import verification as V
from avt.config import load_config
from avt.frontier import (
    FrontierClient,
    FrontierNotConfigured,
    SpendCapExceeded,
    build_frontier_messages,
)
from avt.storage.ids import candidate_id
from avt.verification import (
    DiscreteJudge,
    MissingLabelError,
    build_frontier_single_messages,
)

CONFIG_YAML = """
experiment:
  name: s
  seed: 42
  task_file: __TASKFILE__
  candidates_per_task: 2
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
  model: qwen3.8-27b-6000pro
  endpoint: http://host.docker.internal:8000/v1
  criteria: [specification]
  granularity: 20
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

FRONTIER_SECTION = """
frontier:
  model: gpt-4o-mini
  endpoint: http://127.0.0.1:9999/v1
  api_key_env: FRONTIER_TEST_KEY
  spend_cap_usd: 1.0
  pricing:
    input_per_1k: 0.15
    output_per_1k: 0.60
"""


def _seed_pair(builder: DiscreteJudge, scratch: Path) -> tuple[str, str]:
    cids = [candidate_id(builder.exp, "task_x", i) for i in range(2)]
    (scratch / "c").mkdir(parents=True, exist_ok=True)
    with builder.catalog.connect() as scoped:
        scoped.upsert_experiment_config(builder.exp, {}, "PAIRED")
        scoped.record_task(builder.exp, "task_x", "PUBLIC TASK")
        for i, cid in enumerate(cids):
            ap = scratch / "c" / f"c{i}"
            ap.mkdir(parents=True, exist_ok=True)
            (ap / "trajectory.json").write_text(
                json.dumps({"trial_result": str(scratch / f"t{i}" / "result.json")}),
                encoding="utf-8",
            )
            scoped.record_candidate(cid, builder.exp, "task_x", i, "SUCCEEDED", str(ap), "")
    return cids[0], cids[1]


def _top20(dominant: str) -> list[dict[str, object]]:
    items = []
    for lab in V.labels_for_granularity(20):
        items.append({"token": lab, "logprob": -0.05 if lab == dominant else -2.0})
    return items


def _make_judge(tmp_path: Path, frontier_yaml: str) -> DiscreteJudge:
    (tmp_path / "tasks.txt").write_text("task_x\n", encoding="utf-8")
    cfg = CONFIG_YAML.replace("__TASKFILE__", (tmp_path / "tasks.txt").as_posix()).replace(
        "__ROOT__", (tmp_path / "avt").as_posix()
    )
    cfg += "\n" + frontier_yaml
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    return DiscreteJudge(load_config(cfg_path), tmp_path)


# --- FrontierClient unit tests (no network) ---


def test_frontier_not_authorized_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FRONTIER_TEST_KEY", raising=False)
    client = FrontierClient(
        model="gpt-4o-mini",
        endpoint="http://x/v1",
        api_key_env="FRONTIER_TEST_KEY",
        spend_cap_usd=1.0,
        pricing={"input_per_1k": 0.1, "output_per_1k": 0.3},
        env={},
    )
    assert client.authorized() is False
    with pytest.raises(FrontierNotConfigured):
        client.analyze([{"role": "user", "content": "hi"}], max_tokens=10)


def test_frontier_requires_pricing_for_unknown_model(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FrontierClient(
        model="mystery-model",
        endpoint="http://x/v1",
        api_key_env="K",
        spend_cap_usd=1.0,
        env={"K": "secret"},
    )
    assert client.authorized() is False


def test_frontier_spend_cap_enforced_before_call() -> None:
    calls = []

    def fake_post(url: str, payload: dict[str, object], api_key: str) -> dict[str, Any]:
        calls.append(1)
        return {
            "choices": [{"message": {"content": "A: 8\nB: 3"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        }

    client = FrontierClient(
        model="gpt-4o-mini",
        endpoint="http://x/v1",
        api_key_env="K",
        spend_cap_usd=0.001,
        pricing={"input_per_1k": 5.0, "output_per_1k": 10.0},
        post=fake_post,
        env={"K": "secret"},
    )
    # est cost for any write already exceeds the 0.001 cap -> refused, no call.
    with pytest.raises(SpendCapExceeded):
        client.analyze([{"role": "user", "content": "x" * 1000}], max_tokens=200)
    assert calls == []


def test_frontier_analyze_records_latency_cost_usage() -> None:
    def fake_post(url: str, payload: dict[str, object], api_key: str) -> dict[str, Any]:
        return {
            "choices": [{"message": {"content": "A: 9\nB: 2\nREASON: ok"}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 60},
        }

    client = FrontierClient(
        model="gpt-4o-mini",
        endpoint="http://x/v1",
        api_key_env="K",
        spend_cap_usd=10.0,
        pricing={"input_per_1k": 0.15, "output_per_1k": 0.60},
        post=fake_post,
        env={"K": "secret"},
    )
    a = client.analyze(build_frontier_messages("task", "bodyA", "bodyB", "output"), max_tokens=200)
    assert "A: 9" in a.text
    assert a.prompt_tokens == 120 and a.completion_tokens == 60
    # cost = 120/1000*0.15 + 60/1000*0.60 = 0.018 + 0.036 = 0.054
    assert abs(a.cost_usd - 0.054) < 1e-9
    assert a.latency_s >= 0.0
    assert abs(client.ledger.spent - 0.054) < 1e-9


def test_build_frontier_messages_has_no_grader_leakage() -> None:
    msgs = build_frontier_messages("task", "trajA", "trajB", "specification")
    joined = json.dumps(msgs)
    assert "trajA" in joined and "trajB" in joined and "task" in joined
    assert "1-10" in joined
    assert "reward" not in joined and "pass" not in joined and "grader" not in joined


def test_build_frontier_single_conditioned_on_analysis() -> None:
    msgs = build_frontier_single_messages(
        "task", "bodyA", "bodyB", "A: 8 B: 3", "output", "B", V.labels_for_granularity(20)
    )
    joined = json.dumps(msgs)
    assert "THIRD-PARTY REVIEW" in joined
    assert "A: 8 B: 3" in joined
    assert "Score ONLY TRAJECTORY B" in joined
    assert "A=1" in joined and "T=20" in joined


# --- Two-stage judge integration (both posts mocked; no real calls) ---


def test_frontier_assisted_pair_scores_and_records_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = _make_judge(tmp_path, FRONTIER_SECTION)
    scratch = tmp_path / "scratch"
    ca, cb = _seed_pair(judge, scratch)
    pid = _single_pair(judge, scratch, ca, cb)
    pair: dict[str, object] = {"pair_id": pid, "candidate_a": ca, "candidate_b": cb}
    judge._body_for = lambda cid, task: f"BODY_{cid}"  # type: ignore[assignment]

    calls = []

    def fake_verify_all(
        url: str, payload: dict[str, object], timeout: int = 120
    ) -> dict[str, object]:
        # Each local A/B forced call: dominant P (A) and K (B) within G=20.
        user = payload["messages"][-1]["content"]
        dominant = "P" if "Score ONLY TRAJECTORY B" in user else "K"
        assert "THIRD-PARTY REVIEW" in user  # analysis conditioned into prompt
        content = [{"token": dominant, "top_logprobs": _top20(dominant)}]
        calls.append(payload)
        return {"choices": [{"logprobs": {"content": content}}]}

    fl_calls = []

    def fake_frontier_post(url: str, payload: dict[str, object], api_key: str) -> dict[str, Any]:
        fl_calls.append(payload)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "TRAJECTORY A SCORE: 7\nTRAJECTORY B SCORE: 3\nREASONING: balanced"
                        )
                    }
                }
            ],
            "usage": {"prompt_tokens": 200, "completion_tokens": 50},
        }

    # Authorize via env (needed for the real FrontierClient gate) and inject posts.
    monkeypatch.setenv("FRONTIER_TEST_KEY", "secret")
    judge._frontier._post = fake_frontier_post  # type: ignore[attr-defined]
    monkeypatch.setattr(V, "_post_json", fake_verify_all)

    ps = judge._verify_pair(pair, "task_x", "specification", 0)
    assert ps.status == "SUCCEEDED"
    # K=11, P=16 (G=20)
    assert ps.score_a == 11 and ps.score_b == 16
    assert len(fl_calls) == 1  # one frontier call per pair+criterion
    assert len(calls) == 2  # two local forced calls

    # persisted response carries frontier usage + cost into the artifact
    with judge.catalog.connect() as scoped:
        row = scoped._conn.execute(
            "SELECT response_path FROM verifications WHERE verification_id=?", (ps.verification_id,)
        ).fetchone()
    resp = json.loads(Path(str(row[0])).read_text(encoding="utf-8"))
    assert resp["frontier_assisted"] is True
    assert resp["frontier_analysis"].startswith("TRAJECTORY A SCORE")
    assert resp["frontier_usage"]["prompt_tokens"] == 200
    usd = resp["frontier_usage"]["cost_usd"]
    assert usd > 0.0

    identity = judge._verifier_identity()
    assert identity["scoring"] == "frontier_assisted_single_target"
    assert identity["frontier_model"] == "gpt-4o-mini"
    assert "spend_cap_usd" in identity


def test_frontier_judge_refuses_without_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    judge = _make_judge(tmp_path, FRONTIER_SECTION)
    monkeypatch.delenv("FRONTIER_TEST_KEY", raising=False)
    scratch = tmp_path / "scratch"
    ca, cb = _seed_pair(judge, scratch)
    pid = _single_pair(judge, scratch, ca, cb)
    pair: dict[str, object] = {"pair_id": pid, "candidate_a": ca, "candidate_b": cb}
    judge._body_for = lambda cid, task: f"BODY_{cid}"  # type: ignore[assignment]

    def fake_post(url: str, payload: dict[str, object], timeout: int = 120) -> dict[str, object]:
        raise AssertionError("local verifier must not be called without frontier auth")

    monkeypatch.setattr(V, "_post_json", fake_post)
    with pytest.raises(MissingLabelError):
        judge._verify_pair(pair, "task_x", "specification", 0)


def _single_pair(builder: DiscreteJudge, scratch: Path, ca: str, cb: str) -> str:
    from avt.storage.ids import pair_id

    pid = pair_id(builder.exp, "task_x", (ca, cb))
    with builder.catalog.connect() as scoped:
        scoped.record_pair(pid, builder.exp, "task_x", ca, cb, "PAIRED")
    return pid
