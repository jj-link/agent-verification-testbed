"""Tests for the discrete Qwen judge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from avt import verification as V
from avt.config import load_config
from avt.storage.ids import candidate_id
from avt.verification import (
    DiscreteJudge,
    MalformedVerifier,
    MissingLabelError,
    build_messages,
    expected_scores_from_logprobs,
    normalize_endpoint,
    scores_from_logprobs,
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
  model: m
  endpoint: http://host.docker.internal:8000/v1
  criteria: [specification, output]
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


def _make(tmp_path: Path) -> DiscreteJudge:
    (tmp_path / "tasks.txt").write_text("task_x\n", encoding="utf-8")
    cfg = CONFIG_YAML.replace("__TASKFILE__", (tmp_path / "tasks.txt").as_posix()).replace(
        "__ROOT__", (tmp_path / "avt").as_posix()
    )
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    return DiscreteJudge(load_config(cfg_path), tmp_path)


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


def _top(all_labels: bool, dominant: str, base: float = -2.0) -> list[dict[str, object]]:
    items = []
    for lab in V.SCORE_LABELS:
        lp = base
        if lab == dominant:
            lp = -0.05
        items.append({"token": lab, "logprob": lp})
    if not all_labels:
        items = [i for i in items if i["token"] != "C"]
    return items


def _response(dominant_a: str, dominant_b: str) -> dict[str, object]:
    content = [
        {"token": dominant_a, "top_logprobs": _top(True, dominant_a)},
        {"token": " ", "top_logprobs": [{"token": " ", "logprob": -1.0}]},
        {"token": dominant_b, "top_logprobs": _top(True, dominant_b)},
    ]
    return {
        "choices": [{"logprobs": {"content": content}}],
        "model": "m",
        "id": "x",
    }


def test_normalize_endpoint_maps_host_internal() -> None:
    assert normalize_endpoint("http://host.docker.internal:8000/v1") == "http://127.0.0.1:8000/v1"


def test_build_messages_includes_all_parts_no_grader() -> None:
    msgs = build_messages("TASK DESC", "BODYA", "BODYB", "output")
    joined = json.dumps(msgs)
    assert "BODYA" in joined and "BODYB" in joined and "TASK DESC" in joined
    assert "specification" not in joined or "output" in joined
    # no grader/answer leakage in the template
    assert "reward" not in joined and "pass" not in joined


def test_build_messages_rejects_unknown_criterion() -> None:
    with pytest.raises(ValueError):
        build_messages("t", "a", "b", "nonsense")


def test_scores_from_logprobs_picks_highest_prob_label() -> None:
    content: list[dict[str, object]] = [
        {"token": "C", "top_logprobs": _top(True, "C")},
        {"token": "B", "top_logprobs": _top(True, "B")},
    ]
    assert scores_from_logprobs(content) == (3, 2)


def test_scores_skip_separator_token() -> None:
    content: list[dict[str, object]] = [
        {"token": "E", "top_logprobs": _top(True, "E")},
        {"token": " ", "top_logprobs": []},
        {"token": "A", "top_logprobs": _top(True, "A")},
    ]
    assert scores_from_logprobs(content) == (5, 1)


def test_missing_label_is_configuration_failure() -> None:
    # A score position omitting a configured label is a configuration failure
    # (plan 13.2: never silently assign probability zero).
    content: list[dict[str, object]] = [
        {"token": "B", "top_logprobs": _top(False, "B")},  # omits "C"
        {"token": "B", "top_logprobs": _top(True, "B")},
    ]
    with pytest.raises(MissingLabelError):
        scores_from_logprobs(content)


def test_no_label_logprobs_is_configuration_failure() -> None:
    # A position with no score-label logprob at all is genuine data absence.
    # position 0's token is a label (so it counts as a score position) but its
    # top_logprobs contains no score label -> genuine data absence.
    content: list[dict[str, object]] = [
        {"token": "B", "top_logprobs": [{"token": "word", "logprob": -1.0}]},
        {"token": "B", "top_logprobs": _top(True, "B")},
    ]
    with pytest.raises(MissingLabelError):
        scores_from_logprobs(content)


def test_malformed_verifier_when_fewer_than_two_scores() -> None:
    content: list[dict[str, object]] = [{"token": "A", "top_logprobs": _top(True, "A")}]
    with pytest.raises(MalformedVerifier):
        scores_from_logprobs(content)


def test_judge_verifies_pair_and_persists(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scratch = tmp_path / "scratch"
    judge = _make(tmp_path)
    ca, cb = _seed_pair(judge, scratch)

    # one pair row
    from avt.storage.ids import pair_id

    pid = pair_id(judge.exp, "task_x", (ca, cb))
    with judge.catalog.connect() as scoped:
        scoped.record_pair(pid, judge.exp, "task_x", ca, cb, "PAIRED")
    pair: dict[str, object] = {"pair_id": pid, "candidate_a": ca, "candidate_b": cb}

    def fake_post(url: str, payload: dict[str, object], timeout: int = 120) -> dict[str, object]:
        return _response("C", "B")

    monkeypatch.setattr(V, "_post_json", fake_post)

    # bodies come from fake ATIF files -> provide minimal ones
    def fake_body(cid: str, task: str) -> str:
        return f"BODY_OF_{cid}"

    judge._body_for = fake_body  # type: ignore[assignment]
    ps = judge._verify_pair(pair, "task_x", "specification", 0)
    assert ps.score_a == 3
    assert ps.score_b == 2
    assert ps.display_order in ((ca, cb), (cb, ca))

    with judge.catalog.connect() as scoped:
        rows = scoped.list_pairs(judge.exp, "task_x")
    assert len(rows) == 1
    # verification persisted
    with judge.catalog.connect() as scoped:
        ver = scoped._conn.execute("SELECT COUNT(*) FROM verifications").fetchone()
    assert ver[0] == 1


def test_space_prefixed_label_token_matches() -> None:
    # qwen often emits the second score letter as a leading-space token (" A");
    # it must still be recognized as a label position and score as A (=1).
    content: list[dict[str, object]] = [
        {"token": "A", "top_logprobs": _top(True, "A")},
        {"token": " A", "top_logprobs": _top(True, "A")},
    ]
    assert scores_from_logprobs(content) == (1, 1)


def _toks(probs: dict[str, float]) -> list[dict[str, object]]:
    import math as _m

    return [
        {"token": lab, "logprob": _m.log(prob) if prob else -30.0} for lab, prob in probs.items()
    ]


def test_expected_scores_weighted_sum() -> None:
    pa = {"A": 0.5, "B": 0.5, "C": 0.0, "D": 0.0, "E": 0.0}
    pb = {"A": 0.0, "B": 0.0, "C": 1.0, "D": 0.0, "E": 0.0}
    ca: list[dict[str, object]] = [{"token": "A", "top_logprobs": _toks(pa)}]
    cb: list[dict[str, object]] = [{"token": "C", "top_logprobs": _toks(pb)}]
    ra, rb = expected_scores_from_logprobs(ca + cb)
    assert abs(ra - 1.5) < 1e-6  # 0.5*1 + 0.5*2
    assert abs(rb - 3.0) < 1e-6  # 1.0*3


def _single_pair(builder: DiscreteJudge, scratch: Path, ca: str, cb: str) -> str:
    from avt.storage.ids import pair_id

    pid = pair_id(builder.exp, "task_x", (ca, cb))
    with builder.catalog.connect() as scoped:
        scoped.record_pair(pid, builder.exp, "task_x", ca, cb, "PAIRED")
    return pid


def test_malformed_retries_once_with_reminder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed draw is retried exactly once with the strict-format reminder."""
    scratch = tmp_path / "scratch"
    judge = _make(tmp_path)
    ca, cb = _seed_pair(judge, scratch)
    pid = _single_pair(judge, scratch, ca, cb)
    pair: dict[str, object] = {"pair_id": pid, "candidate_a": ca, "candidate_b": cb}

    calls: list[dict[str, object]] = []

    def fake_post(url: str, payload: dict[str, object], timeout: int = 120) -> dict[str, object]:
        calls.append(payload)
        if len(calls) == 1:
            # malformed: only a single score-token position
            return {
                "choices": [
                    {"logprobs": {"content": [{"token": "A", "top_logprobs": _top(True, "A")}]}}
                ]
            }
        return _response("C", "B")

    monkeypatch.setattr(V, "_post_json", fake_post)
    judge._body_for = lambda cid, task: f"BODY_{cid}"  # type: ignore[assignment]

    ps = judge._verify_pair(pair, "task_x", "specification", 0)
    assert ps.score_a == 3 and ps.score_b == 2
    assert len(calls) == 2  # exactly one malformed retry
    messages = cast(list[dict[str, str]], calls[1]["messages"])
    retry_user = messages[-1]["content"]
    assert V._malformed_reminder(judge._labels) in retry_user
    with judge.catalog.connect() as scoped:
        row = scoped._conn.execute("SELECT malformed_attempts FROM verifications").fetchone()
    assert row[0] == 1


def test_missing_label_fails_immediately_no_persist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing score-token probabilities fail on the first call (plan 15)."""
    scratch = tmp_path / "scratch"
    judge = _make(tmp_path)
    ca, cb = _seed_pair(judge, scratch)
    pid = _single_pair(judge, scratch, ca, cb)
    pair: dict[str, object] = {"pair_id": pid, "candidate_a": ca, "candidate_b": cb}

    calls: list[dict[str, object]] = []

    def fake_post(url: str, payload: dict[str, object], timeout: int = 120) -> dict[str, object]:
        calls.append(payload)
        content = [
            {"token": "B", "top_logprobs": [{"token": "word", "logprob": -1.0}]},
            {"token": "B", "top_logprobs": _top(True, "B")},
        ]
        return {"choices": [{"logprobs": {"content": content}}]}

    monkeypatch.setattr(V, "_post_json", fake_post)
    judge._body_for = lambda cid, task: f"BODY_{cid}"  # type: ignore[assignment]

    with pytest.raises(MissingLabelError):
        judge._verify_pair(pair, "task_x", "specification", 0)
    assert len(calls) == 1  # never retried
    with judge.catalog.connect() as scoped:
        assert scoped._conn.execute("SELECT COUNT(*) FROM verifications").fetchone()[0] == 0


def test_verify_all_resumes_skipping_terminal_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A partial or resumed run makes model calls only for missing keys.

    SUCCEEDED and FAILED verification keys are terminal: a second run must not
    re-call the model for them.
    """
    scratch = tmp_path / "scratch"
    judge = _make(tmp_path)
    ca, cb = _seed_pair(judge, scratch)
    _single_pair(judge, scratch, ca, cb)
    judge._body_for = lambda cid, task: f"BODY_{cid}"  # type: ignore[assignment]

    # Record the criterion name into the user message, not the payload object.
    calls: list[str] = []

    def fake_post(url: str, payload: dict[str, object], timeout: int = 120) -> dict[str, object]:
        msgs = cast(list[dict[str, str]], payload["messages"])
        user = [m["content"] for m in msgs if m["role"] == "user"][0]
        calls.append(str(user))
        return _response("C", "B")

    monkeypatch.setattr(V, "_post_json", fake_post)

    # First run completes both criteria (specification, output => 2 calls).
    judge.verify_all()
    assert len(calls) == 2
    full_run_calls = len(calls)

    # A resumed run over a fully terminal pool adds no model calls.
    judge.verify_all()
    assert len(calls) == full_run_calls


def test_verify_all_resumes_partial_pool_makes_only_missing_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When one criterion is already SUCCEEDED, resume calls only the missing one."""
    scratch = tmp_path / "scratch"
    judge = _make(tmp_path)
    ca, cb = _seed_pair(judge, scratch)
    pid = _single_pair(judge, scratch, ca, cb)
    pair: dict[str, object] = {"pair_id": pid, "candidate_a": ca, "candidate_b": cb}
    judge._body_for = lambda cid, task: f"BODY_{cid}"  # type: ignore[assignment]

    calls: list[str] = []

    def fake_post(url: str, payload: dict[str, object], timeout: int = 120) -> dict[str, object]:
        msgs = cast(list[dict[str, str]], payload["messages"])
        user = [m["content"] for m in msgs if m["role"] == "user"][0]
        calls.append(str(user))
        return _response("C", "B")

    monkeypatch.setattr(V, "_post_json", fake_post)

    # Prior partial run: only 'specification' completed.
    judge._verify_pair(pair, "task_x", "specification", 0)
    assert len(calls) == 1

    judge.verify_all()
    # 'specification' was not re-called; only the missing 'output' was.
    assert len(calls) == 2
    assert "produce the expected output" in calls[-1]


def test_persistent_malformed_records_failed_not_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the single malformed retry, a persistently-malformed response is
    recorded as FAILED with the response artifact, not lost or crashed."""
    scratch = tmp_path / "scratch"
    judge = _make(tmp_path)
    ca, cb = _seed_pair(judge, scratch)
    pid = _single_pair(judge, scratch, ca, cb)
    pair: dict[str, object] = {"pair_id": pid, "candidate_a": ca, "candidate_b": cb}
    judge._body_for = lambda cid, task: f"BODY_{cid}"  # type: ignore[assignment]

    def always_malformed(
        url: str, payload: dict[str, object], timeout: int = 120
    ) -> dict[str, object]:
        return {
            "choices": [
                {"logprobs": {"content": [{"token": "A", "top_logprobs": _top(True, "A")}]}}
            ]
        }

    monkeypatch.setattr(V, "_post_json", always_malformed)

    ps = judge._verify_pair(pair, "task_x", "output", 0)
    assert ps.status == "FAILED"
    assert ps.malformed_attempts == 2  # initial draw plus one retry
    with judge.catalog.connect() as scoped:
        row = scoped._conn.execute(
            "SELECT status, malformed_attempts, response_path FROM verifications"
        ).fetchone()
    assert row[0] == "FAILED"
    assert row[1] == 2
    assert row[2] and Path(str(row[2])).exists()


def test_verify_all_skips_failed_key_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A FAILED (persistently-malformed) key is terminal: resume does not re-call it."""
    scratch = tmp_path / "scratch"
    judge = _make(tmp_path)
    ca, cb = _seed_pair(judge, scratch)
    pid = _single_pair(judge, scratch, ca, cb)
    pair: dict[str, object] = {"pair_id": pid, "candidate_a": ca, "candidate_b": cb}
    judge._body_for = lambda cid, task: f"BODY_{cid}"  # type: ignore[assignment]

    calls: list[str] = []

    def malformed_post(
        url: str, payload: dict[str, object], timeout: int = 120
    ) -> dict[str, object]:
        calls.append("malformed")
        return {
            "choices": [
                {"logprobs": {"content": [{"token": "A", "top_logprobs": _top(True, "A")}]}}
            ]
        }

    monkeypatch.setattr(V, "_post_json", malformed_post)
    failed = judge._verify_pair(pair, "task_x", "output", 0)
    assert failed.status == "FAILED"
    assert len(calls) == 2  # initial draw plus one retry

    # Now the endpoint is well-behaved; a resumed run must call only the
    # missing 'specification' criterion and not re-call the FAILED 'output'.
    def good_post(url: str, payload: dict[str, object], timeout: int = 120) -> dict[str, object]:
        msgs = cast(list[dict[str, str]], payload["messages"])
        user = [m["content"] for m in msgs if m["role"] == "user"][0]
        calls.append(str(user))
        return _response("C", "B")

    monkeypatch.setattr(V, "_post_json", good_post)
    judge.verify_all()
    assert len(calls) == 3  # 2 malformed + 1 specification
    assert "produce the expected output" not in calls[-1]


def test_verify_all_does_not_claim_verified_with_preexisting_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed run that executes no new keys must still not claim VERIFIED when
    a FAILED verification already exists anywhere in the experiment's pairs."""
    scratch = tmp_path / "scratch"
    judge = _make(tmp_path)
    ca, cb = _seed_pair(judge, scratch)
    pid = _single_pair(judge, scratch, ca, cb)
    pair: dict[str, object] = {"pair_id": pid, "candidate_a": ca, "candidate_b": cb}
    judge._body_for = lambda cid, task: f"BODY_{cid}"  # type: ignore[assignment]

    def malformed_post(
        url: str, payload: dict[str, object], timeout: int = 120
    ) -> dict[str, object]:
        return {
            "choices": [
                {"logprobs": {"content": [{"token": "A", "top_logprobs": _top(True, "A")}]}}
            ]
        }

    monkeypatch.setattr(V, "_post_json", malformed_post)
    judge._verify_pair(pair, "task_x", "output", 0)  # records FAILED

    # Now a well-behaved endpoint; a resumed verify_all runs only missing keys.
    def good_post(url: str, payload: dict[str, object], timeout: int = 120) -> dict[str, object]:
        return _response("C", "B")

    monkeypatch.setattr(V, "_post_json", good_post)
    with judge.catalog.connect() as scoped:
        assert scoped.count_failed_verifications(judge.exp) == 1
    judge.verify_all()
    with judge.catalog.connect() as scoped:
        # A pre-existing FAILED still blocks the VERIFIED claim even after resume.
        assert scoped.get_experiment_stage(judge.exp) == "VERIFYING"
        assert scoped.count_failed_verifications(judge.exp) == 1


def test_verifier_max_tokens_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The discrete judge reads max_tokens from config and freezes it in the
    verifier identity, so changing it yields a distinct reproducible run."""
    role_yaml = CONFIG_YAML.replace(
        "\n  granularity: 5\n", "\n  granularity: 5\n  max_tokens: 64\n"
    )
    (tmp_path / "tasks.txt").write_text("task_x\n", encoding="utf-8")
    cfg = role_yaml.replace("__TASKFILE__", (tmp_path / "tasks.txt").as_posix()).replace(
        "__ROOT__", (tmp_path / "avt2").as_posix()
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    judge = DiscreteJudge(load_config(cfg_path), tmp_path)
    assert judge._max_tokens == 64

    scratch = tmp_path / "scratch"
    ca, cb = _seed_pair(judge, scratch)
    pid = _single_pair(judge, scratch, ca, cb)
    pair: dict[str, object] = {"pair_id": pid, "candidate_a": ca, "candidate_b": cb}
    judge._body_for = lambda cid, task: f"BODY_{cid}"  # type: ignore[assignment]

    seen: list[dict[str, object]] = []

    def fake_post(url: str, payload: dict[str, object], timeout: int = 120) -> dict[str, object]:
        seen.append(payload)
        assert payload["max_tokens"] == 64
        return _response("C", "B")

    monkeypatch.setattr(V, "_post_json", fake_post)
    ps = judge._verify_pair(pair, "task_x", "output", 0)
    assert seen
    # identity includes the frozen max_tokens
    assert "max_tokens" in judge._verifier_identity()
    assert judge._verifier_identity()["max_tokens"] == 64
    assert ps.status == "SUCCEEDED"


def test_labels_for_granularity() -> None:
    assert V.labels_for_granularity(5) == ("A", "B", "C", "D", "E")
    assert V.labels_for_granularity(20) == tuple("ABCDEFGHIJKLMNOPQRST")
    assert V.labels_for_granularity(26) == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    with pytest.raises(ValueError):
        V.labels_for_granularity(0)
    with pytest.raises(ValueError):
        V.labels_for_granularity(27)


def _top20(dominant: str, labels: tuple[str, ...]) -> list[dict[str, object]]:
    items = []
    for lab in labels:
        items.append({"token": lab, "logprob": -0.05 if lab == dominant else -2.0})
    return items


def test_scores_from_logprobs_granularity_20() -> None:
    labels = V.labels_for_granularity(20)
    # dominant at position A = 'J' (10th), position B = 'S' (19th)
    content: list[dict[str, object]] = [
        {"token": "J", "top_logprobs": _top20("J", labels)},
        {"token": "S", "top_logprobs": _top20("S", labels)},
    ]
    assert scores_from_logprobs(content, labels) == (10, 19)


def test_judge_uses_granularity_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    yaml = CONFIG_YAML.replace("granularity: 5", "granularity: 20")
    (tmp_path / "tasks.txt").write_text("task_x\n", encoding="utf-8")
    cfg = yaml.replace("__TASKFILE__", (tmp_path / "tasks.txt").as_posix()).replace(
        "__ROOT__", (tmp_path / "avt3").as_posix()
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    judge = DiscreteJudge(load_config(cfg_path), tmp_path)
    assert len(judge._labels) == 20

    scratch = tmp_path / "scratch"
    ca, cb = _seed_pair(judge, scratch)
    pid = _single_pair(judge, scratch, ca, cb)
    pair: dict[str, object] = {"pair_id": pid, "candidate_a": ca, "candidate_b": cb}
    judge._body_for = lambda cid, task: f"BODY_{cid}"  # type: ignore[assignment]

    def fake_post(url: str, payload: dict[str, object], timeout: int = 120) -> dict[str, object]:
        labels = judge._labels
        content = [
            {"token": "K", "top_logprobs": _top20("K", labels)},
            {"token": "P", "top_logprobs": _top20("P", labels)},
        ]
        return {"choices": [{"logprobs": {"content": content}}]}

    monkeypatch.setattr(V, "_post_json", fake_post)
    ps = judge._verify_pair(pair, "task_x", "output", 0)
    assert ps.score_a == 11 and ps.score_b == 16  # K=11, P=16
    assert judge._verifier_identity()["granularity"] == 20
