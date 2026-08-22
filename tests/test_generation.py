"""Tests for generation job lifecycle: retry, resume, and idempotence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from avt.config import load_config
from avt.generation import GenerationService, InfrastructureFailure, _HarborRunner
from avt.storage.ids import candidate_id as cid

CONFIG_YAML = """
experiment:
  name: s
  seed: 42
  task_file: __TASKFILE__
  candidates_per_task: 1
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
  endpoint: e
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


def _write_trial(job_dir: Path, reward: float | None, exc: str | None = None) -> None:
    trial_dir = job_dir / "task_x__XXXX"
    trial_dir.mkdir(parents=True, exist_ok=True)
    verifier = {"rewards": {"reward": reward}} if not exc else {}
    data = {
        "verifier_result": verifier,
        "exception_info": (
            {
                "exception_type": exc,
                "exception_message": "boom",
                "exception_traceback": "",
                "occurred_at": "x",
            }
            if exc
            else None
        ),
    }
    (trial_dir / "result.json").write_text(json.dumps(data), encoding="utf-8")


def _make_service(tmp_path: Path) -> tuple[GenerationService, Path]:
    cfg_text = CONFIG_YAML.replace("__TASKFILE__", (tmp_path / "tasks.txt").as_posix()).replace(
        "__ROOT__", (tmp_path / "avt").as_posix()
    )
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(cfg_text, encoding="utf-8")
    (tmp_path / "tasks.txt").write_text("task_x\n", encoding="utf-8")
    cfg = load_config(cfg_path)
    return GenerationService(cfg, tmp_path), tmp_path


def test_fail_then_success_retry_bookkeeping(tmp_path: Path) -> None:
    service, _ = _make_service(tmp_path)
    calls: list[int] = [0]

    def fake_run(config: object, task_id: str, attempt: int, out_dir: Path, job_name: str) -> Path:
        calls[0] += 1
        if calls[0] == 1:
            raise InfrastructureFailure("container startup failed")
        _write_trial(out_dir / job_name, reward=1.0)
        return out_dir / job_name

    service.runner.run = fake_run  # type: ignore[method-assign]
    res = service.generate_one("task_x", 0)
    assert res.reward == 1.0
    with service.catalog.connect() as scoped:
        assert scoped.count_jobs("candidate", "SUCCEEDED") == 1
    assert calls[0] == 2


def test_exhausted_retries_mark_permanent(tmp_path: Path) -> None:
    service, _ = _make_service(tmp_path)

    def fake_run(config: object, task_id: str, attempt: int, out_dir: Path, job_name: str) -> Path:
        raise InfrastructureFailure("always fails")

    service.runner.run = fake_run  # type: ignore[method-assign]
    with pytest.raises(InfrastructureFailure):
        service.generate_one("task_x", 0)
    with service.catalog.connect() as scoped:
        assert scoped.count_jobs("candidate", "PERMANENT_FAILED") == 1


def test_resume_reuses_pre_existing_round(tmp_path: Path) -> None:
    service, root = _make_service(tmp_path)
    round0 = root / "avt/generation/task_x/0/0/gen-task_x-0-0"
    _write_trial(round0, reward=0.0)
    calls: list[int] = [0]

    def fake_run(config: object, task_id: str, attempt: int, out_dir: Path, job_name: str) -> Path:
        calls[0] += 1
        raise AssertionError("should reuse existing round, not run")

    service.runner.run = fake_run  # type: ignore[method-assign]
    res = service.generate_one("task_x", 0)
    assert res.reward == 0.0  # valid failed candidate is graded, not discarded
    assert calls[0] == 0
    with service.catalog.connect() as scoped:
        assert scoped.count_jobs("candidate", "SUCCEEDED") == 1


def test_agent_timeout_trial_is_rejected_and_retried(tmp_path: Path) -> None:
    """A timed-out trial must NOT be indexed as a successful candidate."""
    service, _ = _make_service(tmp_path)
    rounds: list[int] = []

    def fake_run(config: object, task_id: str, attempt: int, out_dir: Path, job_name: str) -> Path:
        round_no = len(rounds)
        rounds.append(round_no)
        if round_no == 0:
            _write_trial(out_dir / job_name, reward=1.0, exc="AgentTimeoutError")
        else:
            _write_trial(out_dir / job_name, reward=1.0)
        return out_dir / job_name

    service.runner.run = fake_run  # type: ignore[method-assign]
    res = service.generate_one("task_x", 0)
    assert res.reward == 1.0
    assert rounds == [0, 1]
    with service.catalog.connect() as scoped:
        assert scoped.count_jobs("candidate", "SUCCEEDED") == 1
        assert scoped.count_jobs("candidate", "PERMANENT_FAILED") == 0


def test_harbor_env_forces_utf8() -> None:
    env = _HarborRunner.harbor_env()
    assert env["PYTHONUTF8"] == "1"
    assert env["PYTHONIOENCODING"] == "utf-8"


def test_generate_all_reclaims_crash_left_running(tmp_path: Path) -> None:
    """A crash-left RUNNING job is reclaimed and completed on a fresh controller start."""
    service, _ = _make_service(tmp_path)
    cand = cid(service.experiment_id(), "task_x", 0)
    with service.catalog.connect() as scoped:
        scoped.enqueue_job("candidate", cand, {"task": "task_x", "attempt": 0})
        assert scoped.claim_job("candidate", cand) is not None  # simulate crash (leave RUNNING)
    calls: list[int] = [0]

    def fake_run(config: object, task_id: str, attempt: int, out_dir: Path, job_name: str) -> Path:
        calls[0] += 1
        _write_trial(out_dir / job_name, reward=1.0)
        return out_dir / job_name

    service.runner.run = fake_run  # type: ignore[method-assign]
    results = service.generate_all()
    assert results[0].reward == 1.0
    with service.catalog.connect() as scoped:
        assert scoped.count_jobs("candidate", "SUCCEEDED") == 1


def test_success_indexes_candidate_manifest(tmp_path: Path) -> None:
    """A graded candidate must be indexed in the frozen-pool manifest for Stage 8."""
    service, _ = _make_service(tmp_path)

    def fake_run(config: object, task_id: str, attempt: int, out_dir: Path, job_name: str) -> Path:
        _write_trial(out_dir / job_name, reward=1.0)
        return out_dir / job_name

    service.runner.run = fake_run  # type: ignore[method-assign]
    res = service.generate_one("task_x", 0)
    assert res.reward == 1.0
    with service.catalog.connect() as scoped:
        row = scoped._conn.execute(
            "SELECT experiment_id, task_id, attempt_index, status, artifact_path FROM candidates"
        ).fetchone()
    assert row is not None, "candidate not indexed in catalog manifest"
    assert row[0] == service.experiment_id()
    assert (row[1], row[2], row[3]) == ("task_x", 0, "SUCCEEDED")
    cand = cid(service.experiment_id(), "task_x", 0)
    assert row[4] is not None and "candidates" in row[4] and row[4].endswith(cand)
    assert Path(row[4]).is_dir()
    with service.catalog.connect() as scoped:
        assert scoped.get_experiment_stage(service.experiment_id()) == "GENERATING"


def test_a_timed_out_trial_is_not_indexed(tmp_path: Path) -> None:
    """A candidate that never grades must not appear in the frozen-pool manifest."""
    service, _ = _make_service(tmp_path)

    def fake_run(config: object, task_id: str, attempt: int, out_dir: Path, job_name: str) -> Path:
        raise InfrastructureFailure("always fails")

    service.runner.run = fake_run  # type: ignore[method-assign]
    with pytest.raises(InfrastructureFailure):
        service.generate_one("task_x", 0)
    with service.catalog.connect() as scoped:
        assert scoped._conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0


def test_ungraded_trial_is_not_indexed_and_retried(tmp_path: Path) -> None:
    """A trial with no reward and no exception must not enter the frozen pool."""
    service, _ = _make_service(tmp_path)
    rounds: list[int] = []

    def fake_run(config: object, task_id: str, attempt: int, out_dir: Path, job_name: str) -> Path:
        rounds.append(len(rounds))
        if len(rounds) == 1:
            _write_trial(out_dir / job_name, reward=None)  # ungraded, no exception
        else:
            _write_trial(out_dir / job_name, reward=0.5)
        return out_dir / job_name

    service.runner.run = fake_run  # type: ignore[method-assign]
    res = service.generate_one("task_x", 0)
    assert res.reward == 0.5
    assert rounds == [0, 1]  # ungraded round retried, not indexed
    with service.catalog.connect() as scoped:
        assert scoped.count_jobs("candidate", "SUCCEEDED") == 1
        n = scoped._conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        assert n == 1  # only the graded retry is in the manifest
