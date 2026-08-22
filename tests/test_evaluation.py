"""Tests for three-criterion evaluation (plan 13.4/11)."""

from __future__ import annotations

from pathlib import Path

import pytest

from avt.config import load_config
from avt.evaluation import EvaluationError, Evaluator
from avt.storage.ids import candidate_id

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
  endpoint: http://host.docker.internal:8000/v1
  criteria: [specification, output, errors]
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


def _make(tmp_path: Path) -> Evaluator:
    (tmp_path / "tasks.txt").write_text("task_x\n", encoding="utf-8")
    cfg = CONFIG_YAML.replace("__TASKFILE__", (tmp_path / "tasks.txt").as_posix())
    cfg = cfg.replace("__ROOT__", (tmp_path / "avt").as_posix())
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    return Evaluator(load_config(cfg_path), tmp_path)


def _seed_with_scores(evaluator: Evaluator, scores: dict[str, float]) -> str:
    cid = candidate_id(evaluator.exp, "task_x", 0)
    with evaluator.catalog.connect() as sc:
        sc.upsert_experiment_config(evaluator.exp, {}, "VERIFIED")
        sc.record_task(evaluator.exp, "task_x", "PUBLIC TASK")
        sc.record_candidate(cid, evaluator.exp, "task_x", 0, "SUCCEEDED", None, None)
        for criterion, raw in scores.items():
            sc.record_expected_score(cid, criterion, raw, (raw - 1) / 4, 2)
    return cid


def test_aggregate_over_three_criteria(tmp_path: Path) -> None:
    evaluator = _make(tmp_path)
    _seed_with_scores(evaluator, {"specification": 3.0, "output": 5.0, "errors": 1.0})
    results = evaluator.evaluate()
    assert len(results) == 1
    rec = results[0]
    assert abs(rec.aggregate_raw - 3.0) < 1e-6  # mean(3,5,1)
    assert abs(rec.aggregate_normalized - (3.0 - 1) / 4) < 1e-6
    assert rec.observations == 6


def test_aggregation_is_reproducible(tmp_path: Path) -> None:
    evaluator = _make(tmp_path)
    scores = {"specification": 2.0, "output": 4.0, "errors": 3.0}
    cid = _seed_with_scores(evaluator, scores)
    first = evaluator.evaluate()
    second = evaluator.evaluate()  # idempotent rerun
    assert first == second
    with evaluator.catalog.connect() as sc:
        stored = sc.list_evaluation(cid)
        assert stored is not None
        agg = stored["aggregate_raw"]
        assert isinstance(agg, (int, float)) and abs(agg - 3.0) < 1e-6


def test_missing_criterion_raises(tmp_path: Path) -> None:
    evaluator = _make(tmp_path)
    _seed_with_scores(evaluator, {"specification": 3.0, "output": 2.0})  # no 'errors'
    with pytest.raises(EvaluationError):
        evaluator.evaluate()
