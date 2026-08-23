"""Three-criterion evaluation (plan sections 13.4-13.5, 11).

Aggregates each candidate's per-criterion expected scores (Stage 10) into a
single reproducible score across the configured criteria:

    aggregate_raw(tau) = (1/C) * sum_c raw_expected_score_{c}(tau)
    normalized       = (aggregate_raw - 1) / (G - 1)

Requires complete criterion coverage for every frozen candidate; a missing
criterion for any candidate fails the evaluation visibly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from avt.config import Config
from avt.doctor import G5_LABELS
from avt.selection import read_task_names
from avt.storage.catalog import Catalog
from avt.storage.ids import experiment_id

__all__ = ["Evaluator", "EvaluationError", "EvaluatedCandidate"]

_G = len(G5_LABELS)


@dataclass(frozen=True)
class EvaluatedCandidate:
    candidate_id: str
    criteria: tuple[str, ...]
    aggregate_raw: float
    aggregate_normalized: float
    observations: int


class EvaluationError(Exception):
    """A required criterion score is missing for a frozen candidate."""


def _as_float(value: Any) -> float:
    if not isinstance(value, (int, float)):
        raise EvaluationError(f"expected numeric score, got {value!r}")
    return float(value)


def _as_int(value: Any) -> int:
    if not isinstance(value, int):
        raise EvaluationError(f"expected integer observations, got {value!r}")
    return value


class Evaluator:
    def __init__(self, config: Config, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self.catalog = Catalog(Path(config.storage.metadata_db))
        self.exp = experiment_id(config.raw)
        self._criteria = tuple(self.config.verifier.criteria)
        self._G = int(config.verifier.granularity)

    def _candidates(self) -> list[str]:
        ids: list[str] = []
        for task in read_task_names(self.config.experiment.task_file):
            with self.catalog.connect() as scoped:
                for c in scoped.list_candidates(self.exp, task):
                    ids.append(str(c["candidate_id"]))
        return sorted(set(ids))

    def evaluate(self) -> list[EvaluatedCandidate]:
        results: list[EvaluatedCandidate] = []
        for candidate_id_ in self._candidates():
            with self.catalog.connect() as scoped:
                expected = scoped.list_expected_scores(candidate_id_)
            by_criterion = {
                str(r["criterion"]): _as_float(r["raw_expected_score"]) for r in expected
            }
            missing = [c for c in self._criteria if c not in by_criterion]
            if missing:
                raise EvaluationError(f"{candidate_id_}: missing criterion scores {missing}")
            n = len(self._criteria)
            raw = sum(by_criterion[c] for c in self._criteria) / n
            norm = (raw - 1.0) / (self._G - 1.0)
            observations = sum(_as_int(r.get("observations")) for r in expected)
            record = EvaluatedCandidate(candidate_id_, self._criteria, raw, norm, observations)
            results.append(record)
            with self.catalog.connect() as scoped:
                scoped.record_evaluation(
                    candidate_id_, raw, norm, ",".join(self._criteria), observations
                )
        return results
