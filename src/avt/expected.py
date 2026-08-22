"""Continuous Qwen verifier (plan section 13.4).

Computes each candidate's expected score from the label probability
distributions already captured in the stored discrete-judge responses (one
request per (pair, criterion, repetition)):

    raw_expected_score_{c,k}(tau) = sum_g p(score_g | x, c, tau) * value(score_g)

where ``p`` is the renormalized probability over the G score labels. Values are
aggregated per (candidate, criterion) as the mean over the candidate's pairwise
observations, and normalized to [0, 1]:

    normalized_score = (raw_expected_score - 1) / (G - 1)

This is a deterministic, offline pass over immutable verification response
artifacts (no new model calls). A malformed or missing response, or an
incomplete observation set for any (candidate, criterion), fails the job
visibly rather than silently degrading coverage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from avt.config import Config
from avt.doctor import G5_LABELS
from avt.selection import read_task_names
from avt.storage.catalog import Catalog
from avt.storage.ids import experiment_id
from avt.verification import expected_scores_from_logprobs

__all__ = ["ContinuousVerifier", "CoverageError", "ExpectedScore"]

_G = len(G5_LABELS)  # 5


@dataclass(frozen=True)
class ExpectedScore:
    candidate_id: str
    criterion: str
    raw_expected_score: float
    normalized_score: float
    observations: int


class CoverageError(Exception):
    """A required expected-score observation is missing or unusable."""


def _content_logprobs(response: dict[str, Any]) -> list[dict[str, Any]]:
    choices = response.get("choices") or [{}]
    lp = choices[0].get("logprobs") or {} if isinstance(choices[0], dict) else {}
    return list(lp.get("content") or [])


class ContinuousVerifier:
    def __init__(self, config: Config, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self.catalog = Catalog(Path(config.storage.metadata_db))
        self.exp = experiment_id(config.raw)
        self._G = len(G5_LABELS)

    def _candidate_tasks(self) -> tuple[dict[str, str], dict[str, int]]:
        """Map candidate_id -> task_id and pool size per task."""
        cand_task: dict[str, str] = {}
        pool_size: dict[str, int] = {}
        for task in read_task_names(self.config.experiment.task_file):
            with self.catalog.connect() as scoped:
                cands = scoped.list_candidates(self.exp, task)
            pool_size[task] = len(cands)
            for c in cands:
                cand_task[str(c["candidate_id"])] = task
        return cand_task, pool_size

    def compute(self) -> list[ExpectedScore]:
        cand_task, pool_size = self._candidate_tasks()
        with self.catalog.connect() as scoped:
            verifications = scoped.list_verifications()

        agg: dict[tuple[str, str], list[float]] = {}
        for vr in verifications:
            response_path = vr.get("response_path")
            if not response_path:
                raise CoverageError(
                f"verification missing response artifact: {vr.get('verification_id')}"
            )
            try:
                response = json.loads(Path(str(response_path)).read_text(encoding="utf-8"))
                raw_a, raw_b = expected_scores_from_logprobs(_content_logprobs(response))
            except Exception as exc:
                raise CoverageError(
                    f"verification {vr.get('verification_id')}: unusable response ({exc})"
                ) from exc
            disp = str(vr.get("display_order") or "")
            if "+" not in disp:
                raise CoverageError(f"verification {vr.get('verification_id')}: bad display_order")
            cand_a, cand_b = disp.split("+", 1)
            criterion = str(vr.get("criterion") or "")
            agg.setdefault((cand_a, criterion), []).append(raw_a)
            agg.setdefault((cand_b, criterion), []).append(raw_b)

        repetitions = self.config.verifier.repetitions
        criteria = tuple(self.config.verifier.criteria)
        results: list[ExpectedScore] = []
        for candidate_id, task in sorted(cand_task.items()):
            expected = max(0, pool_size[task] - 1) * repetitions
            for criterion in criteria:
                values = agg.get((candidate_id, criterion), [])
                if len(values) != expected:
                    raise CoverageError(
                        f"{candidate_id}/{criterion}: {len(values)} observations, "
                        f"expected {expected}"
                    )
                if expected == 0:
                    continue
                raw = sum(values) / len(values)
                norm = (raw - 1.0) / (self._G - 1.0)
                results.append(ExpectedScore(candidate_id, criterion, raw, norm, len(values)))
                with self.catalog.connect() as scoped:
                    scoped.record_expected_score(
                        candidate_id, criterion, raw, norm, len(values)
                    )
        return results
