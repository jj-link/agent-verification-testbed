"""Leakage-safe local selectors over frozen candidate pools.

The main study compares three selectors on identical candidate IDs:

* deterministic random baseline;
* round-robin Bradley-Terry over discrete argmax verifier scores; and
* round-robin Bradley-Terry over continuous expected verifier scores.

The score-based selectors use plan section 14:

    p_ij = sigmoid(R_i - R_j)
    u_i  = mean_{j != i} p_ij
    i*   = argmax_i u_i

Tie-breaking is higher mean score, fewer malformed verifier records, then the
lexicographically lower deterministic candidate ID. No selector opens the
ground-truth store.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from avt.config import Config
from avt.selection import read_task_names
from avt.storage.catalog import Catalog
from avt.storage.ids import experiment_id, ranking_id, stable_hash

__all__ = ["RoundRobinRanker", "RankedCandidate", "RankingRecord"]

_METHOD = "round_robin_bradley_terry"
_SELECTORS = frozenset({"random", "discrete", "continuous"})


@dataclass(frozen=True)
class RankedCandidate:
    candidate_id: str
    score: float | None
    utility: float | None
    rank: int


@dataclass(frozen=True)
class RankingRecord:
    ranking_id: str
    task_id: str
    pool_hash: str
    selector_config: dict[str, object]
    ranking: tuple[RankedCandidate, ...]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class RoundRobinRanker:
    def __init__(self, config: Config, repo_root: Path, selector: str = "continuous") -> None:
        if selector not in _SELECTORS:
            raise ValueError(f"unknown local selector {selector!r}; expected {sorted(_SELECTORS)}")
        self.config = config
        self.repo_root = repo_root
        self.catalog = Catalog(Path(config.storage.metadata_db))
        self.exp = experiment_id(config.raw)
        self.selector = selector

    def _continuous_scores(self, ids: list[str]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for candidate_id_ in ids:
            with self.catalog.connect() as scoped:
                row = scoped.list_evaluation(candidate_id_)
            value = row.get("aggregate_raw") if row is not None else None
            if not isinstance(value, (int, float)):
                raise ValueError(f"{candidate_id_}: no aggregate expected score")
            scores[candidate_id_] = float(value)
        return scores

    def _discrete_scores(
        self,
        task_id: str,
        ids: list[str],
        pairs: list[dict[str, object]],
    ) -> dict[str, float]:
        pair_ids = {str(pair["pair_id"]) for pair in pairs}
        with self.catalog.connect() as scoped:
            verifications = [
                row
                for row in scoped.list_verifications()
                if str(row["pair_id"]) in pair_ids and row["status"] == "SUCCEEDED"
            ]

        values: dict[str, list[float]] = {candidate_id_: [] for candidate_id_ in ids}
        granularity = int(self.config.verifier.granularity)
        for row in verifications:
            scores_path = row.get("scores_path")
            if not scores_path:
                raise ValueError(
                    f"verification {row['verification_id']}: no discrete score artifact"
                )
            try:
                artifact = json.loads(Path(str(scores_path)).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"verification {row['verification_id']}: unusable discrete "
                    f"score artifact ({exc})"
                ) from exc
            display_order = str(row.get("display_order") or "")
            if "+" not in display_order:
                raise ValueError(f"verification {row['verification_id']}: bad display_order")
            candidate_a, candidate_b = display_order.split("+", 1)
            for candidate_id_, key in ((candidate_a, "score_a"), (candidate_b, "score_b")):
                if candidate_id_ not in values:
                    raise ValueError(
                        f"{task_id}: verification {row['verification_id']} references candidate "
                        f"outside pool: {candidate_id_}"
                    )
                score = artifact.get(key) if isinstance(artifact, dict) else None
                if isinstance(score, bool) or not isinstance(score, int):
                    raise ValueError(
                        f"verification {row['verification_id']}: {key} is not an integer score"
                    )
                if not 1 <= score <= granularity:
                    raise ValueError(
                        f"verification {row['verification_id']}: {key}={score} outside "
                        f"1..{granularity}"
                    )
                values[candidate_id_].append(float(score))

        expected = (
            (len(ids) - 1)
            * len(tuple(self.config.verifier.criteria))
            * int(self.config.verifier.repetitions)
        )
        scores: dict[str, float] = {}
        for candidate_id_, observations in values.items():
            if len(observations) != expected:
                raise ValueError(
                    f"{candidate_id_}: {len(observations)} discrete observations, "
                    f"expected {expected}"
                )
            scores[candidate_id_] = sum(observations) / len(observations)
        return scores

    def _check_pair_coverage(self, task_id: str, ids: list[str]) -> list[dict[str, object]]:
        """Require one usable verifier result for every configured pair/key."""
        criteria = tuple(self.config.verifier.criteria)
        repetitions = int(self.config.verifier.repetitions)
        expected_keys = {
            (criterion, repetition) for criterion in criteria for repetition in range(repetitions)
        }
        required = {(a, b) for a in ids for b in ids if a < b}
        with self.catalog.connect() as scoped:
            pairs = scoped.list_pairs(self.exp, task_id)
            present: dict[tuple[str, str], dict[str, object]] = {}
            for pair in pairs:
                candidate_a = str(pair["candidate_a"])
                candidate_b = str(pair["candidate_b"])
                ordered = (
                    (candidate_a, candidate_b)
                    if candidate_a < candidate_b
                    else (candidate_b, candidate_a)
                )
                present[ordered] = pair
                if ordered not in required:
                    raise ValueError(f"{task_id}: pair {pair['pair_id']} outside pool")
                keys = scoped.succeeded_verification_keys(str(pair["pair_id"]))
                count = scoped.count_succeeded_verifications(str(pair["pair_id"]))
                if keys != expected_keys or count != len(expected_keys):
                    raise ValueError(
                        f"{task_id}: pair {pair['pair_id']} SUCCEEDED "
                        f"(criterion,repetition) coverage {sorted(keys)} ({count} rows), "
                        f"expected {sorted(expected_keys)}"
                    )
        missing = required - set(present)
        if missing:
            raise ValueError(f"{task_id}: missing pairs {sorted(missing)}")
        return pairs

    def _selector_config(self) -> dict[str, object]:
        if self.selector == "random":
            return {
                "selector": "random",
                "method": "deterministic_hash_order",
                "seed": int(self.config.experiment.seed),
            }
        coverage = self.config.ranking.get("minimum_pair_coverage", 1.0)
        return {
            "selector": self.selector,
            "method": _METHOD,
            "scoring": (
                "discrete_argmax" if self.selector == "discrete" else "continuous_expectation"
            ),
            "minimum_pair_coverage": float(coverage if isinstance(coverage, (int, float)) else 1.0),
        }

    def _rank_task(self, task_id: str) -> RankingRecord:
        with self.catalog.connect() as scoped:
            candidates = scoped.list_candidates(self.exp, task_id)
        ids = sorted(str(candidate["candidate_id"]) for candidate in candidates)
        if len(ids) < 2:
            raise ValueError(f"{task_id}: pool too small to rank ({len(ids)})")

        pool_hash = stable_hash(sorted(ids))
        selector = self._selector_config()
        if self.selector == "random":
            seed = int(self.config.experiment.seed)
            ordered = sorted(
                ids,
                key=lambda candidate_id_: (
                    stable_hash("random-selector", seed, task_id, pool_hash, candidate_id_),
                    candidate_id_,
                ),
            )
            scores: Mapping[str, float | None] = dict.fromkeys(ids)
            utilities: Mapping[str, float | None] = dict.fromkeys(ids)
        else:
            pairs = self._check_pair_coverage(task_id, ids)
            numeric_scores = (
                self._discrete_scores(task_id, ids, pairs)
                if self.selector == "discrete"
                else self._continuous_scores(ids)
            )
            with self.catalog.connect() as scoped:
                malformed = {
                    candidate_id_: scoped.malformed_attempts_for(candidate_id_)
                    for candidate_id_ in ids
                }
            numeric_utilities: dict[str, float] = {}
            for candidate_id_ in ids:
                others = [other for other in ids if other != candidate_id_]
                numeric_utilities[candidate_id_] = sum(
                    _sigmoid(numeric_scores[candidate_id_] - numeric_scores[other])
                    for other in others
                ) / len(others)
            ordered = sorted(
                ids,
                key=lambda candidate_id_: (
                    -numeric_utilities[candidate_id_],
                    -numeric_scores[candidate_id_],
                    malformed[candidate_id_],
                    candidate_id_,
                ),
            )
            scores = numeric_scores
            utilities = numeric_utilities

        rid = ranking_id(task_id, pool_hash, selector)
        ranking = tuple(
            RankedCandidate(candidate_id_, scores[candidate_id_], utilities[candidate_id_], rank)
            for rank, candidate_id_ in enumerate(ordered, start=1)
        )
        result_json = json.dumps(
            {
                "task_id": task_id,
                "pool_hash": pool_hash,
                "selector": selector,
                "ranking": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "score": candidate.score,
                        "utility": candidate.utility,
                        "rank": candidate.rank,
                    }
                    for candidate in ranking
                ],
            },
            sort_keys=True,
        )
        with self.catalog.connect() as scoped:
            scoped.record_ranking(
                rid,
                task_id,
                pool_hash,
                json.dumps(selector, sort_keys=True),
                result_json,
                "SUCCEEDED",
            )
        return RankingRecord(rid, task_id, pool_hash, selector, ranking)

    def rank_all(self) -> list[RankingRecord]:
        self._validate_no_grader()
        return [self._rank_task(task) for task in read_task_names(self.config.experiment.task_file)]

    def _validate_no_grader(self) -> None:
        """Guard: ranking never opens the separate ground-truth database."""
        with self.catalog.connect() as scoped:
            _ = scoped
