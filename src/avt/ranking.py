"""Round-robin Bradley-Terry ranking (plan section 14).

For each frozen task pool, ranks candidates by the round-robin win probability
derived only from the Stage-11 verifier aggregate expected scores (never the
official grader):

    p_ij = sigmoid(R_i - R_j)          (R = aggregate expected score)
    u_i  = mean_{j != i} p_ij
    i*   = argmax_i u_i

Tie-breaking (in order): higher mean expected score, fewer malformed verifier
records, lexicographically lower deterministic candidate id. Ranking records
are persisted immutably.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from avt.config import Config
from avt.selection import read_task_names
from avt.storage.catalog import Catalog
from avt.storage.ids import experiment_id, ranking_id, stable_hash

__all__ = ["RoundRobinRanker", "RankedCandidate", "RankingRecord"]

_METHOD = "round_robin_bradley_terry"


@dataclass(frozen=True)
class RankedCandidate:
    candidate_id: str
    expected_score: float
    utility: float
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
    def __init__(self, config: Config, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self.catalog = Catalog(Path(config.storage.metadata_db))
        self.exp = experiment_id(config.raw)

    def _aggregate(self, candidate_id_: str) -> float:
        with self.catalog.connect() as scoped:
            row = scoped.list_evaluation(candidate_id_)
        value = row.get("aggregate_raw") if row is not None else None
        if not isinstance(value, (int, float)):
            raise ValueError(f"{candidate_id_}: no aggregate expected score")
        return float(value)

    def _check_pair_coverage(self, task_id: str, ids: list[str]) -> None:
        """Enforce 100% usable pair coverage (plan 14) before ranking.

        Every unordered pair in the pool must be present and carry a SUCCEEDED
        verification for every (criterion, repetition). A pair's own ``status``
        is its freeze state (e.g. PAIRED), not the usable signal. Missing
        coverage fails the ranking job visibly rather than silently changing
        the method.
        """
        criteria = tuple(self.config.verifier.criteria)
        repetitions = int(self.config.verifier.repetitions)
        expected_keys = {(c, r) for c in criteria for r in range(repetitions)}
        required = {(a, b) for a in ids for b in ids if a < b}
        with self.catalog.connect() as scoped:
            pairs = scoped.list_pairs(self.exp, task_id)
            present: dict[tuple[str, str], dict[str, object]] = {}
            for pa in pairs:
                ca, cb = str(pa["candidate_a"]), str(pa["candidate_b"])
                ordered = (ca, cb) if ca < cb else (cb, ca)
                present[ordered] = pa
                if ordered not in required:
                    raise ValueError(f"{task_id}: pair {pa['pair_id']} outside pool")
                keys = scoped.succeeded_verification_keys(str(pa["pair_id"]))
                n = scoped.count_succeeded_verifications(str(pa["pair_id"]))
                if keys != expected_keys or n != len(expected_keys):
                    raise ValueError(
                        f"{task_id}: pair {pa['pair_id']} SUCCEEDED "
                        f"(criterion,repetition) coverage {sorted(keys)} ({n} rows), "
                        f"expected {sorted(expected_keys)}"
                    )
        missing = required - set(present)
        if missing:
            raise ValueError(f"{task_id}: missing pairs {sorted(missing)}")

    def _rank_task(self, task_id: str) -> RankingRecord:
        with self.catalog.connect() as scoped:
            cands = scoped.list_candidates(self.exp, task_id)
        ids = sorted(str(c["candidate_id"]) for c in cands)
        if len(ids) < 2:
            raise ValueError(f"{task_id}: pool too small to rank ({len(ids)})")
        scores = {cid: self._aggregate(cid) for cid in ids}
        self._check_pair_coverage(task_id, ids)
        with self.catalog.connect() as scoped:
            malformed = {cid: scoped.malformed_attempts_for(cid) for cid in ids}
        utilities: dict[str, float] = {}
        for i in ids:
            others = [j for j in ids if j != i]
            utilities[i] = sum(_sigmoid(scores[i] - scores[j]) for j in others) / len(others)

        # Stable ascending sort; negation inverts the numeric "higher is better"
        # columns so fewer malformed verifier records and a lower deterministic id
        # still rank better (plan 14 tie-breaking).
        def key(cid: str) -> tuple[float, float, int, str]:
            # higher utility, higher expected score, fewer malformed, lower id
            return (-utilities[cid], -scores[cid], malformed[cid], cid)

        ranked = sorted(ids, key=key)
        pool_hash = stable_hash(sorted(ids))
        coverage = self.config.ranking.get("minimum_pair_coverage", 1.0)
        selector = {
            "method": _METHOD,
            "minimum_pair_coverage": float(coverage if isinstance(coverage, (int, float)) else 1.0),
        }
        rid = ranking_id(task_id, pool_hash, selector)
        ranking = tuple(
            RankedCandidate(cid, scores[cid], utilities[cid], rank)
            for rank, cid in enumerate(ranked, start=1)
        )
        result_json = json.dumps(
            {
                "task_id": task_id,
                "pool_hash": pool_hash,
                "selector": selector,
                "ranking": [
                    {
                        "candidate_id": r.candidate_id,
                        "expected_score": r.expected_score,
                        "utility": r.utility,
                        "rank": r.rank,
                    }
                    for r in ranking
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
        records: list[RankingRecord] = []
        self._validate_no_grader()
        for task in read_task_names(self.config.experiment.task_file):
            records.append(self._rank_task(task))
        return records

    def _validate_no_grader(self) -> None:
        """Guard: ranking must never consult the ground-truth store."""
        with self.catalog.connect() as scoped:
            _ = scoped  # ground truth DB is a separate file, never opened here
