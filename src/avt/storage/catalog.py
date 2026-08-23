"""Catalog store (experiment.sqlite) and resumable job state.

Single-controller design. Foreign keys are enforced per connection. Crash-left
``RUNNING`` jobs are reclaimed to ``RETRYABLE_FAILED`` by an explicit
``recover_interrupted()`` call at controller startup — never implicitly on every
connection open (so a reader cannot steal an active job). ``claim_job`` is a
single conditional UPDATE so concurrent claims cannot both win.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from avt.storage.schema import create_experiment_schema

__all__ = ["Catalog", "CatalogConnection", "Job"]

_RECLAIM_STATE = "RETRYABLE_FAILED"
_RUNNABLE = ("PENDING", "RETRYABLE_FAILED")


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Job:
    job_type: str
    job_id: str
    state: str
    attempts: int
    payload: dict[str, object] | None


class CatalogConnection:
    """A scoped connection to the catalog database."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        self._conn.close()

    def set_experiment_stage(self, experiment_id: str, stage: str) -> None:
        self._conn.execute(
            "INSERT INTO experiments(experiment_id, config, stage, created_at) VALUES(?,?,?,?) "
            "ON CONFLICT(experiment_id) DO UPDATE SET stage=excluded.stage",
            (experiment_id, "{}", stage, _now()),
        )
        self._conn.commit()

    def upsert_experiment_config(
        self, experiment_id: str, config: dict[str, object], stage: str
    ) -> None:
        self._conn.execute(
            "INSERT INTO experiments(experiment_id, config, stage, created_at) VALUES(?,?,?,?) "
            "ON CONFLICT(experiment_id) DO UPDATE SET config=excluded.config, "
            "stage=excluded.stage",
            (experiment_id, json.dumps(config, sort_keys=True), stage, _now()),
        )
        self._conn.commit()

    def get_experiment_stage(self, experiment_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT stage FROM experiments WHERE experiment_id=?", (experiment_id,)
        ).fetchone()
        return str(row[0]) if row else None

    def enqueue_job(
        self, job_type: str, job_id: str, payload: dict[str, object] | None = None
    ) -> None:
        """Create a PENDING job if absent; leave any existing state unchanged."""
        self._conn.execute(
            "INSERT OR IGNORE INTO jobs(job_type, job_id, state, attempts, payload, updated_at) "
            "VALUES(?,?,?,?,?,?)",
            (job_type, job_id, "PENDING", 0, json.dumps(payload) if payload else None, _now()),
        )
        self._conn.commit()

    def get_job(self, job_type: str, job_id: str) -> Job | None:
        row = self._conn.execute(
            "SELECT job_type, job_id, state, attempts, payload FROM jobs "
            "WHERE job_type=? AND job_id=?",
            (job_type, job_id),
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[4]) if row[4] else None
        return Job(row[0], row[1], row[2], int(row[3]), payload)

    def claim_job(self, job_type: str, job_id: str) -> Job | None:
        """Atomically transition a runnable job to RUNNING; return it, else None.

        Only ``PENDING``/``RETRYABLE_FAILED`` jobs are claimable; ``SUCCEEDED``,
        ``PERMANENT_FAILED``, and already-``RUNNING`` jobs are not. The conditional
        UPDATE under an immediate transaction guarantees only one claimer wins.
        """
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cur = self._conn.execute(
                "UPDATE jobs SET state='RUNNING', attempts=attempts+1, updated_at=? "
                "WHERE job_type=? AND job_id=? AND state IN ('PENDING', 'RETRYABLE_FAILED')",
                (_now(), job_type, job_id),
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        if cur.rowcount == 0:
            return None
        return self.get_job(job_type, job_id)

    def enqueue_or_claim(
        self, job_type: str, job_id: str, payload: dict[str, object] | None = None
    ) -> Job | None:
        """Ensure the job exists, then claim it (or None if not runnable)."""
        self.enqueue_job(job_type, job_id, payload)
        return self.claim_job(job_type, job_id)

    def mark_succeeded(self, job_type: str, job_id: str) -> None:
        self._conn.execute(
            "UPDATE jobs SET state='SUCCEEDED', updated_at=? WHERE job_type=? AND job_id=?",
            (_now(), job_type, job_id),
        )
        self._conn.commit()

    def mark_failed(self, job_type: str, job_id: str, *, permanent: bool) -> None:
        state = "PERMANENT_FAILED" if permanent else _RECLAIM_STATE
        self._conn.execute(
            "UPDATE jobs SET state=?, updated_at=? WHERE job_type=? AND job_id=?",
            (state, _now(), job_type, job_id),
        )
        self._conn.commit()

    def count_jobs(self, job_type: str, state: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE job_type=? AND state=?",
            (job_type, state),
        ).fetchone()
        return int(row[0])

    def record_candidate(
        self,
        candidate_id: str,
        experiment_id: str,
        task_id: str,
        attempt_index: int,
        status: str,
        artifact_path: str | None,
        generator_usage: str | None,
    ) -> None:
        """Upsert a generated candidate into the frozen-pool manifest.

        The experiment row must already exist (callers ensure it via
        ``upsert_experiment_config``); the FK to ``experiments`` is enforced per
        connection. Re-running is idempotent: terminal state, artifact path, and
        usage are refreshed on conflict, so resume/retry never duplicates rows.
        """
        self._conn.execute(
            "INSERT INTO candidates("
            "candidate_id, experiment_id, task_id, attempt_index, status, "
            "artifact_path, generator_usage, created_at) "
            "VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(candidate_id) DO UPDATE SET "
            "status=excluded.status, artifact_path=excluded.artifact_path, "
            "generator_usage=excluded.generator_usage",
            (
                candidate_id,
                experiment_id,
                task_id,
                attempt_index,
                status,
                artifact_path,
                generator_usage,
                _now(),
            ),
        )
        self._conn.commit()

    def record_task(self, experiment_id: str, task_id: str, instruction: str) -> None:
        """Record a task's public instruction immutably.

        Identical rewrites are a no-op; a conflicting instruction raises so a
        frozen pool can never be silently changed.
        """
        row = self._conn.execute(
            "SELECT instruction FROM tasks WHERE experiment_id=? AND task_id=?",
            (experiment_id, task_id),
        ).fetchone()
        if row is not None:
            if (row[0] or "") != instruction:
                raise ValueError(
                    f"task instruction conflict for {task_id!r}: "
                    f"existing {row[0]!r} vs new {instruction!r}"
                )
            return
        self._conn.execute(
            "INSERT INTO tasks(experiment_id, task_id, instruction) VALUES(?,?,?)",
            (experiment_id, task_id, instruction),
        )
        self._conn.commit()

    def get_task_instruction(self, experiment_id: str, task_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT instruction FROM tasks WHERE experiment_id=? AND task_id=?",
            (experiment_id, task_id),
        ).fetchone()
        return str(row[0]) if row and row[0] else None

    def list_candidates(self, experiment_id: str, task_id: str) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT candidate_id, task_id, attempt_index, status, artifact_path "
            "FROM candidates WHERE experiment_id=? AND task_id=? AND status=? "
            "ORDER BY attempt_index",
            (experiment_id, task_id, "SUCCEEDED"),
        ).fetchall()
        return [dict(r) for r in rows]

    def record_pair(
        self,
        pair_id: str,
        experiment_id: str,
        task_id: str,
        candidate_a: str,
        candidate_b: str,
        status: str,
    ) -> None:
        """Record a frozen candidate pair immutably.

        Identical rewrites are a no-op; any conflict raises so the frozen pair
        records can never be silently mutated on rerun. candidate_a/b are the
        canonical sorted membership; A/B display order lives per-verification.
        """
        row = self._conn.execute(
            "SELECT task_id, candidate_a, candidate_b, status FROM pairs WHERE pair_id=?",
            (pair_id,),
        ).fetchone()
        if row is not None:
            if (row[0], row[1], row[2], row[3]) != (task_id, candidate_a, candidate_b, status):
                raise ValueError(
                    f"pair conflict for {pair_id!r}: existing {row!r} vs "
                    f"new {(task_id, candidate_a, candidate_b, status)!r}"
                )
            return
        self._conn.execute(
            "INSERT INTO pairs(pair_id, experiment_id, task_id, candidate_a, "
            "candidate_b, status, created_at) VALUES(?,?,?,?,?,?,?)",
            (pair_id, experiment_id, task_id, candidate_a, candidate_b, status, _now()),
        )
        self._conn.commit()

    def record_verification(
        self,
        verification_id_: str,
        pair_id: str,
        criterion: str,
        repetition: int,
        display_order_: str,
        status: str,
        request_path: str | None,
        response_path: str | None,
        scores_path: str | None,
        malformed_attempts: int = 0,
    ) -> None:
        """Upsert a verification record. Idempotent: identical rewrites are a
        no-op; any conflict raises so a frozen verification row is immutable.
        ``malformed_attempts`` counts retries that returned an unparsable
        score-token response before the eventual success (plan 14 tie-break)."""
        row = self._conn.execute(
            "SELECT pair_id, criterion, repetition, display_order, status, "
            "request_path, response_path, scores_path, malformed_attempts "
            "FROM verifications WHERE verification_id=?",
            (verification_id_,),
        ).fetchone()
        if row is not None:
            current = (
                row[0],
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                row[8],
            )
            incoming = (
                pair_id,
                criterion,
                repetition,
                display_order_,
                status,
                request_path,
                response_path,
                scores_path,
                malformed_attempts,
            )
            if current != incoming:
                raise ValueError(
                    f"verification conflict for {verification_id_!r}: "
                    f"existing {current!r} vs new {incoming!r}"
                )
            return
        self._conn.execute(
            "INSERT INTO verifications(verification_id, pair_id, criterion, "
            "repetition, display_order, status, request_path, response_path, "
            "scores_path, malformed_attempts, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                verification_id_,
                pair_id,
                criterion,
                repetition,
                display_order_,
                status,
                request_path,
                response_path,
                scores_path,
                malformed_attempts,
                _now(),
            ),
        )
        self._conn.commit()

    def record_expected_score(
        self,
        candidate_id_: str,
        criterion: str,
        raw_expected_score: float,
        normalized_score: float,
        observations: int,
    ) -> None:
        """Record a continuous expected score immutably (no-op if identical)."""
        row = self._conn.execute(
            "SELECT criterion, raw_expected_score, normalized_score, observations "
            "FROM expected_scores WHERE candidate_id=? AND criterion=?",
            (candidate_id_, criterion),
        ).fetchone()
        incoming = (criterion, raw_expected_score, normalized_score, observations)
        if row is not None:
            current = (row[0], row[1], row[2], row[3])
            if current != incoming:
                raise ValueError(
                    f"expected-score conflict for {candidate_id_!r}/{criterion!r}: "
                    f"existing {current!r} vs new {incoming!r}"
                )
            return
        self._conn.execute(
            "INSERT INTO expected_scores(candidate_id, criterion, raw_expected_score, "
            "normalized_score, observations, created_at) VALUES(?,?,?,?,?,?)",
            (candidate_id_, criterion, raw_expected_score, normalized_score, observations, _now()),
        )
        self._conn.commit()

    def record_ranking(
        self,
        ranking_id_: str,
        task_id: str,
        pool_hash: str,
        selector_config: str,
        result: str,
        status: str,
    ) -> None:
        """Record a ranking immutably (no-op if identical, conflict raises)."""
        row = self._conn.execute(
            "SELECT task_id, pool_hash, selector_config, result, status "
            "FROM rankings WHERE ranking_id=?",
            (ranking_id_,),
        ).fetchone()
        incoming = (task_id, pool_hash, selector_config, result, status)
        if row is not None:
            current = (row[0], row[1], row[2], row[3], row[4])
            if current != incoming:
                raise ValueError(
                    f"ranking conflict for {ranking_id_!r}: "
                    f"existing {current!r} vs new {incoming!r}"
                )
            return
        self._conn.execute(
            "INSERT INTO rankings(ranking_id, task_id, pool_hash, selector_config, "
            "result, status, created_at) VALUES(?,?,?,?,?,?,?)",
            (ranking_id_, task_id, pool_hash, selector_config, result, status, _now()),
        )
        self._conn.commit()

    def list_expected_scores(self, candidate_id_: str) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT criterion, raw_expected_score, normalized_score, observations "
            "FROM expected_scores WHERE candidate_id=? ORDER BY criterion",
            (candidate_id_,),
        ).fetchall()
        return [dict(r) for r in rows]

    def record_evaluation(
        self,
        candidate_id_: str,
        aggregate_raw: float,
        aggregate_normalized: float,
        criteria: str,
        observations: int,
    ) -> None:
        """Record a candidate's three-criterion aggregate immutably."""
        row = self._conn.execute(
            "SELECT aggregate_raw, aggregate_normalized, criteria, observations "
            "FROM evaluation WHERE candidate_id=?",
            (candidate_id_,),
        ).fetchone()
        incoming = (aggregate_raw, aggregate_normalized, criteria, observations)
        if row is not None:
            current = (row[0], row[1], row[2], row[3])
            if current != incoming:
                raise ValueError(
                    f"evaluation conflict for {candidate_id_!r}: "
                    f"existing {current!r} vs new {incoming!r}"
                )
            return
        self._conn.execute(
            "INSERT INTO evaluation(candidate_id, aggregate_raw, aggregate_normalized, "
            "criteria, observations, created_at) VALUES(?,?,?,?,?,?)",
            (candidate_id_, aggregate_raw, aggregate_normalized, criteria, observations, _now()),
        )
        self._conn.commit()

    def list_evaluation(self, candidate_id_: str) -> dict[str, object] | None:
        row = self._conn.execute(
            "SELECT candidate_id, aggregate_raw, aggregate_normalized, criteria, observations "
            "FROM evaluation WHERE candidate_id=?",
            (candidate_id_,),
        ).fetchone()
        return dict(row) if row else None

    def list_verifications(self) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT verification_id, pair_id, criterion, repetition, display_order, "
            "status, request_path, response_path, scores_path, "
            "malformed_attempts FROM verifications "
            "ORDER BY pair_id, criterion"
        ).fetchall()
        return [dict(r) for r in rows]

    def list_pairs(self, experiment_id: str, task_id: str) -> list[dict[str, object]]:
        rows = self._conn.execute(
            "SELECT pair_id, candidate_a, candidate_b, status FROM pairs "
            "WHERE experiment_id=? AND task_id=?",
            (experiment_id, task_id),
        ).fetchall()
        return [dict(r) for r in rows]

    def malformed_attempts_for(self, candidate_id_: str) -> int:
        """Total malformed score-token retries across a candidate's verifications."""
        rows = self._conn.execute(
            "SELECT v.malformed_attempts FROM verifications v "
            "JOIN pairs p ON p.pair_id = v.pair_id "
            "WHERE p.candidate_a = ? OR p.candidate_b = ?",
            (candidate_id_, candidate_id_),
        ).fetchall()
        return sum(int(r[0]) for r in rows)

    def count_succeeded_verifications(self, pair_id: str) -> int:
        """Number of SUCCEEDED verifications for a pair (ranking coverage check)."""
        row = self._conn.execute(
            "SELECT COUNT(*) FROM verifications WHERE pair_id=? AND status='SUCCEEDED'",
            (pair_id,),
        ).fetchone()
        return int(row[0])

    def count_failed_verifications(self, experiment_id_: str) -> int:
        """Number of FAILED (persistently-malformed) verifications in an experiment.

        Pool-wide over this experiment's pairs, used to decide whether the
        verification stage achieved full usable coverage before VERIFIED.
        """
        row = self._conn.execute(
            "SELECT COUNT(*) FROM verifications v JOIN pairs p ON v.pair_id = p.pair_id "
            "WHERE p.experiment_id=? AND v.status='FAILED'",
            (experiment_id_,),
        ).fetchone()
        return int(row[0]) if row else 0

    def succeeded_verification_keys(self, pair_id: str) -> set[tuple[str, int]]:
        """SUCCEEDED (criterion, repetition) keys for a pair, for exact coverage."""
        rows = self._conn.execute(
            "SELECT criterion, repetition FROM verifications "
            "WHERE pair_id=? AND status='SUCCEEDED'",
            (pair_id,),
        ).fetchall()
        return {(str(r[0]), int(r[1])) for r in rows}

    def terminal_verification_keys(self, pair_id: str) -> set[tuple[str, int]]:
        """Terminal (criterion, repetition) keys for a pair: SUCCEEDED or FAILED.

        Used for resumable verification so a resumed or partial run only makes
        model calls for keys that have no recorded terminal outcome.
        """
        rows = self._conn.execute(
            "SELECT criterion, repetition FROM verifications "
            "WHERE pair_id=? AND status IN ('SUCCEEDED','FAILED')",
            (pair_id,),
        ).fetchall()
        return {(str(r[0]), int(r[1])) for r in rows}

    def verification_status(self, verification_id_: str) -> str | None:
        """Status for one exact verifier identity/key, or None if it has not run."""
        row = self._conn.execute(
            "SELECT status FROM verifications WHERE verification_id=?",
            (verification_id_,),
        ).fetchone()
        return str(row[0]) if row else None

    def remove_superseded_verifications(
        self, pair_id: str, current_verification_ids: set[str]
    ) -> int:
        """Remove catalog rows for older verifier identities on one pair.

        Verification artifacts remain content-addressed on disk. The catalog is
        the current experiment view consumed by coverage, expected-score, and
        ranking stages, so it must not mix prompt/model/output-policy identities.
        """
        if not current_verification_ids:
            cur = self._conn.execute(
                "DELETE FROM verifications WHERE pair_id=?",
                (pair_id,),
            )
        else:
            placeholders = ",".join("?" for _ in current_verification_ids)
            params: tuple[object, ...] = (pair_id, *sorted(current_verification_ids))
            cur = self._conn.execute(
                f"DELETE FROM verifications WHERE pair_id=? "
                f"AND verification_id NOT IN ({placeholders})",
                params,
            )
        self._conn.commit()
        return cur.rowcount


class Catalog:
    """Owns the experiment catalog database file and opens connections."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.execute("PRAGMA foreign_keys=ON")
        create_experiment_schema(conn)
        conn.close()

    @contextmanager
    def connect(self) -> Iterator[CatalogConnection]:
        """Yield a connection. Does not implicitly reclaim jobs (see recover_interrupted)."""
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        scoped = CatalogConnection(conn)
        try:
            yield scoped
        finally:
            scoped.close()

    def recover_interrupted(self) -> int:
        """Reclaim crash-left RUNNING jobs to RETRYABLE_FAILED. Call once at startup."""
        with self.connect() as scoped:
            cur = scoped._conn.execute(
                "UPDATE jobs SET state=?, updated_at=? WHERE state='RUNNING'",
                (_RECLAIM_STATE, _now()),
            )
            scoped._conn.commit()
        return cur.rowcount
