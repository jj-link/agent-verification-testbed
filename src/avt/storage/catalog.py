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
