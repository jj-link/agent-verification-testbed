"""Ground-truth store (ground_truth.sqlite) — official grader results only.

This database is intentionally never opened by the verification process. It is
read at the analysis/evaluation join stage only.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from avt.storage.schema import create_ground_truth_schema

__all__ = ["GroundTruth", "GroundTruthConnection"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


class GroundTruthConnection:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        self._conn.close()

    def put(
        self,
        candidate_id: str,
        task_id: str,
        reward: float | None,
        raw: object,
    ) -> None:
        import json

        self._conn.execute(
            "INSERT INTO official_results(candidate_id, task_id, reward, raw, created_at) "
            "VALUES(?,?,?,?,?) "
            "ON CONFLICT(candidate_id) DO UPDATE SET reward=excluded.reward, "
            "raw=excluded.raw",
            (candidate_id, task_id, reward, json.dumps(raw) if raw is not None else None, _now()),
        )
        self._conn.commit()

    def get(self, candidate_id: str, task_id: str = "") -> dict[str, object] | None:
        import json

        row = self._conn.execute(
            "SELECT candidate_id, task_id, reward, raw FROM official_results WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "candidate_id": row[0],
            "task_id": row[1],
            "reward": row[2],
            "raw": json.loads(row[3]) if row[3] else None,
        }


class GroundTruth:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path))
        conn.execute("PRAGMA foreign_keys=ON")
        create_ground_truth_schema(conn)
        conn.close()

    @contextmanager
    def connect(self) -> Iterator[GroundTruthConnection]:
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        scoped = GroundTruthConnection(conn)
        try:
            yield scoped
        finally:
            scoped.close()
