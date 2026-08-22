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
        """Insert ground truth immutably; identical rewrites are a no-op.

        Raises ``ValueError`` if an existing record for ``candidate_id`` has a
        different task, reward, or raw payload, so a rerun/reindex cannot change
        frozen ground truth.
        """
        import json

        new_raw = json.dumps(raw, sort_keys=True, ensure_ascii=True) if raw is not None else None
        row = self._conn.execute(
            "SELECT task_id, reward, raw FROM official_results WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        if row is not None:
            existing_raw = row[2]
            if (
                row[0] != task_id
                or (row[1] is None and reward is not None)
                or (row[1] is not None and row[1] != reward)
                or existing_raw != new_raw
            ):
                raise ValueError(
                    f"ground truth conflict for candidate {candidate_id!r}: "
                    f"existing {row!r} vs new (task={task_id!r}, reward={reward!r})"
                )
            return
        self._conn.execute(
            "INSERT INTO official_results(candidate_id, task_id, reward, raw, created_at) "
            "VALUES(?,?,?,?,?)",
            (candidate_id, task_id, reward, new_raw, _now()),
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
