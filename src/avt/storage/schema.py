"""SQLite schema and migrations for the AVT stores.

Two separate databases:

- ``experiment.sqlite``  — catalog of experiments, tasks, candidates, pairs,
  verifications, rankings, and job state. The verifier reads this.
- ``ground_truth.sqlite`` — official grader results only; never opened by
  the verification process.
"""

from __future__ import annotations

import sqlite3

__all__ = [
    "EXPERIMENT_STAGES",
    "JOB_STATES",
    "SCHEMA_VERSION",
    "create_experiment_schema",
    "create_ground_truth_schema",
]

SCHEMA_VERSION = 2

EXPERIMENT_STAGES: tuple[str, ...] = (
    "CREATED",
    "GENERATING",
    "GENERATED",
    "PAIRING",
    "PAIRED",
    "VERIFYING",
    "VERIFIED",
    "RANKING",
    "RANKED",
    "EVALUATING",
    "COMPLETE",
)

JOB_STATES: tuple[str, ...] = (
    "PENDING",
    "RUNNING",
    "SUCCEEDED",
    "RETRYABLE_FAILED",
    "PERMANENT_FAILED",
)

_EXPERIMENT_DDL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id   TEXT PRIMARY KEY,
    config          TEXT NOT NULL,
    stage           TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tasks (
    experiment_id  TEXT NOT NULL,
    task_id        TEXT NOT NULL,
    instruction    TEXT,
    PRIMARY KEY (experiment_id, task_id),
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS candidates (
    candidate_id    TEXT PRIMARY KEY,
    experiment_id   TEXT NOT NULL,
    task_id         TEXT NOT NULL,
    attempt_index   INTEGER NOT NULL,
    status          TEXT NOT NULL,
    artifact_path   TEXT,
    generator_usage TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
);

CREATE TABLE IF NOT EXISTS pairs (
    pair_id       TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    candidate_a   TEXT NOT NULL,
    candidate_b   TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id),
    FOREIGN KEY (candidate_a) REFERENCES candidates(candidate_id),
    FOREIGN KEY (candidate_b) REFERENCES candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS verifications (
    verification_id TEXT PRIMARY KEY,
    pair_id         TEXT NOT NULL,
    criterion       TEXT NOT NULL,
    repetition      INTEGER NOT NULL,
    display_order   TEXT NOT NULL,
    status          TEXT NOT NULL,
    malformed_attempts INTEGER NOT NULL DEFAULT 0,
    request_path    TEXT,
    response_path   TEXT,
    scores_path     TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (pair_id) REFERENCES pairs(pair_id)
);

CREATE TABLE IF NOT EXISTS expected_scores (
    candidate_id        TEXT NOT NULL,
    criterion           TEXT NOT NULL,
    raw_expected_score  REAL NOT NULL,
    normalized_score    REAL NOT NULL,
    observations        INTEGER NOT NULL,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (candidate_id, criterion),
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS evaluation (
    candidate_id        TEXT PRIMARY KEY,
    aggregate_raw       REAL NOT NULL,
    aggregate_normalized REAL NOT NULL,
    criteria            TEXT NOT NULL,
    observations        INTEGER NOT NULL,
    created_at          TEXT NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
);

CREATE TABLE IF NOT EXISTS rankings (
    ranking_id      TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL,
    pool_hash       TEXT NOT NULL,
    selector_config TEXT NOT NULL,
    result          TEXT,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS jobs (
    job_type   TEXT NOT NULL,
    job_id     TEXT NOT NULL,
    state      TEXT NOT NULL,
    attempts   INTEGER NOT NULL DEFAULT 0,
    payload    TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (job_type, job_id)
);

CREATE INDEX IF NOT EXISTS idx_candidates_exp_task
    ON candidates(experiment_id, task_id);
CREATE INDEX IF NOT EXISTS idx_pairs_exp_task
    ON pairs(experiment_id, task_id);
CREATE INDEX IF NOT EXISTS idx_verifications_pair
    ON verifications(pair_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_GROUND_TRUTH_DDL = """
CREATE TABLE IF NOT EXISTS official_results (
    candidate_id TEXT PRIMARY KEY,
    task_id      TEXT NOT NULL,
    reward       REAL,
    raw          TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_CHECKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "experiment": (
        _EXPERIMENT_DDL,
        EXPERIMENT_STAGES + JOB_STATES,
    ),
    "ground_truth": (_GROUND_TRUTH_DDL, ()),
}


def _exec_script(conn: sqlite3.Connection, script: str) -> None:
    conn.executescript(script)
    conn.commit()


def _migrate_experiment(conn: sqlite3.Connection) -> None:
    """Add ``verifications.malformed_attempts`` for pre-v2 databases."""
    cols = conn.execute("PRAGMA table_info(verifications)").fetchall()
    if not any(str(r[1]) == "malformed_attempts" for r in cols):
        conn.execute(
            "ALTER TABLE verifications ADD COLUMN malformed_attempts INTEGER NOT NULL DEFAULT 0"
        )
        conn.commit()


def _bump_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def create_experiment_schema(conn: sqlite3.Connection) -> None:
    _exec_script(conn, _EXPERIMENT_DDL)
    _migrate_experiment(conn)
    _bump_meta(conn, "schema_version", str(SCHEMA_VERSION))


def create_ground_truth_schema(conn: sqlite3.Connection) -> None:
    _exec_script(conn, _GROUND_TRUTH_DDL)
    _bump_meta(conn, "schema_version", str(SCHEMA_VERSION))
