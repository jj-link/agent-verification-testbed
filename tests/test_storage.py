"""Tests for the Stage 6 storage layer and deterministic identifiers."""

from __future__ import annotations

from pathlib import Path

from avt.storage import (
    ArtifactStore,
    Catalog,
    GroundTruth,
    candidate_id,
    experiment_id,
    pair_id,
    ranking_id,
    verification_id,
)

EXP_CONFIG = {"name": "smoke-v1", "candidates_per_task": 3, "seed": 42}
VERIFIER_CFG = {"granularity": 5, "criteria": ["specification", "output", "errors"]}


def test_experiment_id_is_deterministic_and_order_independent() -> None:
    a = experiment_id({"name": "x", "n": 3})
    b = experiment_id({"n": 3, "name": "x"})
    assert a == b
    assert experiment_id({"name": "x", "n": 4}) != a


def test_candidate_id_deterministic() -> None:
    exp = experiment_id(EXP_CONFIG)
    a = candidate_id(exp, "task", 0)
    assert a == candidate_id(exp, "task", 0)
    assert a != candidate_id(exp, "task", 1)


def test_pair_id_is_order_independent() -> None:
    exp = experiment_id(EXP_CONFIG)
    p1 = pair_id(exp, "t", ["b", "a"])
    p2 = pair_id(exp, "t", ["a", "b"])
    assert p1 == p2


def test_verification_and_ranking_ids_deterministic() -> None:
    p = pair_id(experiment_id(EXP_CONFIG), "t", ["a", "b"])
    v1 = verification_id(p, VERIFIER_CFG, "specification", 0, "ab")
    v2 = verification_id(p, VERIFIER_CFG, "specification", 0, "ab")
    assert v1 == v2
    assert v1 != verification_id(p, VERIFIER_CFG, "specification", 0, "ba")
    r = ranking_id("t", "poolhash", {"selector": "continuous"})
    assert r == ranking_id("t", "poolhash", {"selector": "continuous"})


def test_catalog_records_survive_reopen(tmp_path: Path) -> None:
    db = tmp_path / "experiment.sqlite"
    c = Catalog(db)
    exp = experiment_id(EXP_CONFIG)
    cand = candidate_id(exp, "task", 0)
    with c.connect() as scoped:
        scoped.upsert_experiment_config(exp, EXP_CONFIG, "GENERATING")
        scoped.enqueue_job("candidate", cand, {"attempt": 0})
        scoped.mark_succeeded("candidate", cand)
    # Reopen.
    c2 = Catalog(db)
    with c2.connect() as scoped:
        assert scoped.get_experiment_stage(exp) == "GENERATING"
        job = scoped.get_job("candidate", cand)
        assert job is not None and job.state == "SUCCEEDED"


def test_succeeded_job_is_skipped_on_resume(tmp_path: Path) -> None:
    c = Catalog(tmp_path / "experiment.sqlite")
    with c.connect() as scoped:
        scoped.enqueue_job("rank", "r1")
        assert scoped.claim_job("rank", "r1") is not None
        scoped.mark_succeeded("rank", "r1")
    with c.connect() as scoped:
        # Already succeeded -> not claimable again (skip).
        assert scoped.claim_job("rank", "r1") is None
        assert scoped.count_jobs("rank", "SUCCEEDED") == 1


def test_crash_left_running_job_is_reclaimed_and_resumed(tmp_path: Path) -> None:
    c = Catalog(tmp_path / "experiment.sqlite")
    with c.connect() as scoped:
        scoped.enqueue_job("verify", "v1")
        assert scoped.claim_job("verify", "v1") is not None  # crash here, leave RUNNING
    # Controller restart: explicit recovery reclaims the RUNNING job.
    c2 = Catalog(tmp_path / "experiment.sqlite")
    assert c2.recover_interrupted() == 1
    with c2.connect() as scoped:
        job = scoped.get_job("verify", "v1")
        assert job is not None and job.state == "RETRYABLE_FAILED"
        # Resume without duplicating rows.
        assert scoped.claim_job("verify", "v1") is not None
        assert scoped.claim_job("verify", "v1") is None  # now RUNNING, not double-claimed
        scoped.mark_succeeded("verify", "v1")
        assert scoped.count_jobs("verify", "SUCCEEDED") == 1


def test_claim_is_atomic_under_immediate_transaction(tmp_path: Path) -> None:
    c = Catalog(tmp_path / "experiment.sqlite")
    with c.connect() as scoped:
        scoped.enqueue_job("verify", "v2")
        assert scoped.claim_job("verify", "v2") is not None
        # Second claim of the same job cannot win.
        assert scoped.claim_job("verify", "v2") is None


def test_ground_truth_is_separate_file(tmp_path: Path) -> None:
    gt = GroundTruth(tmp_path / "ground_truth.sqlite")
    Catalog(tmp_path / "experiment.sqlite")
    with gt.connect() as g:
        g.put("cand1", "task", 1.0, {"reward": 1.0})
    with gt.connect() as g:
        got = g.get("cand1")
        assert got is not None
        assert got["reward"] == 1.0
    # Catalog and ground truth are distinct, non-overlapping files.
    assert (tmp_path / "experiment.sqlite").exists()
    assert (tmp_path / "ground_truth.sqlite").exists()
    assert (tmp_path / "experiment.sqlite").stat().st_size != 0


def test_artifact_store_writes_json_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    paths = store.write_candidate("cand1", {"m": 1}, {"steps": []}, {"tokens": 3})
    for p in paths.values():
        assert p.exists() and p.read_text(encoding="utf-8")
    vr = store.write_verification("v1", {"req": 1}, {"resp": 2}, {"scores": [1, 2, 3]})
    assert vr["scores"].exists()
    res = store.write_official_result("cand1", {"reward": 1.0})
    assert res.exists()


def test_record_candidate_indexes_pool_and_is_idempotent(tmp_path: Path) -> None:
    c = Catalog(tmp_path / "experiment.sqlite")
    exp = experiment_id(EXP_CONFIG)
    with c.connect() as scoped:
        scoped.upsert_experiment_config(exp, EXP_CONFIG, stage="GENERATING")
        scoped.record_candidate("c1", exp, "task", 0, "SUCCEEDED", "candidates/c1", '{"a":1}')
        # Re-record (resume/retry) must not duplicate the manifest row.
        scoped.record_candidate("c1", exp, "task", 0, "SUCCEEDED", "candidates/c1b", '{"a":1}')
        cnt = scoped._conn.execute(
            "SELECT COUNT(*) FROM candidates WHERE candidate_id='c1'"
        ).fetchone()[0]
        assert cnt == 1
        art = scoped._conn.execute(
            "SELECT artifact_path FROM candidates WHERE candidate_id='c1'"
        ).fetchone()[0]
        assert art == "candidates/c1b"
    # Pool is discoverable as an immutable manifest.
    with c.connect() as scoped:
        n = scoped._conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    assert n == 1
