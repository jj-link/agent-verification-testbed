"""Tests for frozen candidate pair construction."""

from __future__ import annotations

from pathlib import Path

from avt.config import load_config
from avt.pairs import PairBuilder, display_order
from avt.storage.ids import candidate_id

CONFIG_YAML = """
experiment:
  name: s
  seed: 42
  task_file: __TASKFILE__
  candidates_per_task: 3
upstream:
  terminal_bench_commit: abc
  harbor_commit: def
generator:
  harness: qwen-coder
  model: m
  endpoint: e
  temperature: 0.7
  max_tokens: 8192
verifier:
  model: m
  endpoint: e
  criteria: [specification]
  granularity: 5
  repetitions: 1
rendering:
  max_pair_context_tokens: 1000
ranking:
  method: round_robin_bradley_terry
storage:
  root: __ROOT__
  metadata_db: __ROOT__/experiment.sqlite
  ground_truth_db: __ROOT__/ground_truth.sqlite
"""


def _make(tmp_path: Path, tasks: tuple[str, ...] = ("task_x",)) -> PairBuilder:
    (tmp_path / "tasks.txt").write_text("\n".join(tasks) + "\n", encoding="utf-8")
    cfg = CONFIG_YAML.replace("__TASKFILE__", (tmp_path / "tasks.txt").as_posix()).replace(
        "__ROOT__", (tmp_path / "avt").as_posix()
    )
    cfg_path = tmp_path / "smoke.yaml"
    cfg_path.write_text(cfg, encoding="utf-8")
    return PairBuilder(load_config(cfg_path), tmp_path)


def _seed(builder: PairBuilder, task: str, n: int) -> list[str]:
    exp = builder.exp
    with builder.catalog.connect() as scoped:
        scoped.upsert_experiment_config(exp, {}, "GENERATED")
        for i in range(n):
            cid = candidate_id(exp, task, i)
            scoped.record_candidate(cid, exp, task, i, "SUCCEEDED", None, None)
    return [candidate_id(exp, task, i) for i in range(n)]


def _pair_rows(builder: PairBuilder) -> list[tuple[str, str, str]]:
    with builder.catalog.connect() as scoped:
        rows = scoped._conn.execute(
            "SELECT pair_id, candidate_a, candidate_b FROM pairs"
        ).fetchall()
    return [(str(r[0]), str(r[1]), str(r[2])) for r in rows]


def test_builds_all_unordered_pairs_sorted(tmp_path: Path) -> None:
    builder = _make(tmp_path)
    _seed(builder, "task_x", 3)
    pairs = builder.build()
    assert len(pairs) == 3  # C(3,2)

    rows = _pair_rows(builder)
    assert len(rows) == 3
    for _, a, b in rows:
        # canonical unordered membership: stored sorted
        assert a < b
    # all distinct
    assert len({p for p, _, _ in rows}) == 3


def test_pair_id_is_unordered(tmp_path: Path) -> None:
    builder = _make(tmp_path)
    ids = _seed(builder, "task_x", 3)
    p = ids[:2]  # canonical unordered pair, checked in both orders
    from avt.storage.ids import pair_id

    assert pair_id(builder.exp, "task_x", p) == pair_id(
        builder.exp, "task_x", (p[1], p[0])
    )


def test_build_is_idempotent(tmp_path: Path) -> None:
    builder = _make(tmp_path)
    _seed(builder, "task_x", 3)
    first = builder.build()
    before = len(_pair_rows(builder))
    second = builder.build()
    assert len(second) == len(first) == 3
    assert len(_pair_rows(builder)) == before  # no duplicates


def test_skips_task_with_single_candidate(tmp_path: Path) -> None:
    builder = _make(tmp_path, tasks=("task_alone",))
    _seed(builder, "task_alone", 1)
    assert builder.build() == []


def test_display_order_deterministic_and_unbiased(tmp_path: Path) -> None:
    # display order depends only on (seed, pair), so it is stable across calls
    a, b = "cand_b", "cand_c"
    first = display_order(a, b, 42)
    assert display_order(a, b, 42) == first  # deterministic
    # across many pairs we see both orientations (sanity, not a statistical claim)
    orders = {
        display_order(f"cand_{i}", f"cand_{j}", 42)
        for i in range(1, 20)
        for j in range(i + 1, 20)
    }
    aas = {o[0] for o in orders if o[0] > o[1]}
    bbs = {o[0] for o in orders if o[0] < o[1]}
    assert aas and bbs  # both orders occur


def test_experiment_stage_paired(tmp_path: Path) -> None:
    builder = _make(tmp_path)
    _seed(builder, "task_x", 3)
    builder.build()
    with builder.catalog.connect() as scoped:
        assert scoped.get_experiment_stage(builder.exp) == "PAIRED"
