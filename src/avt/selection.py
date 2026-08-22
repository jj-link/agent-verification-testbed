"""Deterministic seeded task selection from a task list."""

from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from pathlib import Path

__all__ = ["fetch_task_pool", "read_task_names", "select_tasks"]


def fetch_task_pool(
    owner: str = "laude-institute",
    repo: str = "terminal-bench-2",
    commit: str = "",
) -> list[str]:
    """List top-level task directories (those containing ``task.toml``) at a commit."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{commit}?recursive=1"
    with urllib.request.urlopen(url, timeout=60) as resp:
        tree = json.load(resp)["tree"]
    task_dirs: set[str] = set()
    for entry in tree:
        path = entry.get("path", "")
        parts = path.split("/")
        if len(parts) == 2 and parts[1] == "task.toml":
            task_dirs.add(parts[0])
    return sorted(task_dirs)


def read_task_names(path: str | Path) -> list[str]:
    tasks = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return tasks


def write_task_file(path: str | Path, tasks: list[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(tasks) + ("\n" if tasks else ""), encoding="utf-8")
    return p


def select_tasks(
    pool: list[str],
    *,
    smoke: int = 2,
    pilot: int = 8,
    main: int = 25,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Deterministically choose disjoint smoke, pilot, and main task subsets.

    The pool is sorted, then split by stable seeded picks so the seed fully
    determines the selection. Smoke and pilot are subsets of main.
    """
    unique_pool = sorted(set(pool))
    if main > len(unique_pool):
        raise ValueError(f"requested {main} main tasks but pool has {len(unique_pool)}")
    if smoke > pilot:
        raise ValueError("smoke count must be <= pilot count")

    rng = random.Random(seed)
    # Stable ordering, then pick main indices in seed order.
    indices = list(range(len(unique_pool)))
    rng.shuffle(indices)
    main_sel = sorted(indices[:main])
    by_name = {unique_pool[i]: i for i in main_sel}
    chosen = {unique_pool[i]: i for i in main_sel}

    # Smoke and pilot picked deterministically from the selected main pool.
    def _pick(count: int) -> list[str]:
        seed_hash = int(hashlib.sha256(f"{seed}:{count}".encode()).hexdigest()[:8], 16)
        r2 = random.Random(seed_hash)
        names = sorted(chosen)
        r2.shuffle(names)
        return names[:count]

    smoke_sel = sorted(_pick(smoke))
    pilot_sel = sorted(_pick(pilot))

    return {
        "smoke": smoke_sel,
        "pilot": pilot_sel,
        "main": sorted(by_name),
    }
