"""Frozen candidate pair construction (plan section 14).

Builds all unordered candidate pairs per task from the frozen pool, assigns a
deterministic A/B display order from the experiment seed, and persists the pairs
to the catalog. Building is idempotent: existing pair rows are refreshed, never
duplicated, so reruns resume rather than regenerate.
"""

from __future__ import annotations

import hashlib
import json
import random
from itertools import combinations
from pathlib import Path

from avt.config import Config
from avt.rendering import load_atif
from avt.selection import read_task_names
from avt.storage.catalog import Catalog
from avt.storage.ids import experiment_id, pair_id

__all__ = ["PairBuilder", "display_order"]

PAIRED = "PAIRED"


def _seed_rand(seed: int, *parts: str) -> random.Random:
    digest = hashlib.sha256(":".join([str(seed), *parts]).encode()).hexdigest()[:8]
    return random.Random(int(digest, 16))


def display_order(candidate_a: str, candidate_b: str, seed: int) -> tuple[str, str]:
    """Deterministically choose which candidate is displayed as A vs B."""
    rng = _seed_rand(seed, candidate_a, candidate_b)
    names = [candidate_a, candidate_b]
    rng.shuffle(names)
    return names[0], names[1]


class PairBuilder:
    def __init__(self, config: Config, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self.catalog = Catalog(Path(config.storage.metadata_db))
        self.exp = experiment_id(config.raw)

    def _list_candidates(self, task_id: str) -> list[dict[str, object]]:
        with self.catalog.connect() as scoped:
            return scoped.list_candidates(self.exp, task_id)

    def _instruction_for(self, task_id: str) -> str | None:
        """Public task instruction from the first candidate's ATIF user step."""
        for cand in self._list_candidates(task_id):
            artifact_path = cand.get("artifact_path")
            if not artifact_path:
                continue
            tj = Path(str(artifact_path)) / "trajectory.json"
            if not tj.exists():
                continue
            try:
                meta = json.loads(tj.read_text(encoding="utf-8"))
                trial_result = meta.get("trial_result")
                if not trial_result:
                    continue
                atif = Path(str(trial_result)).parent / "agent" / "trajectory.json"
                steps = load_atif(atif)
                instrs = [
                    str(s.get("message"))
                    for s in steps
                    if s.get("source") == "user" and s.get("message")
                ]
                if instrs:
                    return "\n\n".join(instrs)
            except Exception:
                continue
        return None

    def _record_task(self, task_id: str, instruction: str | None) -> None:
        with self.catalog.connect() as scoped:
            scoped.record_task(self.exp, task_id, instruction or "")

    def _record_pair(self, pair: tuple[str, str], task_id: str) -> str:
        pid = pair_id(self.exp, task_id, pair)
        # candidate_a/b are the canonical unordered membership (sorted); A/B
        # display order is assigned per-verification in verifications.display_order.
        with self.catalog.connect() as scoped:
            scoped.record_pair(pid, self.exp, task_id, pair[0], pair[1], PAIRED)
        return pid

    def build(self) -> list[str]:
        """Build frozen pairs for every task with >= 2 candidates."""
        tasks = read_task_names(self.config.experiment.task_file)
        built: list[str] = []
        for task in tasks:
            cands = self._list_candidates(task)
            ids = sorted(str(c["candidate_id"]) for c in cands)
            if len(ids) < 2:
                continue
            self._record_task(task, self._instruction_for(task))
            for pair in combinations(ids, 2):
                built.append(self._record_pair(pair, task))
        with self.catalog.connect() as scoped:
            scoped.set_experiment_stage(self.exp, PAIRED)
        return built
