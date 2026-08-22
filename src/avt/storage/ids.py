"""Deterministic, content-addressed identifiers.

Stable IDs are derived from canonical serialized inputs so every stage is
idempotent and resumable. Order-independent inputs (dicts, sets) are
canonicalized before hashing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping

__all__ = [
    "canonical_json",
    "stable_hash",
    "experiment_id",
    "candidate_id",
    "pair_id",
    "verification_id",
    "ranking_id",
]

_ID_DIGEST_BYTES = 16  # 32 hex chars


def canonical_json(obj: object) -> str:
    """Serialize *obj* to a canonical string, order-independent for mappings."""

    def _canon(v: object) -> object:
        if isinstance(v, Mapping):
            return {str(k): _canon(val) for k, val in sorted(v.items(), key=lambda kv: str(kv[0]))}
        if isinstance(v, (list, tuple)):
            return [_canon(x) for x in v]
        if isinstance(v, set):
            return sorted((_canon(x) for x in v), key=str)
        if isinstance(v, (str, int, float, bool)) or v is None:
            return v
        raise TypeError(f"non-serializable identifier input: {type(v)!r}")

    return json.dumps(_canon(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(*parts: object) -> str:
    """Return a stable hex identifier for the canonicalized parts."""
    canonical = canonical_json(parts)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[: _ID_DIGEST_BYTES * 2]


def _ids(iterable: Iterable[str]) -> list[str]:
    return sorted(iterable)


def experiment_id(frozen_config: Mapping[str, object]) -> str:
    """ID for a frozen experiment configuration."""
    return stable_hash("experiment", canonical_json(dict(frozen_config)))


def candidate_id(experiment_id_: str, task_id: str, attempt_index: int) -> str:
    """ID for one generation attempt of a task under an experiment."""
    return stable_hash("candidate", experiment_id_, task_id, attempt_index)


def pair_id(
    experiment_id_: str,
    task_id: str,
    candidate_ids: Iterable[str],
) -> str:
    """ID for an unordered candidate pair (sorted, order-independent)."""
    return stable_hash("pair", experiment_id_, task_id, _ids(candidate_ids))


def verification_id(
    pair_id_: str,
    verifier_config: Mapping[str, object],
    criterion: str,
    repetition: int,
    display_order: str,
) -> str:
    """ID for one verifier call on a pair, criterion, repetition, and display order."""
    return stable_hash(
        "verification",
        pair_id_,
        canonical_json(dict(verifier_config)),
        criterion,
        repetition,
        display_order,
    )


def ranking_id(task_id: str, pool_hash: str, selector_config: Mapping[str, object]) -> str:
    """ID for a ranking of one candidate pool under a selector."""
    return stable_hash("ranking", task_id, pool_hash, canonical_json(dict(selector_config)))
