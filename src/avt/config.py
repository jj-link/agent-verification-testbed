"""Frozen experiment configuration loading.

Loads the project's YAML experiment configuration, interpolating ``${VAR}``
from the environment (secrets stay in the environment, never the file).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import yaml

__all__ = ["Config", "load_config"]

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate(value: object) -> object:
    if isinstance(value, str):

        def repl(m: re.Match[str]) -> str:
            name = m.group(1)
            if name not in os.environ:
                raise ValueError(f"configuration references unset env var {name!r}")
            return os.environ[name]

        return _ENV_RE.sub(repl, value)
    if isinstance(value, list):
        return [_interpolate(v) for v in value]
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    return value


@dataclass(frozen=True)
class Experiment:
    name: str
    seed: int
    task_file: str
    candidates_per_task: int
    max_parallel: int = 1


@dataclass(frozen=True)
class Generator:
    harness: str
    model: str
    endpoint: str
    temperature: float
    max_tokens: int
    timeout_multiplier: float = 1.0
    context_window: int | None = None
    agent_version: str = "0.22.0"


@dataclass(frozen=True)
class Verifier:
    model: str
    endpoint: str
    criteria: tuple[str, ...]
    granularity: int
    repetitions: int
    max_tokens: int = 16


@dataclass(frozen=True)
class Storage:
    root: str
    metadata_db: str
    ground_truth_db: str


@dataclass(frozen=True)
class Config:
    experiment: Experiment
    generator: Generator
    verifier: Verifier
    storage: Storage
    upstream: dict[str, object] = field(default_factory=dict)
    rendering: dict[str, object] = field(default_factory=dict)
    ranking: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)


def _section(cfg: dict[str, object], name: str) -> dict[str, object]:
    value = cfg.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"configuration missing [{name}] section")
    return value


def _str(d: dict[str, object], key: str) -> str:
    value = d[key]
    assert isinstance(value, str)
    return value


def _int(d: dict[str, object], key: str) -> int:
    value = d[key]
    assert isinstance(value, (int, float, str))
    return int(value)


def _float(d: dict[str, object], key: str) -> float:
    value = d[key]
    assert isinstance(value, (int, float, str))
    return float(value)


def _str_list(d: dict[str, object], key: str) -> tuple[str, ...]:
    value = d[key]
    assert isinstance(value, list)
    return tuple(str(item) for item in value)


def load_config(path: str | Path) -> Config:
    loaded: object = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"experiment configuration {path} is not a mapping")
    raw = cast(dict[str, object], _interpolate(loaded))

    exp = _section(raw, "experiment")
    gen = _section(raw, "generator")
    ver = _section(raw, "verifier")
    sto = _section(raw, "storage")

    return Config(
        experiment=Experiment(
            name=_str(exp, "name"),
            seed=_int(exp, "seed"),
            task_file=_str(exp, "task_file"),
            candidates_per_task=_int(exp, "candidates_per_task"),
            max_parallel=_int(exp, "max_parallel") if "max_parallel" in exp else 1,
        ),
        generator=Generator(
            harness=_str(gen, "harness"),
            model=_str(gen, "model"),
            endpoint=_str(gen, "endpoint"),
            temperature=_float(gen, "temperature"),
            max_tokens=_int(gen, "max_tokens"),
            timeout_multiplier=_float(gen, "timeout_multiplier")
            if "timeout_multiplier" in gen
            else 1.0,
            context_window=_int(gen, "context_window") if "context_window" in gen else None,
            agent_version=_str(gen, "agent_version") if "agent_version" in gen else "0.22.0",
        ),
        verifier=Verifier(
            model=_str(ver, "model"),
            endpoint=_str(ver, "endpoint"),
            criteria=_str_list(ver, "criteria"),
            granularity=_int(ver, "granularity"),
            repetitions=_int(ver, "repetitions"),
            max_tokens=_int(ver, "max_tokens") if "max_tokens" in ver else 16,
        ),
        storage=Storage(
            root=_str(sto, "root"),
            metadata_db=_str(sto, "metadata_db"),
            ground_truth_db=_str(sto, "ground_truth_db"),
        ),
        upstream=_section(raw, "upstream"),
        rendering=_section(raw, "rendering"),
        ranking=_section(raw, "ranking"),
        raw=raw,
    )
