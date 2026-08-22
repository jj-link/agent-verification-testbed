# Agent Verification Testbed (AVT)

Reproducible evaluation of methods for selecting the best agent trajectory
from multiple candidate runs, built around Terminal-Bench 2.0 and a local LLM
verifier.

## Status

Development follows the staged plan in [`docs/avt-plan.md`](docs/avt-plan.md).
Current stage and progress are tracked in [`docs/progress.md`](docs/progress.md).

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv).

```bash
uv sync --dev
```

## Development

```bash
uv run avt --version
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy .
```

## Layout

`src/avt/` contains the implementation. Terminal-Bench and Harbor integrations,
experiment definitions, and analysis live under their respective directories as
stages complete. See [`docs/avt-plan.md`](docs/avt-plan.md) for the full
structure and methodology.
