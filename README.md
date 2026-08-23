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

## Results (main study, `experiment-v1.1`)

25 Terminal-Bench tasks × 5 candidates, verified by a local `qwen3.8-27b-6000pro`
G=5 judge, ranked by three leakage-safe local selectors on identical frozen pools.

Pool base pass rate: **20.8%** (26/125).

| Selector | Top-pass rate | 95% CI |
|---|---:|---:|
| random (uniform expectation) | 20.8% | [9.6%, 34.4%] |
| discrete (argmax) | 44.0% (11/25) | [24.0%, 64.0%] |
| continuous (expectation) | 44.0% (11/25) | [24.0%, 64.0%] |

Continuous (and discrete) selected top **+0.232** mean reward over the uniform
random baseline, task-bootstrap 95% CI **[0.104, 0.368]** (excludes zero). On the
42 polarized pairs the per-pair verifier accuracy was 1.00 (continuous) and 0.95
(discrete) — small-n caution applies.

Full report and figures: [`results/REPORT.md`](results/REPORT.md). Regenerate
with `python analysis/generate_figures.py --config experiments/frozen_main.yaml
--root /home/workbench/avt-data/main-v1`.
