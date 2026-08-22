# Handoff: Agent Verification Testbed (AVT)

This handoff summarizes how the repository is structured, what the project is
for, and how to continue implementation. The authoritative technical plan is
[`docs/avt-plan.md`](docs/avt-plan.md); keep that document and this one in sync.

## Purpose

AVT is an experiment controller around Terminal-Bench 2.0. It runs a local LLM
multiple times on tasks, saves the completed trajectories, and compares methods
for selecting the best trajectory:

- first candidate (baseline)
- random candidate
- Qwen discrete judge
- Qwen continuous verifier (probabilistic score-token expectation)
- oracle (analysis-only upper bound)

The initial verifier uses the same checkpoint as the generator with separate
prompts and fresh contexts. Later stages add a frontier-assisted verifier.

## Key constraints

- `docs/avt-plan.md` is the source of truth.
- Implementation proceeds through the plan's stages in order.
- Official grader results are isolated and never used during verification,
  ranking, or tie-breaking.
- Candidate pools and frozen experiment configurations are immutable.
- No paid APIs are required by CI.

## Repository layout

- `src/avt/` — implementation package with an `avt` CLI.
- `docs/` — plan, methodology, integration notes, and progress log.
- `tests/` — unit and integration tests.
- `experiments/` — smoke, pilot, and main configurations and task lists.
- `configs/` — shared runtime configuration.

## Operating instructions

- Work one staged milestone at a time; verify its acceptance criterion before
  advancing.
- Prefer specific file-level edits and focused changes over broad rewrites.
- Update `docs/progress.md` after each milestone with decisions, checks,
  commit SHA, blockers, and next action.
- Use HTTP(S) for Git pushes to GitHub.
- Never commit `.env`, credentials, model weights, private data, or
  unauthorized benchmark artifacts.

## Current status

Stage 1 (public repository foundation: package, CI, README, handoff) is the
current milestone. See `docs/progress.md` for the latest state and next action.
