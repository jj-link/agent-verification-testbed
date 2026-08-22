# Terminal-Bench 2.0 / Harbor Integration

This note records how AVT pins and runs the official upstream stack, the exact
revisions used, the baseline result, and how to reproduce the run.

## Requirement

Stage 2 acceptance: one official baseline task runs and grades successfully via
the official Terminal-Bench/Harbor stack.

## Pinned revisions

| Component | Pin | Identity |
|---|---|---|
| Harbor | `0.22.0` | PyPI package `harbor==0.22.0`; source tag `v0.22.0` → commit `4407eb5227a2ff4f0d3f16b2eb48849382fdf276` |
| Terminal-Bench 2.0 | Dataset `terminal-bench@2.0` | Task repo `laude-institute/terminal-bench-2`, resolved commit `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c` ("update storage") |

Harbor pins in `pyproject.toml` under the `tbench` extra:
`harbor==0.22.0; python_version >= '3.12'`. Harbor requires Python >= 3.12.

The `terminal-bench@2.0` resolution to `69671fba` is recorded by Harbor in each
trial's `result.json` under `task_id` and `config.task`, and by the task checksum.

## Baseline task

- `cancel-async-tasks` (async concurrency / cancellation, Python,
  category `software-engineering`)
- Instruction: `docs/`-level task in the pinned commit; writes `/app/run.py`
  with an `async run_tasks(...)` implementation.
- Selected without consulting task outcomes; it exercises a normal
  agent-environment + official verifier cycle.

## Reproducible run command

```bash
uv sync --extra tbench
uv run harbor run \
  -d terminal-bench@2.0 \
  -a oracle \
  -i cancel-async-tasks \
  -n 1 \
  -k 1 \
  -o .avt/upstream-baseline \
  --job-name stage2-baseline \
  --yes
```

Requires a running Docker daemon. No model or paid API is needed for the oracle
run.

## Baseline result

- Job: `6056a02c-fe65-4995-bdd2-0157a6854b54`
- Trial: `cancel-async-tasks__L5tukP8`
- Stats: 1 trial, 1 completed, 0 errors, 0 retries
- Reward: `1.0` (from `verifier_result.rewards.reward`)
- Total runtime: ~32s
- Records under `.avt/upstream-baseline/stage2-baseline/`

## Output format observed

- `result.json` (job): aggregate `stats`, per-eval `reward_stats`,
  `pass_at_k`, token/cost fields (null for the oracle).
- `result.json` (trial): `task_id.gite_commit_id`, `task_checksum`,
  `config`, `agent_result`, `verifier_result.rewards`, timings.
- `verifier/reward.txt`, `verifier/test-stdout.txt`, `verifier/ctrf.json`.
- `agent/` holds the agent's own artifact; the oracle leaves no trajectory.

## Host platform constraint (verified)

Terminal-Bench 2.0 task environments are prebuilt container images. The one we
probed, `alexgshaw/cancel-async-tasks:20251031`, is published as
**`linux/amd64`** only (single-arch; no `arm64` manifest). The qwen-code actor
runs *inside* the task environment image (Node + npm are installed in-container
during agent setup), so it is architecture-bound to the task image's platform.

The DGX Spark hosts (`spark1`/`spark2`/`spark3`, NVIDIA GB10) are **`aarch64`**
(`uname -m` → `aarch64`). A Harbor trial on spark1 fails at environment start:

```
The requested image's platform (linux/amd64) does not match the detected host
platform (linux/arm64/v8) and no specific platform was requested
Container cancel-async-tasks__<trial>__env-main-1 exited (255)
```

Verified directly: `alexgshaw/cancel-async-tasks:20251031` reports
`amd64/linux` and has no arm64 variant; an oracle probe job on spark1 errored
with the platform mismatch above.

Consequence: official Harbor generation must run on an **x86_64 Docker host**;
the Spark machines serve the model, not the trial containers. AVT keeps
generation on the x86_64 workstation Docker, parallelized with
`experiment.max_parallel` (4 for the smoke run), and points every candidate's
model calls at the spark1 endpoint. QEMU `qemu-x86_64` emulation is
registerable on the Spark hosts, but running multi-hour agent+environment
containers under emulation is not viable for the study and is not used.
