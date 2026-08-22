# AVT Development Progress

This log records milestone completion, decisions, checks, commit SHAs,
blockers, and the next action in line with the plan.

- Source of truth: [`docs/avt-plan.md`](docs/avt-plan.md).

## Stage 1 — Public repository foundation

**Status:** complete.

### Work

- Python package metadata (`pyproject.toml`), `hatchling` build, `uv` managed.
- Installable `avt` CLI entrypoint with `--version` and a `doctor` placeholder.
- `.env.example`, `LICENSE` (MIT), and `.gitignore`.
- `README.md`, `HANDOFF.md`, and this progress log.
- GitHub Actions CI across Python 3.11 and 3.13.
- CLI contract tests.
- Public repository created at GitHub under `jj-link`.
- Fresh-clone-equivalent install verified.

### Decisions

- Python `>= 3.11`, `uv` for dependency management.
- Lint/format via Ruff; type check via mypy (strict); tests via pytest.
- Ruff excludes `docs/` because the formatter lints the Python code fences
  inside the plan document.
- MIT license for the repository.
- Git pushes use HTTPS.

### Checks

- `uv sync --dev` resolves and builds the package.
- `uv run avt --version` prints `avt 0.1.0`.
- `uv run pytest -q` — 3 passed.
- `uv run ruff check .` — clean.
- `uv run ruff format --check .` — clean.
- `uv run mypy .` — no issues.

### Commit

- `35b0a8d` (docs-only init).
- `a0c2d75` — `feat: add AVT package, CLI, CI, and documentation foundation`.

### Next action

Stage 2 — inspect and pin the Terminal-Bench and Harbor integrations, and run
one official baseline task through the official grader.

---

## Stage 2 — Terminal-Bench / Harbor integration

**Status:** complete.

### Work

- Identified the official stack: Harbor (`harbor-framework/harbor`) as the
  harness and Terminal-Bench 2.0 (`laude-institute/terminal-bench-2`) as the
  dataset.
- Pinned Harbor `0.22.0` (source commit `4407eb52`) in the project's `tbench`
  extra; the `terminal-bench@2.0` dataset resolves to commit `69671fba`.
- Selected baseline task `cancel-async-tasks` (without consulting outcomes).
- Ran it through the official oracle agent; graded reward `1.0`.
- Reproduced the run through the pinned project env.
- Documented the integration and output format in
  `docs/terminal-bench-integration.md`; recorded pins in `.env.example`.

### Decisions

- Harbor requires Python >= 3.12, so it is a `[project.optional-dependencies]`
  `tbench` extra with a `python_version >= '3.12'` marker; the base package
  keeps `requires-python >= 3.11` and ordinary CI stays free of Harbor.
- No paid API is needed for the oracle baseline; it runs on local Docker.

### Checks

- `uv run harbor --version` → `0.22.0`.
- Oracle run: 1 trial, 0 errors, reward `1.0` (two independent runs).
- Docker Desktop running locally for task containers.

### Commit

- `8c25000` — `feat(tbench): pin Harbor 0.22.0 and document Terminal-Bench 2.0
  integration`.

### Next action

Stage 3 — inspect the local Qwen deployment and record exact model, server, and
hardware identity.

---

## Stage 3 — Local Qwen deployment

**Status:** complete.

### Work

- Inventoried `spark1` (DGX Spark / GB10) hardware and incumbent services.
- Replaced the incumbent Qwen3.6-35B server with the plan's recipe deployment:
  `RadixArk/Qwen3.8-27B-NVFP4` served via SGLang DSpark on port 8888.
- Recorded exact model, checkpoint revision, serving image digest, recipe
  commit, context, and hardware in `docs/qwen-deployment.md`.
- Verified the endpoint from the workstation over the private network.

### Decisions

- Used the recipe's `start-dspark.sh` with native 262K context, YaRN off, and
  `MAX_CONCURRENT_REQUESTS=4`.
- Took down the prior `Qwen3.6-35B` server (authorized by plan); its full
  `docker inspect` snapshot is retained on `spark1` for rollback.
- AVT client endpoint is `http://100.86.3.45:8888/v1` (loopback is on-host
  health-check only), reflected in `.env.example`.

### Checks

- From the workstation: `/v1/models` → `qwen3.8-27b-sglang`,
  `max_model_len 262144`.
- Chat completion round-trip succeeded.
- Checkpoint revision served: `91cea059647696fd83964e43d57db122ff745993`.

### Commit

- `63d6d9e` — `feat(models): record and verify spark1 Qwen3.8-27B deployment`.

### Next action

Stage 4 — validate score-logprob capability (G=5 labels scoreable), via a
diagnostic against this endpoint.

---

## Stage 4 — Score-logprob validation

**Status:** complete.

### Work

- Probed the endpoint for token-logprob support using letter-based single-token
  score labels.
- Found DSpark cannot return logprobs; switched the deployment to the MTP
  launcher (same recipe/checkpoint/context).
- Implemented a real `avt doctor` diagnostic (stdlib-only) verifying endpoint
  identity, logprob access, and single-position G=5 label coverage.
- Verified `avt doctor` passes against the live deployment.
- Recorded findings in `docs/serving-diagnostics.md` and
  `docs/qwen-deployment.md`.

### Decisions

- Score labels for `G=5`: ordered single tokens `A B C D E`.
- Use the MTP launcher (logprob-capable) rather than DSpark for the AVT
  verification pipeline.
- `avt` with no subcommand prints help; `avt doctor` runs diagnostics.

### Checks

- `uv run avt doctor` → all four checks PASS (exit 0).
- `uv run pytest -q` → 3 passed; ruff check/format clean; mypy clean.

### Commit

- `63d6d9e` (previous); Stage 4 commit to follow.

### Next action

Stage 5 — connect the Qwen actor to a Harbor-supported harness and produce one
complete, officially graded local trajectory.
