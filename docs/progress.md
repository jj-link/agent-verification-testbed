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

- `a0c2d75` (previous); Stage 2 commit to follow.

### Next action

Stage 3 — inspect the local Qwen deployment and record exact model, server, and
hardware identity.
