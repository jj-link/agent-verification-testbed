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

- `35b0a8d` (previous docs-only init); Stage 1 foundation commit to follow.

### Next action

Stage 2 — inspect and pin the Terminal-Bench and Harbor integrations, and run
one official baseline task through the official grader.
