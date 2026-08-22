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

- `65b3dc1` — `feat(verification): add avt doctor score-logprob diagnostics
  (G=5)`.

### Next action

Stage 5 — connect the Qwen actor to a Harbor-supported harness and produce one
complete, officially graded local trajectory.

---

## Stage 5 — Qwen actor integration

**Status:** complete.

### Work

- Selected Harbor agent `qwen-coder` (Alibaba Qwen Code CLI) as the actor.
- Found Qwen3.8 thinking mode makes Qwen Code generate without terminating
  (unbounded reasoning until the stream-lifetime cap).
- Added `configs/qwen-settings.json` to disable thinking
  (`generationConfig.extra_body.chat_template_kwargs.enable_thinking=false`)
  and bound output tokens (`samplingParams.max_tokens=8192`).
- Injected it via a Harbor bind mount + `QWEN_CODE_SYSTEM_SETTINGS_PATH`.
- Ran one complete Qwen actor trajectory on `cancel-async-tasks`.

### Decisions

- Use the `qwen-coder` agent with the local endpoint
  (`OPENAI_BASE_URL=http://100.86.3.45:8888/v1`, model `qwen3.8-27b-sglang`).
- Disable thinking for the generation phase; it is a separate concern from the
  verification phase (which needs per-token logprobs).

### Checks

- Trial `cancel-async-tasks__9MeAdy2`: no agent exception, official grader
  reward `1.0`.
- Qwen Code validated against the endpoint with the settings in ~45 s.

### Commit

- `58fa6f9` — `feat(generation): connect Qwen actor via Harbor qwen-coder
  (reward 1.0)`.

### Next action

Stage 6 — implement storage and deterministic identifiers (two DBs and an
artifact store; records survive restart and resume).

---

## Stage 6 — Storage and deterministic identifiers

**Status:** complete.

### Work

- Implemented deterministic content-addressed identifiers
  (`src/avt/storage/ids.py`) with order-independent canonical serialization.
- Created the experiment catalog DB (`experiment.sqlite`) and the isolated
  ground-truth DB (`ground_truth.sqlite`) with foreign-key enforcement.
- Created the filesystem artifact store with atomic (temp+rename) writes.
- Implemented resumable job state with crash recovery of RUNNING jobs and
  atomic single-statement claiming.

### Decisions

- `pairs` stores only canonical unordered membership; A/B display order lives
  per `verifications` row (so a pair can be presented in either order for
  position-sensitivity runs).
- Foreign keys are enforced per connection (`PRAGMA foreign_keys=ON`).
- Crash-left `RUNNING` jobs are reclaimed to `RETRYABLE_FAILED` by an explicit
  `Catalog.recover_interrupted()` at controller startup — never implicitly on
  every connection open.
- `claim_job` is a single conditional UPDATE (`BEGIN IMMEDIATE`) so concurrent
  claims cannot both win.
- Ground truth is stored in a separate file never opened by verification.

### Checks

- `uv run pytest -q` — 13 passed (IDs, persistence across reopen, atomic
  claims, crash recovery + resume without duplicate rows, artifact writes).
- `uv run ruff check .` / `ruff format --check .` / `mypy .` — clean.

### Commit

- `6b079cc` — `feat(storage): add catalog, ground-truth, artifact store, and
  deterministic IDs`.

### Next action

Stage 7 — implement candidate generation (resumable generation jobs yielding
three usable candidates per smoke task).

## Stage 7 — Candidate generation

**Status:** implementation complete (committed); smoke candidate run in progress
(live 4-way parallel on the x86_64 workstation Docker; every candidate's model
calls hit the spark1 endpoint).

### Work

- Implemented frozen config loading (`config.py`) with `${VAR}` environment
  interpolation and typed accessors.
- Implemented deterministic seeded task selection (`selection.py`).
- Implemented the generation service (`generation.py`): persisted, resumable
  candidate jobs, bounded retries, artifact + immutable ground-truth indexing,
  config-derived Qwen settings, crash-left RUNNING reclaim at startup, UTF-8
  subprocess env, and configurable agent-timeout multiplier.
- Added `avt select-tasks`, `avt generate`; selected smoke tasks
  `distribution-search`, `path-tracing-reverse` (seed 42).
- Indexed each graded candidate in the frozen-pool manifest (`candidates`
  catalog table, with an ensured experiment row) so the Stage 8 pair builder can
  discover the pool; the write is idempotent, tested, and backfilled for
  candidates that succeeded before the fix.
- Verified and documented the execution-host constraint: the DGX Spark hosts
  (`spark1`/`spark2`/`spark3`) are `aarch64`, but the Terminal-Bench prebuilt
  task image we probed (`cancel-async-tasks`) is `linux/amd64` only; a Harbor
  trial on spark1 fails at environment start with a platform mismatch. Harbor
  generation therefore runs on the x86_64 workstation; the Spark hosts serve the
  model only. See `docs/terminal-bench-integration.md`.

### Checks

- 25 tests pass; ruff check/format clean, mypy clean.
- Generation job lifecycle (retry, resume, reuse, timeout rejection) covered.
- Candidate-manifest indexing covered (success is indexed; a never-graded
  candidate is not).

### Findings / blockers

- The selected smoke tasks are expert-difficulty; the 27B actor exceeds the
  default agent timeout. A no-practical timeout (`timeout_multiplier` large) is
  used so candidates run to a graded finish.
- Smoke progress so far: `distribution-search` attempts 0 and 1 graded
  `reward 1.0`; the remaining four candidates (including all three
  `path-tracing-reverse`) are still running. The three-candidates-per-smoke-task
  acceptance is **not yet met** pending the finished live run.

### Commit

- `6b079cc`, `87eb8b9` (Stage 6/7 init), `6e4d949` (reclaim fix),
  `57f4063` (UTF-8 fix), `ec86d06` (timeout multiplier), `60e5f4e` (parallel
  with no practical timeout), `1633fdf` (Stage-7 status docs).
- (manifest-indexing fix and this status update are in the current working
  commit)

### RTX pivot — Stage 7 acceptance met

**Status:** complete. Six candidates (three per smoke task) generated and
officially graded on the local RTX 6000 Pro.

#### Decisions

- **Host pivot.** The Spark DGX hosts are `aarch64` but the task images are
  `linux/amd64`, so generation stays on the x86_64 workstation Docker. Model
  serving moved from spark1 to the workstation's own **RTX PRO 6000** via a
  local LMW deployment, so generation no longer depends on the Spark model.
- **Deployment.** WSL-specific SGLang crash (multimodal CUDA IPC returning
  `invalid resource handle`) fixed by `--mm-feature-transport cpu` in the pinned
  recipe `qwen38-27b-radixark-nvfp4-dflash2-avt` (bumped to `1.0.1`);
  `lmw recipe validate` reports `valid:true`. Endpoint verified:
  `qwen3.8-27b-6000pro` (SGLang, 262144 context), incl. a thinking-disabled
  completion returning `AVTREADY-OK`.
- **In-container endpoint.** The qwen actor runs inside each task container, so
  loopback is the container. Generation uses
  `http://host.docker.internal:8000/v1` (verified from a throwaway container).
- **Context-window mismatch (root cause of the earlier `path-tracing-reverse`
  failures).** qwen-code assumes a 1,000,000-token window while the server
  ceiling is 262,144; the actor replays full history, so long runs cross the
  ceiling and SGLang returns HTTP 400 (`[API Error: 400 (no body)]`), after
  which qwen aborts via `UnknownApiError`/`NonZeroAgentExitCodeError`. Fixed by
  setting `model.generationConfig.contextWindowSize=200000` in the mounted qwen
  settings (now part of the frozen config via `generator.context_window`) so
  auto-compaction triggers before the ceiling. Verified a 220k-token request
  returns 200.
- **Task reselection (user-approved).** `path-tracing-reverse` sits past the 27B
  actor's capability (most runs abort via the agent guard; only a rare clean
  stop yields a usable `0.0`). Per explicit user decision (outside the frozen
  pool), replaced it with `cancel-async-tasks` (known feasible — the Stage 5
  actor run graded `1.0`). New experiment `smoke-rtx-v3` with its own storage
  root; the `smoke-v1` (spark1), `smoke-rtx` (v1), and `smoke-rtx-v2` pools are
  preserved untouched.
- **Subprocess UTF-8 decode.** `subprocess.run(..., text=True)` crashed on
  non-cp1252 bytes; fixed with `encoding="utf-8", errors="replace"`.

#### Candidates (smoke-rtx-v3, exp `6a774416…`)

- `distribution-search`: attempts 0,1,2 → reward `1.0, 1.0, 1.0`
- `cancel-async-tasks`: attempts 0,1,2 → reward `0.0, 0.0, 1.0`
- All six candidates SUCCEEDED with an official verifier reward
  (`official_results`), satisfying the Stage 7 acceptance ("three candidates for
  each smoke task").

#### Checks

- `uv run pytest -q` — 26 passed (incl. an idempotent-rerun test).
- `uv run ruff check .` / `ruff format --check .` — clean (scratch files removed).
- `uv run mypy .` — no issues (16 source files).

#### Commit

- Code: `src/avt/config.py` (`generator.context_window`), `src/avt/generation.py`
  (`contextWindowSize` in mounted qwen settings; UTF-8 subprocess decode;
  recovered frozen reward on idempotent rerun of a complete pool).
- Frozen configs: `experiments/smoke-rtx.yaml` (v1),
  `experiments/smoke-rtx-v2.yaml`, `experiments/smoke-rtx-v3-tasks.txt`,
  `experiments/smoke-rtx-v3.yaml`.

### Reproducibility follow-up — actor version pinned

Harbor installs the qwen-code agent as `@latest` unless a version kwarg is
passed; the v3 trials recorded `0.22.0`. `generator.agent_version` now defaults
to `"0.22.0"` (a code default, so the frozen `smoke-rtx-v3` experiment id is
unchanged) and is passed to Harbor as `--agent-kwarg version=0.22.0`, so a
rerun of any experiment cannot silently install a different actor. Covered by
`test_generation_pins_agent_version` (27 tests pass).

## Stage 8 — Safe renderer and pair builder

**Status:** complete. Frozen pair records built for `smoke-rtx-v3`; leakage and
unit tests pass.

### Work

- `src/avt/rendering.py`: deterministic, verifier-safe trajectory renderer.
  Reads a candidate's Harbor ATIF (`agent/trajectory.json`) only — never
  `verifier/` artifacts (hidden tests, reward), reference solutions, or
  pass/fail labels. Preserves the full public instruction (user-source steps)
  verbatim; renders agent message, tool calls, and embedded tool outputs in
  chronological order; head+tail truncates oversized bodies with an explicit
  `[OUTPUT TRUNCATED: original_tokens=…, retained=head:…+tail:…]` marker; records
  original vs rendered token counts (deterministic chars/token proxy, applied
  symmetrically to A and B). qwen ATIF embeds `observation.results` on the agent
  step.
- `src/avt/pairs.py`: `PairBuilder` builds all unordered pairs per task from the
  frozen SUCCEEDED pool, persists them idempotently, and records the task
  instruction. `pairs.candidate_a/b` store the canonical sorted membership
  (Stage 6 design); deterministic A/B display order is delegated to
  `display_order()` for per-verification `verifications.display_order` rows.
- `src/avt/storage/catalog.py`: added `record_task`, `get_task_instruction`,
  `list_candidates`, `record_pair`, `list_pairs`.
- CLI: `avt build-pairs --config experiments/smoke-rtx-v3.yaml`.

### Checks

- `uv run pytest -q` — 38 passed (11 new: renderer determinism/symmetry,
  instruction preservation, head/tail truncation, token counts, verifier-payload
  leakage; pair build count/idempotency/unordered-identity/task-skip/order).
- `uv run ruff check .` / `ruff format --check .` — clean.
- `uv run mypy .` — no issues (20 source files).
- Verified on the real pool: `avt build-pairs` builds 6 pairs
  (`distribution-search` 3, `cancel-async-tasks` 3), sorted canonical membership,
  task instruction stored (e.g. 1139 chars); a real trajectory renders to
  instruction + body with recorded token counts.

### Commit

- 6d11a7d, 1581b4d (Stage 7 + actor-version pin).

### Integrity hardening (follow-up)

`record_task`/`record_pair` are now immutable (insert-or-validate-identical:
an identical rewrite is a no-op, any conflict raises). `PairBuilder.build`
requires exactly `candidates_per_task` SUCCEEDED candidates and a nonempty
public instruction per task, else it fails loudly and never marks `PAIRED` from
an incomplete pool. Covered by conflict and incomplete-pool tests (41 pass,
ruff/mypy clean).

## Stage 9 — Discrete Qwen judge

**Status:** complete. Every smoke pair received a valid score.

### Work

- `src/avt/verification.py`: discrete Qwen judge (plan §13.1-13.3). One model
  request per (pair, criterion) reads the probability the model places on the
  single-token score labels `"1".."5"` (G=5) for trajectory A and trajectory B
  from the response `top_logprobs`; the discrete score is each trajectory's
  highest-probability label. The request contains only the public task
  description and the two rendered trajectories — no grader outcome, hidden
  tests, reference solution, or pass/fail labels.
- Score labels validated on the RTX endpoint: SGLang returns per-token
  `top_logprobs` including all five labels (single-token). Missing-label
  probabilities raise `MissingLabelError` (configuration failure, never silent
  zero), per §13.2.
- The judge runs host-side, so `host.docker.internal` (a container-only alias) is
  mapped to `127.0.0.1`; the frozen config / experiment id stays stable.
- Paired prompt uses the stored public task instruction (Stage 8) and each
  candidate's rendered body (Stage 8 renderer).
- `avt verify-pairs --config …`; per-(pair,criterion) records persisted to
  `verifications` with request/response/scores artifacts (immutable).
- `avt.pairs.display_order` assigns the deterministic A/B display order stored in
  each verification's `display_order` column.

### Checks

- `uv run pytest -q` — 49 passed (8 new: prompt content/no-grader, endpoint
  normalization, logprob score extraction, separator-skip, missing-label config
  failure, malformed verifier, judge verify+persist).
- `uv run ruff check .` / `ruff format --check .` — clean.
- `uv run mypy .` — no issues (22 source files).
- Verified on the frozen `smoke-rtx-v3` pool: `avt verify-pairs` produced 18
  verification records (6 pairs × 3 criteria), all `SUCCEEDED` with valid
  integer scores; experiment stage advanced to `VERIFIED`.

### Commit

- Stage 9: `src/avt/verification.py`, `src/avt/storage/catalog.py`
  (`record_verification`), `src/avt/cli.py` (`verify-pairs`), tests.

### Next action

Proceed to Stage 10 (continuous Qwen verifier) using the frozen `smoke-rtx-v3`
pairs.
