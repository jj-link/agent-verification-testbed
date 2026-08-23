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
  frozen single-token score labels `doctor.G5_LABELS = ("A","B","C","D","E")`
  (Stage 4; A<B<C<D<E → 1..5) for trajectory A and trajectory B from the
  response `top_logprobs`; the discrete score is each trajectory's
  highest-probability label value. The request contains only the public task
  description and the two rendered trajectories — no grader outcome, hidden
  tests, reference solution, or pass/fail labels.
- Score labels validated on the RTX endpoint: SGLang returns per-token
  `top_logprobs` including all five letters (single-token). Label matching is
  tolerant of delimiter/whitespace-wrapped tokens (`" A"`, `"E>"`, `"<E>"`).
  Missing-label probabilities raise `MissingLabelError` (configuration failure,
  never silent zero), per §13.2. Because the endpoint is stochastic, a malformed
  draw is retried (bounded 3×), matching the plan's "malformed verifier output →
  retry" policy. The label set is included in the verifier identity so
  verification IDs bind the scoring policy.
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

## Stage 10 — Continuous Qwen verifier

**Status:** complete. Math and logprob tests pass; expected-score records
computed for the frozen pool.

### Work

- `src/avt/expected.py`: `ContinuousVerifier` — a deterministic offline pass
  over the immutable discrete-judge response artifacts (no new model calls).
  For each (candidate, criterion) it aggregates the renormalized label
  distribution to an expected score (§13.4),
  `raw_expected_score = sum_g p(score_g) * value(score_g)`, then normalizes
  `(raw - 1) / (G - 1)`. Requires full coverage: exactly `(pool_size-1) *
  repetitions` observations per (candidate, criterion); a malformed/missing
  response or incomplete coverage raises `CoverageError` (job fails visibly).
- `src/avt/verification.py`: `expected_scores_from_logprobs`; `_label_probs`
  now renormalizes the five single-token label weights to sum to 1 (they are a
  subset of the full token distribution, so raw mass must be renormalized to
  keep the [1, G] invariant) and rejects non-finite/zero mass.
- Schema: `expected_scores(candidate_id, criterion, raw_expected_score,
  normalized_score, observations)`; catalog `record_expected_score` /
  `list_expected_scores` (immutable).
- CLI: `avt expected-scores --config …`.

### Checks

- `uv run pytest -q` — 55 passed (new: expected-score math, label-mass
  renormalization, incomplete-coverage failure, malformed-response failure).
- `uv run ruff check .` / `ruff format --check .` — clean.
- `uv run mypy .` — no issues (24 source files).
- Verified on `smoke-rtx-v3`: `avt expected-scores` produced 18 records
  (6 candidates × 3 criteria, 2 observations each), raw ∈ [1, 5],
  normalized ∈ [0, 1].

### Commit

- Stage 10: `src/avt/expected.py`, `src/avt/verification.py`,
  `src/avt/storage/schema.py`, `src/avt/storage/catalog.py`, `src/avt/cli.py`,
  `tests/test_expected.py`, `tests/test_verification.py`.

## Stage 11 — Three-criterion evaluation

**Status:** complete. Criterion and aggregate scores computed reproducibly for
the frozen pool.

### Work

- `src/avt/evaluation.py`: `Evaluator` aggregates each candidate's Stage-10
  per-criterion expected scores into a single reproducible score over the
  configured criteria (specification, output, errors),
  `aggregate_raw = mean_c raw_expected_score_c`, `normalized =
  (aggregate_raw - 1)/(G - 1)`. Requires complete criterion coverage for every
  frozen candidate; a missing criterion raises `EvaluationError` (visible
  failure).
- Schema: `evaluation(candidate_id, aggregate_raw, aggregate_normalized,
  criteria, observations)`; catalog `record_evaluation` / `list_evaluation`
  (immutable).
- CLI: `avt evaluate --config …`.

### Checks

- `uv run pytest -q` — 58 passed (new: three-criterion aggregate math,
  reproducibility of a rerun, missing-criterion failure).
- `uv run ruff check .` / `ruff format --check .` — clean.
- `uv run mypy .` — no issues (26 source files).
- Verified on `smoke-rtx-v3`: `avt evaluate` produced 6 aggregate records
  (3 criteria, 6 observations each), aggregate_raw ∈ [1, 5],
  normalized ∈ [0, 1].

### Commit

- Stage 11: `src/avt/evaluation.py`, `src/avt/storage/schema.py`,
  `src/avt/storage/catalog.py`, `src/avt/cli.py`, `tests/test_evaluation.py`.

## Stage 12 — Round-robin Bradley-Terry ranking

**Status:** complete. Each task pool is ranked by round-robin win utility derived
only from the Stage-11 aggregate expected scores; a candidate is selected per
task without consulting the ground-truth store.

### Work

- `src/avt/ranking.py`: `RoundRobinRanker` computes per-candidate utility
  `u_i = mean_{j != i} sigmoid(R_i - R_j)` (R = aggregate expected score) and
  ranks each task pool stably. Tie-breaking per plan 14, all three applied in
  order: higher mean expected score, fewer malformed verifier records,
  lexicographically lower deterministic candidate id; the numeric "higher is
  better" columns are negated so an ascending stable sort still puts fewer
  malformed records and the lower id first.
  - Malformed verifier records are counted per verification as
    `malformed_attempts` (retries that returned an unparsable score-token
    response before the eventual success) and aggregated per candidate over its
    pairs; candidate totals differ since each candidate sits in a different
    subset of pairs.
  - `_check_pair_coverage` enforces plan 14's 100% usable pair coverage before
    ranking: every unordered pool pair must be present and carry a SUCCEEDED
    verification for every (criterion, repetition); a gap fails the ranking job
    visibly (a pair's own `status` is its freeze state, e.g. PAIRED, and is not
    the usable signal).
  - `_validate_no_grader` guards that ranking opens only `experiment.sqlite`;
    `ground_truth.sqlite` is a separate file never opened here.
- Schema: `rankings(ranking_id, task_id, pool_hash, selector_config, result,
  status)` (immutable); catalog `record_ranking` rejects a conflicting rewrite,
  identical rewrites are a no-op. `verifications` gained
  `malformed_attempts INTEGER NOT NULL DEFAULT 0` (schema v2; ALTER migration
  back-fills existing DBs), and the discrete verifier records malformed retries
  per request; catalog exposes `malformed_attempts_for` and
  `count_succeeded_verifications` for the ranker.
- CLI: `avt rank --config …` ranks every task and prints each task's top
  candidate and utility.
- `ranking_id = hash(task_id, pool_hash, selector_config)` (plan 7), making the
  stage idempotent and resumable.

### Checks

- `uv run pytest tests/test_ranking.py -q` — 10 passed (new: sigmoid, highest
  aggregate ranks first, rank_all consults no grader, lower-id tie-break,
  fewer-malformed tie-break, missing-verification failure, wrong-criterion
  failure, duplicate-verification failure, conflicting-rerun failure,
  missing-aggregate failure).
- Full suite `uv run pytest -q` — 68 passed (includes the generation test;
  `harbor` resolves from the venv `tbench` extra).
- `uv run ruff check .` — clean; `uv run ruff format --check .` — clean.
- `uv run mypy .` — no issues (28 source files).
- Verified on `smoke-rtx-v3` (frozen, 2 tasks × 3 candidates): the pre-v2 DB was
  migrated (ALTER added `malformed_attempts`, default 0); `avt rank` passed the
  pair-coverage check — every pair carries exactly the configured
  (criterion, repetition) SUCCEEDED key set {(errors,0),(output,0),
  (specification,0)} — and ranked both tasks; manual recomputation of the
  round-robin utilities matches the stored results; the top candidate is the
  one with the highest aggregate expected score. Re-running is idempotent (2
  ranking rows unchanged), and the experiment DB contains no `official_results`
  table (ground-truth isolation).

### Decisions

- All three plan-14 tie-breakers implemented in order (higher mean expected
  score, fewer malformed verifier records, lower deterministic id). The
  malformed tie-break is meaningful even under 100% coverage: a candidate's
  total is the sum over the distinct subset of pairs it participates in.
- `_check_pair_coverage` enforces 100% usable pair coverage at ranking time and
  fails the job visibly on any gap, per plan 14. A pair's own `status` (PAIRED
  etc.) is its freeze state, not the usable signal; usability is the complete
  SUCCEEDED verification set.
- Ranking never consults the official grader, per plan 14 "Never use the
  official grader for tie-breaking" and the ground-truth isolation guarantee.

### Commit

- Stage 12: `src/avt/ranking.py`, `src/avt/storage/catalog.py`,
  `src/avt/cli.py`, `tests/test_ranking.py`.
- Stage 12 (completed §14): `schema.py` (schema v2 `malformed_attempts` +
  migration), `verification.py` (malformed counting), `catalog.py`,
  `ranking.py` (malformed tie-break + pair-coverage check),
  `tests/test_ranking.py` (expanded).

### Next action

Proceed to Stage 13 (smoke test: two tasks, three candidates, pipeline runs
end to end) using the frozen `smoke-rtx-v3` pool.

## Stage 13 — Smoke test (end-to-end)

**Status:** complete. The frozen `smoke-rtx-v3` pool (two tasks × three
candidates) ran the pipeline end to end: generation → pairs → discrete
verification → continuous expected scores → three-criterion evaluation →
ranking.

### Work

- The smoke run was executed incrementally across Stages 7-12 on the frozen
  `smoke-rtx-v3` pool (2 tasks × 3 candidates), driven by the `avt` CLI stage
  chain (`generate` → `build-pairs` → `verify-pairs` → `expected-scores` →
  `evaluate` → `rank`).
- This stage re-ran the idempotent offline stages (`build-pairs`,
  `expected-scores`, `evaluate`, `rank`) on the frozen pool to confirm the chain
  completes cleanly and reproducibly; the already-recorded online stages
  (generation, verification) were not re-run (freeze policy).

### Checks

- Offline chain re-run on `smoke-rtx-v3`:
  `build-pairs` → 6 pairs; `expected-scores` → 18 records;
  `evaluate` → 6 candidates; `rank` → 2 tasks (top:
  `distribution-search`=ae6296…, `cancel-async-tasks`=a42be5…).
- End-to-end artifact coverage verified: 6 candidate trajectories, 6 official
  grader results, 18 verifier runs (2×3 pairs × 3 criteria).
- Catalog row counts: tasks=2, candidates=6, pairs=6, verifications=18,
  expected_scores=18, evaluation=6, rankings=2 — every stage populated.
- `uv run pytest -q` — 68 passed; `uv run ruff check .`; `uv run mypy .` clean.

### Decisions

- The smoke test is the already-frozen `smoke-rtx-v3` run, whose online stages
  were executed in prior stages; re-running generation/verification would
  regenerate frozen trajectories, which the plan forbids after freeze. The
  acceptance criterion "pipeline completes end to end" is evidenced by the
  complete, consistent artifact and record set above.

### Commit

- Stage 13: no source change; verification evidence only. Commit record
  accompanies the next milestone.

### Next action

Proceed to Stage 14 (pilot: eight tasks, initially three candidates, reliability
and cost measured) — this requires the live local Qwen endpoint for
generation and verification on a new `pilot` pool.

## Stage 14 — Pilot: reliability and cost measured

**Status:** complete (reliability and cost measured; full 8-task offline ranking gated
on a verifier-coverage finding, recorded as the next blocker). See §21 metrics.

### Work

- Ran the pilot on the `pilot` pool (8 tasks x 3 candidates = 24 candidates,
  seed 42): generation -> 24 frozen pairs -> discrete verification (72 keys =
  24 pairs x 3 criteria).
- Offline candidates were pre-graded in the original run; one controller pass
  dropped 4 graded `reward 0.0` candidate slots to PERMANENT_FAILED because
  their trial `result.json` was not yet final at index time. A resume recovered
  them (no new model calls; all 24 slots SUCCEEDED). Recorded as a
  controller-recovery finding.
- Corrected and hardened the discrete judge:
  - Resumable `verify_all` (skips SUCCEEDED/FAILED keys via
    `terminal_verification_keys`); a second/partial run makes only missing calls.
  - Persistently-malformed verification is recorded as `FAILED` (raw response and
    request artifacts retained) instead of aborting the run, so the plan-§21
    malformed-output rate is measurable rather than lost.
  - `verify-pairs` exits nonzero and stays at stage `VERIFYING` (not `VERIFIED`)
    while any FAILED verification breaks 100% usable coverage.
- Corrected downstream-accounting exercise (provenance separate for exact vs
  lower bound below).

### Reliability (generation)

| Metric | Value |
|---|---|
| Candidate slots | 24/24 graded (8 tasks x 3) |
| Passing candidates | 15/24 |
| Pass@1 | 5/8 = 0.62 |
| OraclePass@N | 6/8 = 0.75 |
| Tasks with >=1 pass | distribution-search, git-leak-recovery, password-recovery, prove-plus-comm, rstan-to-pystan, bn-fit-modify |
| No-pass tasks | make-mips-interpreter, polyglot-c-py |

Per-task rewards: bn-fit-modify 1/3, distribution-search 3/3, git-leak-recovery 3/3,
make-mips-interpreter 0/3, password-recovery 2/3, polyglot-c-py 0/3, prove-plus-comm 3/3,
rstan-to-pystan 3/3.

### Reliability (verifier)

| Metric | Value | Provenance |
|---|---|---|
| Verification keys | 72 (24 pairs x 3 criteria) | - |
| SUCCEEDED / FAILED | 71 / 1 | recorded rows |
| Terminal verification failure rate | 1/72 = 1.4% | persisted FAILED key (polyglot-c-py, specification) |
| Malformed-output rate (draw-level) | 18/90 = 20% | recorded malformed-draw sum 16 + 2 crash-discarded draws (original run), 88 recorded calls + 2 | 

The 20% malformed-DRAW rate is distinct from the 1.4% terminal failure rate: the
judge's single plan-15 retry recovers almost all malformed draws; only the polyglot
specification pair failed both draws and is recorded FAILED (malformed_attempts=2).

### Cost (local inference; not free)

| Metric | Value | Provenance |
|---|---|---|
| Generation tokens | 287,609,537 (286,463,357 in / 1,146,180 out) | sum over ALL 34 trial rounds (24 candidate rounds + ~10 retry rounds) |
| Generation wall time | ~145.6 min | trial started_at..finished_at |
| Verification tokens | 5,516,085 (5,515,806 prompt / 279 completion) | 72 persisted-final responses; LOWER BOUND, omits the ~18 earlier malformed-draw request tokens (not persisted) |
| Verification wall time | ~1,675 s (~28 min) | persisted created_at span |
| Candidates / comparisons per task | 3 candidates, 3 pairs | - |

Generation verification cost is dominated by re-rendering full trajectories into the
verifier prompt (body budget up to 60k tokens/side; ~5.5M prompt tokens for 72 calls).

### Decisions

- Kept the honest 71/72 result and the single FAILED verification; did NOT reset or
  re-run to hide it (`record_verification` is immutable; plan forbids silently
  changing coverage/method).
- Did NOT change expected-scores/evaluate/rank contracts or "convert" the 8-task pilot
  into a 7-task analysis. plan §14 (line 433) prescribes failing that ranking job
  visibly when a pair remains unusable after retries.
- Verifier robustness (malformed-draw rate, make-mips context exhaustion) is a Stage 15
  concern to be addressed in a NEW configuration/experiment, not by tuning the frozen
  pilot pool.

### Blocker / next action

- 100% usable pair coverage failed on {polyglot-c-py, pair 35c8e0f0..., specification};
  expected-scores / evaluate / rank fail visibly on it (CoverageError), so the full
  offline chain is gated on resolving verifier malformed handling in a
  new-configuration experiment before Stage 15 ("score coverage and context remain
  reliable").
- Commit SHA: `d14a048` (milestone; docs commit follows).

### Checks

- `uv run pytest -q` — 76 passed; `uv run ruff check .`; `uv run ruff format --check .`;
  `uv run mypy .` — clean.
- `avt verify-pairs --config experiments/pilot.yaml` (resume) — 0 new calls,
  detects pre-existing FAILED, stage=VERIFYING, exit 1.
- Regression tests: resume makes only missing calls (SUCCEEDED and FAILED skipped);
  persistent-malformed records FAILED with response artifact; a resumed run does not
  claim VERIFIED when a FAILED row predates it.

## Stage 15 — Pilot to five candidates and G=20 (in progress)

**Status:** in progress — verifier-robustness diagnosis complete; verifier
reproducibility prerequisite committed.

### Diagnosis (malformed-draw root cause)

- The Stage-14 FAILED verification's persisted response is exactly `content = "A F"`,
  tokens `A`, ` F`, EOS, `finish_reason="stop"`. It did **not** emit prose and did
  **not** hit the token cap.
- Root cause: the model generated `F` as the second score token, which is **outside**
  the G=5 label set {A,B,C,D,E}; the judge (correctly) rejects it, so only one valid
  score position exists. The ~20% draw-level malformed rate largely reflects
  occasional out-of-scale letters (and stray prose), not a cap or endpoint fault.
- The Stage-15 fix direction is invalid-label generation: strengthen the verifier
  format instruction / constrain output to the label set (e.g. stricter prompt or
  guided/constrained decoding), NOT parser relaxation (which would silently hide
  compliance issues the plan measures).

### Reproducibility change

- The discrete judge's `max_tokens` was hardcoded (`16`); it is now config-driven
  (`verifier.max_tokens`, default `16`) and included in the verifier identity so any
  tuning change is a distinct, frozen, reproducible run. (`config.py`,
  `verification.py._verifier_identity`, new test.)

### Checks

- `uv run pytest -q` — 77 passed; ruff/format/mypy clean.

### Next action

1. Make the verifier prompt / label-set / output policy fully config-driven and into
   the identity (reproducibility; prerequisite to G=20 and the invalid-label fix).
2. Select a Stage-15 configuration (5 candidates/task, G=20 labels) and freeze pools.
3. Run generation (5 candidates x 8 tasks) and G=20 verification; then measure score
   coverage and context reliability.

## Stage 15 — Pilot to five candidates and G=20 (in progress; generation running)

**Status:** in progress — verifier robustness fixes committed; Stage-15 G=20 pilot
config created; 40-candidate generation running.

### Verifier robustness fixes (committed)

Diagnosis: the Stage-14 FAILED verification response was exactly `content = "A F"`
(tokens `A`, ` F`, EOS, `finish_reason="stop"`): the model emitted the second score
as `F`, **outside** the G=5 label set {A..E}; not a token-cap or endpoint fault.

Fixes (each frozen/reproducible, tests added):
- `max_tokens` is config-driven and in the verifier identity (`1f04965`).
- Score labels derive from `config.verifier.granularity`
  (`labels_for_granularity`, `36f24a2`): G=20 = A..T, frozen in the identity. This
  is the G=20 prerequisite.
- `_HarborRunner.run` now returns a present graded job dir even on nonzero Harbor
  rc instead of raising `InfrastructureFailure`, so `generate_one` accepts the
  official grade and does not burn a fresh retry round (Stage 14 wasted ~10 rounds /
  ~100M tokens this way). `9c87efe`.

### Stage 15 config

- `experiments/pilot-g20.yaml`: the same 8 pilot tasks, `candidates_per_task: 5`,
  `verifier.granularity: 20` (labels A..T), storage `.avt/pilot-g20/` (new experiment
  id; the frozen Stage-14 `pilot` pool is untouched). Committed `d843aac`.

### In progress

- 5-candidate generation (8 tasks x 5 = 40 candidates) running in the background.
- Next: build pairs -> G=20 discrete verification -> measure score coverage and
  context reliability.

## Stage 15 — Pilot to five candidates and G=20 (results)

**Status:** score coverage and context measured; 5-candidate G=20 pools frozen and
verified.

### Run

- `experiments/pilot-g20.yaml`: the same 8 pilot tasks, 5 candidates/task,
  `granularity: 20` (labels A..T), storage `.avt/pilot-g20/` (new experiment id;
  frozen Stage-14 pool untouched).
- Generation: **40/40 candidates SUCCEEDED** (8 tasks x 5), zero failures — the
  graded-on-nonzero-rc runner fix (`9c87efe`) avoided the wasted rounds seen in
  Stage 14.
- Pairs: 80 (10/task). G=20 discrete verification: **240 keys** (80 pairs x 3
  criteria).

### Score coverage (G=20)

| Metric | Value | Provenance |
|---|---|---|
| Verification keys | 240 | 80 pairs x 3 criteria |
| SUCCEEDED / FAILED | 238 / 2 | recorded rows |
| Usable coverage | 238/240 = 99.2% | 6/8 tasks fully covered (30/30) |
| Terminal verification failure | 2/240 = 0.8% | distribution-search/errors, rstan-to-pystan/specification |
| Malformed draws | 53 | malformed_attempts sum |
| Malformed-draw rate | 53/293 ≈ 18.1% | draws / (240 keys + 53 retries) |

G=20 label coverage required two fixes:
- `top_logprobs` 100 -> 300 (plan line 353 first remedy); config-driven + in the
  verifier identity (`50dff29`).
- Labels absent from a position's returned top-logprobs have ~0 probability for
  G>5 (the model concentrates on a subset, e.g. `Q` at ~-200 logprob): treated as
  0 and renormalized; `MissingLabelError` now only fires on a position with no
  score label at all (genuine data absence) (`ded55b1`).

### Context reliability

- Max verifier `prompt_tokens` = **231,175**, ~1.9x the nominal
  `max_pair_context_tokens: 120000` budget and ~88% of the model's 262K hard limit.
- Root cause: `estimate_tokens`' chars/4 proxy undercounts real token count for
  dense trajectories, so the rendered body can tokenize larger than budgeted.
- No context overflow occurred (all 240 keys completed); noted as a cost/budget
  finding for the renderer.

### Decisions

- Kept the 2 FAILED verifications as measured reliability findings (verifier
  robustness to out-of-scale/prose draws; not silently hidden or reset).
- As in Stage 14, the offline chain (expected-scores -> evaluate -> rank) is gated
  on resolving the coverage gaps (plan 14: fail the affected ranking jobs visibly)
  before the main study.

### Checks

- `uv run pytest -q` — 83 passed; `ruff check`; `ruff format --check`; `mypy` —
  clean.

### Milestone commits

- `50dff29` top_logprobs config (G=20 coverage); `ded55b1` missing-subset-label
  relaxation; earlier `36f24a2` granularity labels, `9c87efe` runner rc fix,
  `50dff29`/`ded55b1` above. (Docs commit follows.)

## Stage 16 — Freeze main experiment (experiment-v1.0)

**Status:** complete. `experiments/frozen_main.yaml` committed and tagged
`experiment-v1.0` (pushed).

### Work

- Selected 25 main-study tasks deterministically (seed 42) from terminal-bench-2 at
  the pinned commit; pilot (8) and smoke (2) are subsets of main (25). Wrote
  `experiments/main_tasks.txt` (25 tasks).
- Recorded the exact serving identity of the RTX-6000-pro deployment used for
  generation+verification: `qwen3.8-27b-6000pro` (Qwen3.5), modelopt mixed
  FP8/NVFP4 (attention/linear_attn FP8, MLP + lm_head NVFP4 gs16, KV cache FP8),
  base snapshot `52d1adc5`, drafter `50307d4c`, SGLang `qwen38-27b`, context
  262144. Frozen in `frozen_main.yaml`.
- Frozen all plan-§26 decisions: task IDs, checkpoint/quantization, server+context,
  upstream commits, harness (qwen-coder 0.22.0), generation params (temp 0.7,
  max_tokens 8192, 5 candidates), rendering (`head_tail`, 120k), pairwise
  prompt/labels (G=20, A..T), criteria (specification/output/errors), granularity,
  repetitions (1), ranker (round-robin Bradley-Terry), and the ablation list
  (random / discrete / continuous / +optional frontier).

### Checks

- `frozen_main.yaml` loads: 25 tasks, 5 candidates, G=20 (A-T), 3 criteria, BT
  ranking; experiment id `5d5889e9…`.
- `git tag experiment-v1.0` created and pushed.

### Commit / tag

- Commit `730c282` (`frozen_main.yaml`); tag `experiment-v1.0` → `730c282`.

### Next action

Stage 17 — run the main study: 25 tasks x 5 candidates on the frozen `main-v1`
pool, ranked by all local selectors (no ground truth) with G=20 verification.
