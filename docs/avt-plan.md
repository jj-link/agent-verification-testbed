# Implementation Handoff: Agent Verification Testbed

| Field | Decision |
|---|---|
| Project | **Agent Verification Testbed (AVT)** |
| Repository | `https://github.com/jj-link/agent-verification-testbed` |
| Benchmark | Terminal-Bench 2.0 |
| Local hardware | NVIDIA RTX 6000 Pro |
| Initial model | Existing local Qwen 27B endpoint, currently identified as `qwen2.8-27b` |
| Testing server | Run [`MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark`](https://github.com/MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark) on `spark1` for testing; it is acceptable to take down any other server running on `spark1` |
| Testing checkpoint | Ensure [`RadixArk/Qwen3.8-27B-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-27B-NVFP4) is available on `spark1`; download it if needed |
| Initial roles | The same Qwen model acts as both trajectory generator and verifier |
| Method | Pairwise probabilistic trajectory verification from *LLM-as-a-Verifier* |
| Ground truth | Official Terminal-Bench grader |
| Initial objective | Determine whether Qwen can select its own successful trajectories better than random or a discrete judge |
| Later extension | Compare the local verifier with a frontier-assisted verifier |

---

# 1. Assignment

Build, evaluate, document, and publicly release AVT.

AVT will run the local Qwen model multiple times on Terminal-Bench tasks, save the completed trajectories, and compare methods for selecting the best trajectory. The initial study uses the same Qwen checkpoint as both generator and verifier, but with separate prompts and fresh contexts.

Proceed through the entire project without waiting for approval between milestones. Ask the user only when blocked by GitHub authorization, unavailable infrastructure details, paid API authorization, destructive operations, or licensing concerns.

Do not stop after creating the repository or scaffolding.

---

# 2. Research question

> Can a local Qwen 27B agent improve its Terminal-Bench success rate by probabilistically ranking its own candidate trajectories, and does continuous score-token verification outperform a conventional discrete LLM judge?

The initial experiment compares selectors on identical frozen candidate pools:

| Selector | Description |
|---|---|
| First candidate | Single-attempt local-agent baseline |
| Random candidate | Chance selection from the local candidate pool |
| Qwen discrete judge | Select using the highest-probability score token |
| Qwen continuous verifier | Select using expected value over the score-token distribution |
| Oracle | Select using official grader results; analysis-only upper bound |

After the entirely local pipeline works, add:

| Selector | Description |
|---|---|
| Frontier-assisted verifier | Frontier comparative reasoning followed by Qwen logit scoring, as in Appendix B.6 |
| Cross-local verifier | Optional second open-weight verifier, if hardware permits |

---

# 3. Scope

## Included

- Official Terminal-Bench 2.0 tasks, environments, trajectories, and graders
- Existing Terminal-Bench/Harbor-compatible agent harness
- Local Qwen trajectory generation
- Multiple independent candidate trajectories per task
- Pairwise trajectory comparison
- Discrete score-token judging
- Continuous score-token expectation
- Specification, Output, and Errors criteria
- Round-robin candidate ranking
- Objective evaluation with official grader results
- Token, latency, GPU-time, and API-cost accounting
- Reproducible experiments and statistical analysis
- Public GitHub development with regular commits

## Excluded from version 1

- A new benchmark or custom grader
- Verifier-guided trajectory repair
- Intervention during active trajectories
- Reinforcement learning or fine-tuning
- Browser or GUI agents
- A general-purpose multi-agent framework
- Distributed execution infrastructure
- Probabilistic Pivot Tournament for small candidate pools

---

# 4. Fixed experimental decisions

| Decision | Smoke test | Pilot | Main study |
|---|---:|---:|---:|
| Tasks | 2 | 8 | 25 |
| Candidates per task | 3 | 3, then 5 if stable | 5 |
| Generator | Local Qwen 27B | Local Qwen 27B | Local Qwen 27B |
| Local verifier | Same Qwen checkpoint | Same Qwen checkpoint | Same Qwen checkpoint |
| Criteria | Specification, Output, Errors | Same | Same |
| Score granularity | `G=5` | Compare `G=5` and `G=20` | `G=20` |
| Repetitions | `K=1` | `K=1`; test `K=4` offline | `K=1` primary |
| Ranking | Full round robin | Full round robin | Full round robin |
| Ground truth | Official grader | Official grader | Official grader |

The exact model ID, revision, quantization, serving software, and context length must be queried from the running endpoint or local configuration and recorded. Do not leave `qwen2.8-27b` as the sole identifier if a more exact checkpoint ID is available.

---

# 5. System boundary

AVT is an experiment controller around Terminal-Bench; it is not a replacement for Terminal-Bench or its agent harness.

| Terminal-Bench / Harbor owns | AVT owns |
|---|---|
| Tasks and task containers | Experiment configuration |
| Agent-environment execution | Repeated candidate generation |
| Supported agent harnesses | Local endpoint integration |
| Official grading | Trajectory indexing and storage |
| Task lifecycle | Pair construction |
| Benchmark result format | Verifier prompts and scoring |
|  | Candidate ranking |
|  | Ground-truth isolation |
|  | Metrics, cost analysis, and reports |

---

# 6. System architecture

```text
                                 ZDDDD                          3 Evaluation and Analysis            3
                          3 success z verifier quality z cost  3
                          @DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDY
```

---

# 7. Runtime topology

Run the MVP on one workstation.

```text
RTX 6000 Pro
    @DD One existing Qwen model server
            CDD Generation phase: Terminal-Bench actor calls
            @DD Verification phase: fresh pairwise verifier calls

CPU / RAM
    CDD AVT controller
    CDD Harbor and task containers
    CDD SQLite metadata databases
    @DD Filesystem artifact store
```

## Runtime decisions

| Concern | Decision |
|---|---|
| Generator and verifier deployment | Share one Qwen server |
| Generation and verification scheduling | Sequential, not concurrent |
| Verifier context | Fresh context for each pair and criterion |
| Actor state reuse | Forbidden |
| Distributed workers | Not implemented |
| Local concurrency | One GPU-bound model workload initially |
| Container concurrency | Conservative; configurable after smoke testing |
| Frontier requests | Deferred until local study works; bounded concurrency and spending cap |

---

# 8. Storage design

Use two SQLite databases and a filesystem artifact store.

```text
.avt/
CDD experiment.sqlite
CDD ground_truth.sqlite
CDD artifacts/
3   CDD candidates/{candidate_id}/
3   3   CDD manifest.json
3   3   CDD trajectory.json
3   3   @DD usage.json
3   CDD verifier-runs/{verification_id}/
3   3   CDD request.json
3   3   CDD response.json
3   3   @DD scores.json
3   @DD official-results/{candidate_id}/result.json
@DD logs/
```

| Data | Location | Verifier access |
|---|---|---:|
| Task ID and public instruction | `experiment.sqlite` | Yes |
| Candidate manifest | `experiment.sqlite` | Yes through safe view |
| Full trajectory | Artifact store | Yes |
| Generator usage | `experiment.sqlite` | No need |
| Pair membership and display order | `experiment.sqlite` | Yes |
| Verifier prompt, output, and logprobs | Artifact store + `experiment.sqlite` | N/A |
| Official grader result | `ground_truth.sqlite` | **No** |
| Hidden tests and reference solution | Upstream benchmark only | **No** |
| Ranking result | `experiment.sqlite` | N/A |
| Evaluation join | Analysis stage only | **No** |

The verification process must not open `ground_truth.sqlite`.

---

# 9. Deterministic identifiers

Generate stable IDs from canonical serialized inputs:

```text
experiment_id   = hash(frozen experiment configuration)
candidate_id    = hash(experiment_id, task_id, attempt_index)
pair_id         = hash(experiment_id, task_id, sorted candidate IDs)
verification_id = hash(pair_id, verifier config, criterion, repetition, display order)
ranking_id      = hash(task_id, candidate-pool hash, selector config)
```

Stable IDs make every stage idempotent and resumable.

---

# 10. Core interfaces

```python
class TerminalBenchAdapter(Protocol):
    def run_candidate(
        self,
        task_id: str,
        attempt_index: int,
        generator_config: GeneratorConfig,
    ) -> CandidateRecord: ...

class CandidateRepository(Protocol):
    def get_safe(self, candidate_id: str) -> VerifierCandidateView: ...
    def list_pool(self, experiment_id: str, task_id: str) -> list[CandidateRecord]: ...

class PairBuilder(Protocol):
    def build(self, task_id: str, candidate_ids: list[str]) -> list[CandidatePair]: ...

class PairVerifier(Protocol):
    def score(
        self,
        pair: CandidatePair,
        criterion: Criterion,
        repetition: int,
    ) -> PairScore: ...

class Ranker(Protocol):
    def rank(
        self,
        candidate_ids: list[str],
        pair_scores: list[PairScore],
    ) -> RankingResult: ...

class GroundTruthRepository(Protocol):
    def get(self, candidate_id: str) -> OfficialResult: ...

class Evaluator(Protocol):
    def evaluate(
        self,
        rankings: list[RankingResult],
        ground_truth: GroundTruthRepository,
    ) -> EvaluationSummary: ...
```

Verifier code must accept only `VerifierCandidateView`, never arbitrary database records.

---

# 11. Candidate generation

For each selected task:

1. Launch the official task through the pinned Terminal-Bench/Harbor integration.
2. Run the local Qwen actor with a fixed harness and generation configuration.
3. Save the complete trajectory and usage information.
4. Run the official grader.
5. Save the grader result only in the ground-truth store.
6. Repeat for the configured candidate count.

## Candidate rules

| Situation | Treatment |
|---|---|
| Agent completes task but grader gives zero | Valid failed candidate |
| Agent reaches time/turn limit normally | Valid candidate if official grader can evaluate it |
| Container fails to launch | Infrastructure failure; retry |
| Model endpoint times out before usable run | Infrastructure failure; bounded retry |
| Candidate lacks official grader result | Unusable until grading succeeds |
| Candidate performs poorly | Never discard for quality reasons |
| Candidate pool is frozen | Never regenerate under the same experiment ID |

Failed candidates are essential for verifier evaluation.

---

# 12. Verifier-safe trajectory rendering

Terminal trajectories may exceed model context. Implement a deterministic renderer.

## Renderer priorities

1. Preserve the complete public task instruction.
2. Preserve chronological action order.
3. Preserve commands, return codes, and concise outputs.
4. Preserve the final portion of the trajectory.
5. Truncate individual oversized outputs using explicit markers.
6. Apply the same rendering policy to candidates A and B.
7. Record original and rendered token counts.
8. Reserve enough context for both candidates, instructions, and scores.
9. Do not use an LLM-generated summary in the primary experiment.

Example truncation marker:

```text
[OUTPUT TRUNCATED: original_tokens=8421, retained=head:1000+tail:1000]
```

---

# 13. Verification methods

## 13.1 Pairwise prompt

Each verifier request receives:

- domain-reviewer instruction
- one evaluation criterion
- public Terminal-Bench task description
- rendered trajectory A
- rendered trajectory B
- required score tags

Use fresh context for every request.

Do not include:

- grader outcome
- hidden tests
- reference solution
- actor KV cache or conversation state
- labels suggesting which candidate succeeded

## 13.2 Score labels

Start with `G=5` for the smoke test. Increase to `G=20` after validating logprob support.

Requirements:

- every label must tokenize to exactly one token;
- every score token must have a stable numeric ordering;
- the server must return a logit or logprob for every configured label;
- missing labels must not be silently assigned probability zero.

Fallback order if the current endpoint does not expose every label:

1. increase `top_logprobs`;
2. reduce to `G=5`;
3. add direct local scoring using the same checkpoint;
4. use forced-token scoring.

## 13.3 Discrete judge

For each trajectory, use the numeric value of its highest-probability score token.

## 13.4 Continuous verifier

For trajectory \(\tau\), criterion \(c\), and repetition \(k\):

\[
R_{c,k}(\tau)=\sum_{g=1}^{G}p(\psi_g\mid x,c,\tau)\phi(\psi_g)
\]

Aggregate:

\[
R(\tau)=\frac{1}{CK}\sum_{c=1}^{C}\sum_{k=1}^{K}R_{c,k}(\tau)
\]

Store:

```text
raw_expected_score ? [1, G]
normalized_score   = (raw_expected_score - 1) / (G - 1)
```

## 13.5 Criteria

| Criterion | Definition |
|---|---|
| Specification | Satisfies all explicit task requirements |
| Output | Produces the expected output, artifacts, or format |
| Errors | Avoids meaningful failure signals in logs and tool output |

## 13.6 Repetition

Primary study: `K=1`.

Ablation: run `K=4` on the already frozen pairs if affordable. Do not regenerate candidate trajectories.

---

# 14. Pair construction and ranking

For `N=5`, build all ten unordered pairs.

Randomize displayed A/B order deterministically using the experiment seed and save the mapping.

For a pair \((i,j)\):

\[
p_{ij}=\frac{1}{1+\exp(-(R_i-R_j))}
\]

Candidate utility:

\[
u_i=\frac{1}{N-1}\sum_{j\ne i}p_{ij}
\]

Select:

\[
i^*=\arg\max_i u_i
\]

## Tie-breaking

1. Higher mean expected score across comparisons
2. Fewer malformed verifier records
3. Lexicographically lower deterministic candidate ID

Never use the official grader for tie-breaking.

## Pair coverage

Primary ranking requires 100% usable pair coverage. If a pair remains unusable after retries, fail that ranking job visibly rather than silently changing the method.

---

# 15. Job state model

Experiment stage:

```text
CREATED
   GENERATING
   GENERATED
   PAIRING
   PAIRED
   VERIFYING
   VERIFIED
   RANKING
   RANKED
   EVALUATING
   COMPLETE
```

Individual job state:

```text
PENDING  RUNNING  SUCCEEDED
                  ? RETRYABLE_FAILED  RUNNING
                  ? PERMANENT_FAILED
```

## Retry policy

| Failure | Policy |
|---|---|
| Container startup failure | Retry twice with backoff |
| Local endpoint transient error | Retry twice |
| Candidate model timeout | Retry once if no usable trajectory was created |
| Malformed verifier output | Retry once with strict format reminder |
| Missing score-token probabilities | Configuration failure; do not run experiment |
| Corrupt artifact | Mark failed; do not silently regenerate after freeze |
| Analysis error | Rebuild from immutable stored records |
| Frontier rate limit | Exponential backoff with bounded retries |

Every CLI stage must skip successful records by default and resume incomplete work.

---

# 16. CLI

Provide these commands or close equivalents:

```bash
avt doctor
avt select-tasks --seed 42 --pilot-count 8 --main-count 25
avt generate --experiment experiments/smoke.yaml
avt index-results --experiment experiments/smoke.yaml
avt build-pairs --experiment experiments/smoke.yaml
avt verify --experiment experiments/smoke.yaml --selector qwen-discrete
avt verify --experiment experiments/smoke.yaml --selector qwen-continuous
avt rank --experiment experiments/smoke.yaml --selector qwen-continuous
avt evaluate --experiment experiments/smoke.yaml
avt analyze --experiment experiments/smoke.yaml
```

`avt doctor` must validate:

- Git and repository state
- Docker
- Terminal-Bench/Harbor pins
- local Qwen endpoint availability
- exact endpoint model identity
- tokenizer identity
- score-token single-token property
- complete score-token logprob access
- configured context window
- database and artifact paths
- optional frontier credentials

---

# 17. Configuration

Use a frozen YAML configuration.

```yaml
experiment:
  name: smoke-v1
  seed: 42
  task_file: experiments/smoke_tasks.txt
  candidates_per_task: 3

upstream:
  terminal_bench_commit: REPLACE_AFTER_INSPECTION
  harbor_commit: REPLACE_AFTER_INSPECTION

generator:
  harness: REPLACE_AFTER_INSPECTION
  model: REPLACE_WITH_EXACT_QWEN_MODEL_ID
  endpoint: ${LOCAL_QWEN_URL}
  temperature: REPLACE_AFTER_BASELINE_INSPECTION
  max_tokens: REPLACE_AFTER_BASELINE_INSPECTION

verifier:
  model: REPLACE_WITH_EXACT_QWEN_MODEL_ID
  endpoint: ${LOCAL_QWEN_URL}
  criteria:
    - specification
    - output
    - errors
  granularity: 5
  repetitions: 1

rendering:
  max_pair_context_tokens: REPLACE_AFTER_ENDPOINT_INSPECTION
  oversized_output_policy: head_tail

ranking:
  method: round_robin_bradley_terry
  minimum_pair_coverage: 1.0

storage:
  root: .avt
  metadata_db: .avt/experiment.sqlite
  ground_truth_db: .avt/ground_truth.sqlite
```

Do not guess values marked `REPLACE_AFTER_INSPECTION`. Determine them from the installed upstream software and running endpoint, then commit the resolved configuration.

Secrets belong in environment variables and `.env` only.

---

# 18. Repository structure

```text
agent-verification-testbed/
CDD .github/workflows/ci.yml
CDD README.md
CDD HANDOFF.md
CDD LICENSE
CDD pyproject.toml
CDD uv.lock
CDD .env.example
CDD .gitignore
CDD src/avt/
3   CDD cli.py
3   CDD domain/
3   CDD tbench/
3   CDD storage/
3   CDD generation/
3   CDD rendering/
3   CDD verification/
3   CDD ranking/
3   CDD evaluation/
3   CDD orchestration/
3   @DD analysis/
CDD configs/
CDD experiments/
3   CDD smoke_tasks.txt
3   CDD pilot_tasks.txt
3   CDD main_tasks.txt
3   CDD smoke.yaml
3   CDD pilot.yaml
3   @DD frozen_main.yaml
CDD tests/
CDD docs/
3   CDD terminal-bench-integration.md
3   CDD system-design.md
3   CDD methodology.md
3   @DD progress.md
CDD analysis/
CDD results/
@DD report/
```

---

# 19. Implementation plan

| Stage | Work | Deliverable | Acceptance criterion |
|---:|---|---|---|
| 1 | Create/inspect public GitHub repository | Package, CI, README, handoff | Fresh clone installs; CI passes |
| 2 | Inspect and pin upstream systems | Terminal-Bench/Harbor integration note | One official baseline task runs and grades |
| 3 | Inspect local Qwen deployment | Exact model/server/hardware record | Endpoint and context verified |
| 4 | Validate score-logprob capability | Diagnostic and token-label selection | All `G=5` labels are scoreable |
| 5 | Connect Qwen actor to supported harness | Local actor configuration | One complete local trajectory is graded |
| 6 | Implement storage and deterministic IDs | Two DBs and artifact store | Records survive restart and resume |
| 7 | Implement candidate generation | Resumable generation jobs | Three candidates for each smoke task |
| 8 | Implement safe renderer and pair builder | Frozen pair records | Leakage tests pass |
| 9 | Implement discrete Qwen judge | Pair scores | Every smoke pair receives valid scores |
| 10 | Implement continuous Qwen verifier | Expected-score records | Math and logprob tests pass |
| 11 | Implement three-criterion evaluation | Criterion and aggregate scores | Aggregation is reproducible |
| 12 | Implement round-robin ranker | Ranking records | Candidate selected without ground truth |
| 13 | Run smoke test | Two tasks, three candidates | Pipeline completes end to end |
| 14 | Run pilot | Eight tasks, initially three candidates | Reliability and cost measured |
| 15 | Scale pilot to five candidates and `G=20` | Expanded frozen pools | Score coverage and context remain reliable |
| 16 | Freeze main experiment | `frozen_main.yaml` and tag | `experiment-v1.0` pushed |
| 17 | Run main study | 25 tasks x five candidates | Identical pools ranked by all local selectors |
| 18 | Add frontier-assisted method | Optional two-stage verifier | Runs on frozen pools only |
| 19 | Analyze and publish | Figures, report, README, release | Public tagged release available |

Proceed through all stages without pausing for approval except for listed blockers.

---

# 20. Tests

Unit tests must cover:

- deterministic ID generation
- score-label tokenization
- complete score-token logprob availability
- logprob normalization
- expected-score calculation
- criterion averaging
- repetition averaging
- Bradley-Terry preference calculation
- round-robin utility calculation
- tie-breaking
- A/B display-order restoration
- deterministic random selector
- malformed verifier output
- database isolation
- verifier payload leakage prevention
- idempotent job execution and resume behavior

Integration tests must cover:

- fixture candidate indexing
- pair construction
- mocked discrete and continuous verifier calls
- ranking without access to ground truth
- evaluation join after ranking
- one optional live local-endpoint smoke test excluded from ordinary CI

Paid APIs must never be required for CI.

---

# 21. Metrics

## Generator

\[
\operatorname{Pass@1}
=
\frac{\text{successful first candidates}}{\text{tasks}}
\]

\[
\operatorname{OraclePass@N}
=
\frac{\text{tasks with at least one successful candidate}}{\text{tasks}}
\]

## Selection

\[
\operatorname{SelectedPass@1}
=
\frac{\text{tasks where selected candidate succeeds}}{\text{tasks}}
\]

\[
\operatorname{GapRecovered}
=
\frac{
\operatorname{SelectedPass@1}
-
\operatorname{RandomPass@1}
}{
\operatorname{OraclePass@N}
-
\operatorname{RandomPass@1}
}
\]

Report gap recovery as undefined when the denominator is zero.

## Verifier

For pairs containing one passing and one failing candidate:

\[
\operatorname{PairwiseAccuracy}
=
P(R_{\mathrm{pass}}>R_{\mathrm{fail}})
\]

Also report:

- tie rate
- malformed-output rate
- A/B position sensitivity
- successful-candidate rank
- mean reciprocal rank where meaningful
- discrete-versus-continuous disagreement

## Efficiency

Report separately:

- local generation tokens
- local verification tokens
- generation and verification wall time
- measured GPU-hours where available
- candidates generated per task
- comparisons per task
- optional frontier tokens and API cost
- compute or cost per selected successful task

Do not describe local inference as free.

---

# 22. Statistical analysis

Use paired task-level analysis because all selectors operate on the same frozen candidate pools.

Report:

- task-bootstrap 95% confidence intervals for selected pass rates
- paired task-bootstrap intervals for selector differences
- pairwise verifier accuracy intervals
- tie and malformed-output rates
- median and interquartile range for tokens and latency
- McNemar's test only where sample size is suitable

Bootstrap tasks, not individual candidate pairs, for end-to-end selector comparisons.

Keep claims proportional to the 25-task sample size.

---

# 23. Targeted ablations

Run ablations on saved trajectories only.

| Ablation | Values |
|---|---|
| Scoring | Discrete argmax vs continuous expectation |
| Granularity | `G=5` vs `G=20` |
| Criteria | Holistic vs Specification + Output + Errors |
| Repetition | `K=1` vs `K=4`, if affordable |
| Candidate count | `N=3` vs `N=5` |
| Verifier | Same Qwen vs frontier-assisted; cross-local if later available |

Do not run every Cartesian combination. Use one primary configuration and focused comparisons.

---

# 24. Frontier-assisted extension

Implement only after the local pipeline and main local experiment are functional.

For each frozen candidate pair:

1. Ask the authorized frontier model for comparative reasoning and coarse 1-10 scores.
2. Save its raw response, token use, latency, and API cost.
3. Give the frontier analysis and original pair to the local Qwen scorer.
4. Extract Qwen's score-token distribution.
5. Compute continuous expected scores.
6. Rank candidates with the same round-robin engine.
7. Compare against the same local candidate pools.

Require an explicit API spending cap before making paid calls.

---

# 25. GitHub and commit policy

Create or use:

`https://github.com/jj-link/agent-verification-testbed`

If authorization under `jj-link` is unavailable, stop and request authentication. Do not create the repository elsewhere.

Develop publicly from the beginning. Make and push meaningful commits after coherent, tested changes.

Expected history includes:

```text
chore: initialize public AVT repository
ci: add lint and test workflow
feat(tbench): add pinned Terminal-Bench integration
feat(models): inspect and validate local Qwen endpoint
feat(storage): add isolated metadata and ground-truth stores
feat(generation): collect and index local trajectories
feat(rendering): add deterministic trajectory renderer
feat(verification): add leakage-safe pair builder
feat(verification): add discrete Qwen judge
feat(verification): add continuous score-token verifier
feat(verification): add criteria decomposition
feat(ranking): add round-robin Bradley-Terry ranking
experiment: freeze main local verification study
analysis: add success and compute metrics
feat(frontier): add frontier-assisted verification
docs: publish results and reproduction guide
```

Before every commit:

1. run relevant tests, linting, formatting, and type checks;
2. inspect `git status`;
3. inspect the staged diff;
4. confirm no secrets, credentials, weights, or large temporary artifacts are staged;
5. commit one coherent change;
6. push regularly.

Never commit `.env`, API keys, model weights, private data, or unauthorized benchmark artifacts.

Update `docs/progress.md` throughout development with milestone, decisions, tests, commit SHA, blockers, and next action.

---

# 26. Experiment freeze

After the pilot:

1. Freeze task IDs.
2. Freeze exact Qwen checkpoint and quantization.
3. Freeze server and context settings.
4. Freeze Terminal-Bench and Harbor commits.
5. Freeze harness configuration.
6. Freeze generation parameters and candidate count.
7. Freeze trajectory-rendering policy.
8. Freeze pairwise prompt and score labels.
9. Freeze criteria, granularity, repetitions, and ranker.
10. Freeze the ablation list.
11. Commit `experiments/frozen_main.yaml`.
12. Tag the commit `experiment-v1.0`.
13. Push the tag before the main run.

Do not tune on the main-study results.

If an infrastructure bug requires a change, document it, create a new tag, and rerun every affected condition consistently.

---

# 27. Publication

Publish:

- source code
- exact upstream commit SHAs
- exact Qwen checkpoint and serving configuration
- hardware description
- selected task IDs and selection seed
- frozen experiment configuration
- compact candidate and pair indexes
- ranking outputs
- official-result joins
- analysis scripts
- tables and figures
- technical report
- README reproduction commands

Large trajectories may be stored as compressed GitHub release assets or in an appropriate external dataset repository. Publish hashes and retrieval instructions.

The report must distinguish:

- direct reproduction of the paper's scoring method
- the same-model local generator/verifier experiment
- deviations caused by model-server or API constraints
- frontier-assisted extension
- deferred paper components

Do not claim state of the art.

---

# 28. Definition of done

| Requirement | Completion condition |
|---|---|
| Public repository | Exists under `jj-link` with incremental history |
| CI | Passes on `main` |
| Upstream integration | Exact Terminal-Bench/Harbor versions pinned |
| Qwen deployment | Exact checkpoint, quantization, server, and context recorded |
| Local actor | Produces complete, officially graded trajectories |
| Candidate pools | Five usable candidates for every main-study task |
| Isolation | Verification process cannot access ground truth |
| Discrete judge | Implemented and evaluated |
| Continuous verifier | Full expected score-token calculation implemented |
| Criteria | Specification, Output, and Errors implemented |
| Ranking | Round-robin Bradley-Terry selection implemented |
| Main study | Frozen 25-task local study completed |
| Analysis | Success, verifier accuracy, compute, cost, and uncertainty reported |
| Frontier extension | Completed if credentials and budget are authorized |
| Reproduction | Documented commands work from a fresh clone |
| Publication | Report, results, tag, and GitHub release are public |

---

# 29. Permitted blockers

Ask the user only if:

1. GitHub authentication cannot access `jj-link`.
2. The current local endpoint does not reveal the actual model/checkpoint and configuration cannot be found.
3. The model server cannot support required scoring and choosing a fallback requires installing or loading a different runtime.
4. Frontier API credentials or spending approval are required.
5. An operation could overwrite or destroy existing work.
6. Licensing prevents publication of an intended artifact.

Otherwise, choose sensible implementation details, document them, commit them, and continue.

---

# 30. Final implementation report

When complete, report:

| Item | Required information |
|---|---|
| Repository | Public URL |
| Release | Experiment tag and release URL |
| Upstream | Terminal-Bench and Harbor commit SHAs |
| Local model | Exact Qwen ID, revision, quantization, and serving stack |
| Hardware | RTX 6000 Pro configuration plus relevant CPU/RAM |
| Scale | Task, candidate, pair, and verifier-call counts |
| Generator | Pass@1 and oracle pass@5 |
| Selection | Random, discrete, continuous, and optional frontier-assisted pass rates |
| Verifier | Pairwise accuracy, ties, malformed outputs, and position sensitivity |
| Efficiency | Tokens, latency, GPU-hours, optional API cost, and cost per success |
| Reproduction | Exact commands |
| Limitations | Main validity and scope constraints |

Begin by checking GitHub authorization, inspecting or creating the public repository, pinning the actual Terminal-Bench integration, and validating the existing Qwen endpoint. Continue through the complete project rather than stopping after setup.DDDDDDDDDDDDDDDDDDDDDDDDD?
                                 3        AVT Controller       3
                                 3 CLI z jobs z resume z config3
                                 @DDDDDDDDDDDDDDBDDDDDDDDDDDDDDY
                                                3
                                       generate N attempts
                                                3
                                 ZDDDDDDDDDDDDDD                                 3 Terminal-Bench / Harbor     3
                                 3 tasks z containers z harness3
                                 @DDDDDDDDDBDDDDDDDDDDDBDDDDDDDY
                                           3           3
                                  trajectories       grader results
                                           3           3
                          ZDDDDDDDDDDDDDDDD                          3 Candidate Store   3   3 Ground-Truth Store  3
                          3 verifier-safe     3   3 evaluation-only     3
                          @DDDDDDDDDBDDDDDDDDDY   @DDDDDDDDDBDDDDDDDDDDDY
                                    3                       3
                               construct pairs              3
                                    3                       3
                          ZDDDDDDDDD                          3 Verification Engine      3       3
                          3 discrete z continuous    3       3
                          3 frontier-assisted later  3       3
                          @DDDDDDDDDBDDDDDDDDDDDDDDDDY       3
                                    3 pair scores            3
                          ZDDDDDDDDD                          3 Ranking Engine     3              3
                          3 round robin + BT   3              3
                          @DDDDDDDDDBDDDDDDDDDDY              3
                                    3 selected candidate      3
                          ZDDDDDDDDD
