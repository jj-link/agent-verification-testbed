# Agent Verification Testbed (AVT)

Reproducible evaluation of methods for selecting the best agent trajectory
from multiple candidate runs, built around Terminal-Bench 2.0 and a local LLM
verifier.

## Provenance

AVT is an independent reproduction and extension of the pairwise probabilistic
verification method from **"LLM-as-a-Verifier: A General-Purpose Verification
Framework"** — Jacky Kwok, Shulu Li, Pranav Atreya, Yuejiang Liu, Yixing Jiang,
Chelsea Finn, Marco Pavone, Ion Stoica, Azalia Mirhoseini
([arXiv:2607.05391](https://arxiv.org/abs/2607.05391v2); a local copy is in
[`docs/`](docs/)).

From the paper, AVT reproduces: the continuous verifier (expected value over
the score-token logprob distribution), the discrete argmax LM-judge baseline,
Bradley–Terry pairwise ranking with round-robin aggregation, the
Specification/Output/Errors criteria decomposition, and the two-stage
frontier-assisted workaround (frontier reasoning feeding an open-model logit
scorer). A copy of the paper PDF is in [`docs/2607.05391v2.pdf`](docs/2607.05391v2.pdf).

Deliberate deviations from the paper's setup:

- **Self-verification study.** The paper verifies Gemini candidates with
  Gemini 2.5 Flash. AVT instead uses a local `qwen3.8-27b` as both candidate
  generator and verifier — testing whether a small open model can select its
  own successful trajectories.
- **Round-robin ranking only.** The paper's Probabilistic Pivot Tournament is
  not implemented (at N=5 candidates, full round-robin is affordable: 10 pairs).
- **Forced-token G=20.** The served model does not surface all 20 letter labels
  in top-logprobs, so the G=20 ablation uses forced single-target logit-bias
  scoring (plan §13.2 fallback #4) instead of the paper's top-logprob
  extraction. The main study is frozen at G=5.
- **Compressed verifier prompt.** The verifier outputs two bare score letters
  with no reasoning text ("No other text"), whereas the paper lets the model
  analyze both trajectories before emitting scores inside tags.
- **Not exercised.** Repeated-evaluation scaling (K>1), task-progress
  estimation (VOC), and dense-reward RL from the paper are out of scope.

Nothing here is a state-of-the-art claim: this is a small-scale (25-task)
replication with an open-weights verifier.

## Status

Development follows the staged plan in [`docs/avt-plan.md`](docs/avt-plan.md).
Current stage and progress are tracked in [`docs/progress.md`](docs/progress.md).

## Results (main study, `experiment-v1.1`)

25 Terminal-Bench tasks × 5 candidates, verified by a local `qwen3.8-27b-6000pro`
G=5 judge, ranked by three leakage-safe local selectors on identical frozen pools.
These selectors instantiate the discrete-vs-continuous comparison from the
paper (§4, Fig. 7) in the self-verification setting described above.

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

## G=20 granularity ablation (Stage 15)

The main study is frozen at G=5 (every label reliably present for the served
model). The plan-§13.2-compliant fallback #4 (forced single-target logit-bias
scoring) unlocks full G=20: `experiments/pilot-g20.yaml` verified at 240/240
with all 20 labels covered (no zero-fill), and the G5-vs-G20 ablation
(`experiments/pilot-g20-g5f-abl.yaml`, identical forced protocol on the same
frozen trajectories) shows G=20 uses 17 distinct discrete scores vs 5 at G=5.
Main study results above are unaffected (G=5).
