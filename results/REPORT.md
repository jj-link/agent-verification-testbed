# AVT Main Study — Stage 17/19 Results (`experiment-v1.1`)

**Experiment id:** `945140192175d4775e1997f80192a3e6`
**Config:** `experiments/frozen_main.yaml`, tag `experiment-v1.1`
**Model:** local `qwen3.8-27b-6000pro` (SGLang, single GPU), G=5 verifier
**Date:** 2026-08-23
**Pool:** 25 Terminal-Bench tasks × 5 candidates = 125 graded candidates

## 1. Pipeline completion

| Stage | Count | Status |
|---|---:|---|
| Candidates generated | 125 / 125 | all `SUCCEEDED` (4 concurrent workers, ~1h20m) |
| Pairs built | 250 (10 per task) | all `PAIRED` |
| Verifications | 750 (250 × 3 criteria) | all `SUCCEEDED`, 0 malformed/failed |
| Expected-score records | 375 (125 × 3 criteria) | complete |
| Evaluations | 125 | complete |
| Rankings | 75 (25 tasks × 3 selectors) | all `SUCCEEDED` |

## 2. Compute (local, single GPU)

- Generation: **68,923,223 input tokens**, **473,816 output tokens** across 125 candidates.
- Generation latency: median **88.2 s**, IQR [66.0, 176.0] per candidate.
- Verify: 750 single-token calls (G=5). Generator and verifier were sequential, one GPU-bound model server.

## 3. Local selectors on identical frozen pools

All three selectors rank the same candidate IDs. Top-pick agreement:
`discrete` == `continuous` 21/25; `continuous` == `random` 3/25.

Pool base pass rate: **26/125 = 20.8%** (task-bootstrap over the 25 per-task
pass fractions, 95% CI [9.6%, 34.4%]).

| Selector | Top-pass rate | 95% CI | Mean selected-top reward |
|---|---:|---|---:|
| random (realized seed-42 draw) | 5/25 = 20.0% | [4.0%, 36.0%] | 0.200 |
| random (uniform expectation) | 20.8% | [9.6%, 34.4%] | 0.208 |
| discrete (argmax) | 11/25 = 44.0% | [24.0%, 64.0%] | 0.440 |
| continuous (expectation) | 11/25 = 44.0% | [24.0%, 64.0%] | 0.440 |

The random arm's uncertainty is its **expected** random reward (`passing/5` per
task), which spans the whole pool; the 5/25 row is the single realized seeded
draw and is reported only as that realization.

### Selector differences (paired task-bootstrap, 20k resamples, mean selected-top reward)

| Comparison | Mean diff | 95% CI |
|---|---:|---:|
| continuous − expected-random | +0.232 | [0.104, 0.368] |
| discrete − expected-random | +0.232 | [0.104, 0.368] |
| continuous − discrete | 0.000 | [0.000, 0.000] |

The verifier-based selectors more than double the top-pass rate over a uniform
random baseline (44% vs 21%), and the difference is statistically significant
(the 95% CI excludes zero).

## 4. Verifier quality (per-pair, polarized pairs only)

On the 42 pairs where official rewards differ (one candidate passes, one fails),
using the stored per-pair responses (never the pool-wide aggregate):

| Verifier | Correct / polarized | Accuracy |
|---|---:|---:|
| continuous (expected score) | 42 / 42 | 1.000 |
| discrete (argmax) | 40 / 42 | 0.952 |

Caution: n=42 polarized pairs is small; the estimate has a wide interval and
the continuous perfect score likely reflects the strong score separation below.

## 5. Verifier score separation (fig 3)

Official PASS candidates have markedly higher verifier aggregate expected scores
(mean 4.62) than official FAIL candidates (mean 2.78) on the 1–5 scale,
consistent with the high pairwise discrimination.

## 6. Limitations

- 25 tasks is a modest sample; conclusions are proportional to it.
- Rewards are binary (26 pass / 99 fail); partially-solved or graded-0.5
  candidates are treated as failures, which is coarse.
- One model, one GPU, one seed; the continuous–random gap is significant on
  this pool but replication on more tasks/checkpoints would strengthen it.
- Stage 18 frontier-assisted method was not run (permitted blocker: paid API
  requires explicit authorization and a spending cap; no credentials configured).

## 7. Reproduction

```bash
# Frozen freeze
git checkout avt-v1.0.0  # release tag; experiment-v1.1 is the config freeze
# Regenerate figures/metrics (reads the frozen SQLite catalogs)
python analysis/generate_figures.py \
  --config experiments/frozen_main.yaml \
  --root /home/workbench/avt-data/main-v1
# produces results/figures/*.svg and results/stats.json
```

Raw catalogs (`experiment.sqlite`, `ground_truth.sqlite`) and artifacts live
under `/home/workbench/avt-data/main-v1/` on the NVIDIA-Workbench WSL engine.
