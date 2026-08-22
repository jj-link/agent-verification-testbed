# Score-Logprob Validation (Stage 4)

Stage 4 deliverable: a diagnostic and token-label selection, confirming every
`G=5` label is scoreable. Acceptance: all `G=5` labels are scoreable.

## Result: PASS

`avt doctor` against `http://100.86.3.45:8888/v1` reports:

- endpoint availability: reachable
- endpoint model identity: `qwen3.8-27b-sglang`, `max_model_len 262144`
- logprob access: per-token logprobs returned
- score-token single-token property: labels `A B C D E` occur together as
  distinct tokens at content position 0

## Score-label selection (G=5)

Ordered single-token letter labels: `A` < `B` < `C` < `D` < `E`, mapped to
scores `1..5`. Letters tokenize to single tokens with a stable ordering, which
is the property the paper's probabilistic method requires.

- `G5_LABELS = ("A", "B", "C", "D", "E")` in `src/avt/doctor.py`.

## Diagnostic implementation

`avt doctor` (`src/avt/doctor.py`, stdlib only) checks, in order:

1. `/models` reachability;
2. served model identity + `max_model_len`;
3. that a chat completion returns per-token logprobs;
4. that a single `content` position's `top_logprobs` contains every configured
   label together (no cross-position union, so the check cannot false-pass).

Configured base URL: `LOCAL_QWEN_URL` env, default
`http://100.86.3.45:8888/v1`.

## Required serving mode

DSpark cannot return logprobs. The deployment therefore runs **MTP** (`start.sh`)
instead of DSpark. See `docs/qwen-deployment.md` → "Serving-mode note".

## Reproduce

```bash
uv run avt doctor
```

Requires the MTP server on `spark1` and network reachability from the
workstation.
