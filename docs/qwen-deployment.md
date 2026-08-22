# Local Qwen Deployment Record (Stage 3)

This is the recorded identity of the inference deployment used by AVT on
`spark1`. It fulfils the Stage 3 deliverable: exact model/server/hardware
record, with the endpoint and context verified.

## Hardware (spark1)

- Host: DGX Spark (`gx10-a6c7`), aarch64
- GPU: NVIDIA GB10 (unified memory; `free` shows ~121 GB)
- CPU: 20 logical cores
- Kernel: Ubuntu `6.17.0-1021-nvidia`
- NVIDIA driver: `580.159.03`

## Serving software

- Recipe: `MiaAI-Lab/Qwen3.8-27B-SGLang-DGX-Spark`
  - Pinned commit: `c90d8c34cf795185ee8de736b7ded9bca3fe0de1`
  - Launcher: `start-dspark.sh` → `start.sh`
- SGLang image: `lmsysorg/sglang:qwen38-27b`
  - Container digest: `sha256:0076dffa60b76b7bf033c04d05e0cc69d46f2b8cd60aa2468827782afe9bc38f`
- Namespace: container `qwen3.8-27b-sglang`, host port `8888`

## Model / endpoint identity

- Served model id: `qwen3.8-27b-sglang`
- Checkpoint: `RadixArk/Qwen3.8-27B-NVFP4`
- Checkpoint revision: `91cea059647696fd83964e43d57db122ff745993`
- Quantization: NVFP4
- Speculative decoding: DSpark (draft `RadixArk/Qwen3.8-27B-DSpark`)
- Served API: OpenAI-compatible `/v1`
  - Client endpoint for AVT on the workstation (private network):
    `http://100.86.3.45:8888/v1`
  - On-host health-check URL (loopback on spark1):
    `http://127.0.0.1:8888/v1`
- Context length: **262144** (native; YaRN off for DSpark)

## Configuration (`.env`)

- `QUANT=nvfp4`, `YARN=0`, `CONTEXT_LENGTH=262144`,
  `MAX_CONCURRENT_REQUESTS=4`
- Recipe Arg defaults applied (mem 0.90, bf16 GDN, DSpark block 7, etc.)

## Verification

- `/v1/models` → `qwen3.8-27b-sglang`, `max_model_len: 262144`
- Chat completion round-trip succeeded (thinking mode on; responded within
  `max_tokens`).
- Checkpoint downloads to `/root/.cache/huggingface` in the container on first
  boot; served revision matches `91cea059`.

## Prior server on spark1

Replaced (per plan, it is acceptable to take down other servers on `spark1`):

- Container: `inference-sglang-spark-redhatai_qwen36_35b_a3b_nvfp4_dflash`
- Image: `lmsysorg/sglang@sha256:4219417054...`
- Served: `Qwen3.6-35B-A3B-NVFP4-DFLASH`, port `8000`
- Rollback: full `docker inspect` snapshot saved at
  `/home/jjlink/avt-serve/incumbent-inspect.json` on `spark1`; restore by
  re-running the original `docker run` (image retained, container stopped, not
  removed).
