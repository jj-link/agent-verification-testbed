# Qwen Actor Integration (Stage 5)

Stage 5 deliverable: connect the Qwen actor to a supported harness so it
produces one complete, officially graded local trajectory. Acceptance: one
complete local trajectory is graded.

## Result: PASS

The Qwen actor (`qwen-coder`, v0.21.15) completed `cancel-async-tasks` and the
official Terminal-Bench grader returned **reward 1.0** (no agent exception).

- Trial: `cancel-async-tasks__9MeAdy2`
- Agent: `qwen-coder` 0.21.15, model `qwen3.8-27b-sglang` (provider `openai`)
- Agent execution: ~9m44s
- Verifier reward: `1.0`

## Harness

Harbor (`harbor-framework/harbor`) agents, agent `qwen-coder`, which installs
Alibaba's Qwen Code CLI and runs it inside the task container pointed at the
local SGLang endpoint.

## Required: disable thinking

Qwen3.8 thinking mode is on by default. When Qwen Code talks to the Qwen3.8
server with thinking enabled, generation never terminates — the model streams
unbounded reasoning until Qwen Code's stream-lifetime cap (default 15 min)
errors out. Symptom: repeated `qwen-code.api_error` events in the session
JSONL, no assistant completion.

Fix: point Qwen Code at a system settings file that disables thinking and
bounds output tokens. See `configs/qwen-settings.json`:

- `model.generationConfig.extra_body.chat_template_kwargs.enable_thinking=false`
  (SGLang request shape that turns thinking off per request)
- `model.generationConfig.samplingParams.max_tokens=8192` (bounds output)

## Reproduce

Run from the project root (endpoint must be up on `spark1`):

```bash
uv run harbor run \
  -d terminal-bench@2.0 \
  -a qwen-coder \
  -m qwen3.8-27b-sglang \
  -i cancel-async-tasks \
  --mounts '[
    {"type":"bind","source":"C:/Users/josep/Projects/personal/agent-verification-testbed/configs/qwen-settings.json",
     "target":"/mnt/qwen-settings.json","read_only":true}
  ]' \
  --ae QWEN_CODE_SYSTEM_SETTINGS_PATH=/mnt/qwen-settings.json \
  --ae OPENAI_BASE_URL=http://100.86.3.45:8888/v1 \
  --ae OPENAI_API_KEY=sk-local \
  --yes
```

`OPENAI_API_KEY` value is a placeholder; the local endpoint performs no auth.
