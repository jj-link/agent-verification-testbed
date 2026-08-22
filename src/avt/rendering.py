"""Verifier-safe, deterministic trajectory rendering (plan section 12).

Reads a candidate's Harbor ATIF trajectory (``agent/trajectory.json``) and
renders it as a plain-text action log for the pairwise verifier prompt. It
never reads ``verifier/`` artifacts (hidden tests, reward), reference solutions,
or any candidate pass/fail label, so the output cannot leak the grader outcome.

Truncation is deterministic and symmetric: the full public task instruction
(the user-source steps) is always preserved verbatim; oversized action bodies
are head+tail truncated through ``oversized_output_policy`` (``head_tail``).
Token counts use a deterministic ``chars/token`` estimate, applied identically
to every candidate so it is a fair relative budget proxy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

__all__ = ["RenderedTrajectory", "estimate_tokens", "render_trajectory", "load_atif"]

# Deterministic token-count proxy: characters-per-token. Documented approximation
# used only for context budgeting; applied identically to candidates A and B.
_CHARS_PER_TOKEN = 4

_TRUNCATION_MARKER = "[OUTPUT TRUNCATED: original_tokens={orig}, retained=head:{head}+tail:{tail}]"


def estimate_tokens(text: str) -> int:
    """Deterministic token-count estimate from character length."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))


@dataclass(frozen=True)
class RenderedTrajectory:
    candidate_id: str
    task_id: str
    instruction_text: str  # full public task instruction (never truncated)
    body_text: str  # chronological action log (head+tail truncated)
    original_tokens: int  # estimate over the full instruction + body
    rendered_tokens: int  # estimate over the rendered instruction + body
    truncated: bool


def load_atif(path: Path) -> list[dict[str, object]]:
    """Load an ATIF trajectory and return its steps (schema-agnostic)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data.get("steps", data) if isinstance(data, dict) else data
    if not isinstance(steps, list):
        raise ValueError(f"{path}: ATIF trajectory has no step list")
    return [s for s in steps if isinstance(s, dict)]


def _observation_content(step: dict[str, object]) -> str:
    obs = step.get("observation")
    results: list[object] = []
    if isinstance(obs, dict):
        raw = obs.get("results") or []
        if isinstance(raw, list):
            results = raw
    elif isinstance(obs, list):
        results = obs
    parts: list[str] = []
    for item in results:
        content = item.get("content") if isinstance(item, dict) else item
        if content:
            parts.append(str(content))
    return "\n".join(parts) if parts else ""


def _render_action(step: dict[str, object]) -> list[str]:
    """Render one non-instruction step into log lines, in chronological order:
    agent message, tool calls, then embedded tool outputs. qwen ATIF embeds
    ``observation.results`` on the ``agent`` step (there are no separate
    ``tool_result`` steps); user steps are handled separately as instructions.
    """
    source = step.get("source")
    if source == "user":
        return []
    lines: list[str] = []

    if source == "tool_result":
        content = _observation_content(step)
        if content:
            lines.append(f"[TOOL OUTPUT]\n{content}")
        return lines

    if source != "agent":
        return lines

    message = step.get("message") or ""
    if message and message != "(tool use)":
        lines.append(str(message))

    tool_calls = step.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = tc.get("function_name") or tc.get("name") or "?"
            args = tc.get("arguments") or {}
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False, sort_keys=True)
            lines.append(f"> {name}({args})")

    content = _observation_content(step)
    if content:
        lines.append(f"[TOOL OUTPUT]\n{content}")
    return lines


def _truncate_head_tail(
    items: list[str],
    body_budget_tokens: int,
) -> tuple[list[str], list[str], int, int]:
    """Split an ordered list of action lines into head+tail that fits the token
    budget, returning kept head, kept tail, head token estimate, tail token
    estimate. Keeps an even split of the available budget between head and tail.
    """
    if body_budget_tokens <= 0:
        return [], [], estimate_tokens("".join(items)), 0
    costs = [estimate_tokens(line) for line in items]
    total = sum(costs)
    if total <= body_budget_tokens:
        return items, [], total, 0

    head_budget = body_budget_tokens // 2
    tail_budget = body_budget_tokens - head_budget
    head: list[str] = []
    tail: list[str] = []
    head_tok = 0
    # greedy head from the front
    for line, cost in zip(items, costs, strict=False):
        if head_tok + cost <= head_budget:
            head.append(line)
            head_tok += cost
        else:
            break
    # greedy tail from the back
    tail_tok = 0
    for line, cost in reversed(list(zip(items, costs, strict=False))):
        if tail_tok + cost <= tail_budget:
            tail.append(line)
            tail_tok += cost
        else:
            break
    tail.reverse()
    # avoid overlap when head+tail would cover the whole list
    head_len = len(head)
    tail_len = len(tail)
    if head_len + tail_len >= len(items):
        overlap = head_len + tail_len - len(items)
        if overlap:
            head = head[:-overlap]
            head_tok = sum(estimate_tokens(ln) for ln in head)
    return head, tail, head_tok, tail_tok


def render_trajectory(
    candidate_id: str,
    task_id: str,
    atif_path: Path,
    body_budget_tokens: int,
    oversized_output_policy: str = "head_tail",
) -> RenderedTrajectory:
    """Render a candidate trajectory within ``body_budget_tokens``.

    The public instruction (user-source steps) is preserved verbatim; the action
    body is truncated head+tail to fit the budget when the policy is
    ``head_tail``.
    """
    steps = load_atif(atif_path)

    instructions: list[str] = []
    actions: list[str] = []
    for step in steps:
        if step.get("source") == "user":
            msg = step.get("message")
            if msg:
                instructions.append(str(msg))
            continue
        actions.extend(_render_action(step))

    instruction_text = "\n\n".join(instructions)
    body_parts = actions

    original_tokens = estimate_tokens(instruction_text + "\n\n" + "\n".join(body_parts))

    truncated = False
    if oversized_output_policy != "head_tail":
        raise ValueError(f"unsupported oversized_output_policy: {oversized_output_policy!r}")
    head, tail, head_tok, tail_tok = _truncate_head_tail(body_parts, body_budget_tokens)
    if head_tok + tail_tok < estimate_tokens("\n".join(body_parts)):
        marker = _TRUNCATION_MARKER.format(
            orig=estimate_tokens("\n".join(body_parts)), head=head_tok, tail=tail_tok
        )
        rendered_body = "\n".join(head) + "\n\n" + marker + "\n\n" + "\n".join(tail)
        truncated = True
    else:
        rendered_body = "\n".join(body_parts)

    body_text = rendered_body.strip("\n")
    rendered_tokens = estimate_tokens(instruction_text + "\n\n" + body_text)

    return RenderedTrajectory(
        candidate_id=candidate_id,
        task_id=task_id,
        instruction_text=instruction_text,
        body_text=body_text,
        original_tokens=original_tokens,
        rendered_tokens=rendered_tokens,
        truncated=truncated,
    )
