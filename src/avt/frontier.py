"""Frontier-assisted two-stage verifier (plan section 24)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen

__all__ = [
    "FrontierAnalysis",
    "FrontierClient",
    "FrontierError",
    "FrontierNotConfigured",
    "SpendCapExceeded",
    "SpendLedger",
    "build_frontier_messages",
]

# Default pricing (USD per 1k tokens) for known frontier models when the config
# does not supply explicit prices. A paid call is refused if the model is not in
# this table and no explicit price is configured.
_DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5": (5.00, 10.00),
    "gpt-5-mini": (0.25, 1.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.150, 0.600),
    "claude-3-7-sonnet-20250219": (3.00, 15.00),
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "deepseek-reasoner": (0.55, 2.19),
}


class FrontierError(Exception):
    """A frontier-assisted verification step failed or was refused."""


class FrontierNotConfigured(FrontierError):
    """No authorized frontier endpoint / spend cap is configured."""


class SpendCapExceeded(FrontierError):
    """A paid frontier call would exceed the configured spend cap."""


@dataclass(frozen=True)
class FrontierAnalysis:
    """A frontier model's comparative reasoning + coarse 1-10 scores for a pair.

    Also carries the raw response plus token usage, latency, and cost required
    by plan 24 step 2 (save raw response, token use, latency, API cost).
    """

    text: str
    raw: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cost_usd: float


class SpendLedger:
    """Tracks cumulative API spend and enforces an explicit cap."""

    def __init__(self, cap_usd: float, initial_spent: float = 0.0) -> None:
        if cap_usd <= 0.0:
            raise ValueError("spend cap must be positive")
        self._cap = cap_usd
        self._spent = initial_spent

    @property
    def cap(self) -> float:
        return self._cap

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def remaining(self) -> float:
        return self._cap - self._spent

    def reserve(self, cost_usd: float) -> None:
        """Check and reserve ``cost_usd`` against the cap before a call."""
        if cost_usd < 0.0:
            raise ValueError("negative cost")
        if self._spent + cost_usd > self._cap:
            raise SpendCapExceeded(
                f"frontier spend cap ${self._cap:.4f} would be exceeded by "
                f"${cost_usd:.4f} (already spent ${self._spent:.4f})"
            )
        self._spent += cost_usd


def _pricing_for(model: str, cfg: dict[str, object]) -> tuple[float, float] | None:
    """Return (usd_per_1k_in, usd_per_1k_out) for ``model`` from config/table."""
    explicit = cfg.get("pricing")
    if isinstance(explicit, dict):
        per_k = (
            explicit.get("input_per_1k"),
            explicit.get("output_per_1k"),
        )
        if all(isinstance(x, (int, float)) for x in per_k):
            return float(per_k[0]), float(per_k[1])  # type: ignore[arg-type]
    return _DEFAULT_PRICING.get(model)


def _post_and_read(
    url: str, payload: dict[str, object], api_key: str, timeout: int = 180
) -> dict[str, Any]:
    req = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:  # noqa: S310 (explicit frontier URL)
        return json.loads(resp.read())  # type: ignore[no-any-return]


def build_frontier_messages(
    task_description: str,
    trajectory_a: str,
    trajectory_b: str,
    criterion: str,
    max_tokens_hint: int = 600,
) -> list[dict[str, str]]:
    """Comparative-reasoning prompt for the frontier model (plan 24 step 1).

    Asks for coarse 1-10 scores plus brief reasoning for each trajectory. Only
    public task context and the two trajectories are included (no grader
    outcome, hidden tests, or pass/fail labels), preserving the §13.1 leakage
    rules.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are an expert reviewer evaluating two AI coding-agent "
                "trajectories for the same software engineering task. Give each "
                "a coarse integer score from 1 (poor) to 10 (excellent) for the "
                f"stated criterion, followed by 2-3 sentences of reasoning. "
                f"Respond in at most about {max_tokens_hint} tokens."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{criterion}\n\n"
                f"TASK:\n{task_description}\n\n"
                f"TRAJECTORY A:\n{trajectory_a}\n\n"
                f"TRAJECTORY B:\n{trajectory_b}\n\n"
                "Output ONLY:\n"
                "TRAJECTORY A SCORE: <1-10>\n"
                "TRAJECTORY B SCORE: <1-10>\n"
                "REASONING: <2-3 sentences>"
            ),
        },
    ]


def _usage_tokens(raw: dict[str, Any]) -> tuple[int, int]:
    usage = raw.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    return (
        int(prompt_tokens) if isinstance(prompt_tokens, int) else 0,
        int(completion_tokens) if isinstance(completion_tokens, int) else 0,
    )


class FrontierClient:
    """Frontier endpoint backed by a hard spend cap and authorization gate.

    ``authorized()`` is False unless an API key for ``api_key_env`` is present
    in the environment AND a positive spend cap is configured. No network call
    is made unless a caller first checks ``authorized()`` and ``reserves``
    against the ledger. The ``post`` callable is injectable for tests.
    """

    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        api_key_env: str,
        spend_cap_usd: float,
        pricing: dict[str, object] | None = None,
        post: Any = _post_and_read,
        env: dict[str, str] | None = None,
    ) -> None:
        self._model = model
        self._endpoint = endpoint.rstrip("/")
        self._api_key_env = api_key_env
        self._cfg: dict[str, object] = dict(pricing or {})
        self._pricing = _pricing_for(model, self._cfg)
        self._post = post
        self._env = os.environ if env is None else env
        self._ledger = SpendLedger(spend_cap_usd)

    def authorized(self) -> bool:
        """True only when an API key and a pricing entry (or explicit price) exist."""
        return bool(self._env.get(self._api_key_env)) and self._pricing is not None

    def key_missing(self) -> bool:
        return not self._env.get(self._api_key_env)

    @property
    def ledger(self) -> SpendLedger:
        return self._ledger

    def analyze(self, messages: list[dict[str, str]], max_tokens: int = 600) -> FrontierAnalysis:
        """Stage-1 frontier call: returns analysis + usage/latency/cost.

        Refuses to make the call unless authorized; reserves the cost against
        the spend cap before sending.
        """
        if not self.authorized():
            raise FrontierNotConfigured(
                f"frontier {self._model!r} not authorized: set {self._api_key_env!r} "
                "in the environment (no API key) or configure prices/cap"
            )
        api_key = self._env[self._api_key_env]
        payload: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        # Estimate the outgoing cost from the prompt text so the cap is enforced
        # before (not after) the call.
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        approx_in = max(1, prompt_chars // 4)
        per_in, per_out = self._pricing  # type: ignore[misc]
        est_cost = (approx_in / 1000.0) * per_in + (max_tokens / 1000.0) * per_out
        self._ledger.reserve(est_cost)

        started = time.monotonic()
        raw = self._post(f"{self._endpoint}/chat/completions", payload, api_key)
        latency_s = time.monotonic() - started

        prompt_tokens, completion_tokens = _usage_tokens(raw)
        if self._pricing is not None:
            per_in, per_out = self._pricing
            actual_cost = (prompt_tokens / 1000.0) * per_in + (completion_tokens / 1000.0) * per_out
        else:
            actual_cost = 0.0
        # Settle: replace the pre-call reserve with the actual billed cost.
        self._ledger._spent = self._ledger.spent - est_cost + actual_cost  # noqa: SLF001

        choices = raw.get("choices") or [{}]
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "")
        text = content if isinstance(content, str) else json.dumps(content)
        return FrontierAnalysis(
            text=text,
            raw=raw,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_s=latency_s,
            cost_usd=actual_cost,
        )
