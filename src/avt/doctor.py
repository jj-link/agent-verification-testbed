"""Endpoint diagnostics for the AVT doctor command.

Validates the local Qwen endpoint identity and its score-logprob capability for
an ordered single-token label set, using only the standard library so the base
package needs no HTTP dependency.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

__all__ = ["G5_LABELS", "DoctorResult", "run_doctor"]


# Ordered, single-token score labels for granularity G=5 (letters tokenize to
# one token and have a stable ordering). A < B < C < D < E maps to 1..5.
G5_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E")

_TIMEOUT_S = 60


class DoctorResult:
    """Aggregated diagnostic output."""

    def __init__(self, check: str, ok: bool, detail: str) -> None:
        self.check = check
        self.ok = ok
        self.detail = detail

    def __str__(self) -> str:
        return f"[{'PASS' if self.ok else 'FAIL'}] {self.check}: {self.detail}"


def _default_base_url() -> str:
    return os.environ.get("LOCAL_QWEN_URL", "http://100.86.3.45:8888/v1")


def _post_json(base: str, path: str, payload: dict[str, object]) -> dict[str, object]:
    req = urllib.request.Request(
        base.rstrip("/") + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        data: dict[str, object] = json.load(resp)
        return data


def _get_json(base: str, path: str) -> dict[str, object]:
    req = urllib.request.Request(base.rstrip("/") + path, method="GET")
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        data: dict[str, object] = json.load(resp)
        return data


def _is_http_error(exc: Exception) -> tuple[bool, str]:
    if isinstance(exc, urllib.error.HTTPError):
        return True, f"HTTP {exc.code}: {exc.read().decode()[:300]}"
    if isinstance(exc, urllib.error.URLError):
        return True, f"connection error: {exc.reason}"
    return False, str(exc)


def run_doctor(
    base_url: str | None = None,
    labels: tuple[str, ...] = G5_LABELS,
) -> list[DoctorResult]:
    """Run endpoint identity and logprob-scoreability checks."""
    base = (base_url or _default_base_url()).rstrip("/")
    results: list[DoctorResult] = []

    # 1. Endpoint reachability / availability.
    try:
        models = _get_json(base, "/models")
        results.append(DoctorResult("endpoint availability", True, f"{base}/models reachable"))
    except Exception as exc:  # noqa: BLE001
        results.append(DoctorResult("endpoint availability", False, _is_http_error(exc)[1]))
        return results

    # 2. Endpoint model identity.
    served = models.get("data")
    model_id: str | None = None
    if isinstance(served, list) and served:
        first = served[0]
        if isinstance(first, dict):
            model_id = first.get("id")
            max_len = first.get("max_model_len")
            detail = f"served model id={model_id!r}"
            if max_len is not None:
                detail += f", max_model_len={max_len}"
            results.append(DoctorResult("endpoint model identity", bool(model_id), detail))
        else:
            results.append(DoctorResult("endpoint model identity", False, "malformed /models data"))
    else:
        results.append(DoctorResult("endpoint model identity", False, "/models returned no models"))

    # 3 + 4. Score-logprob capability: complete per-position top-logprobs, and
    # presence of every configured single-token label.
    prompt_tail = (
        "You are an expert reviewer. Output ONLY your final score as a single "
        "letter after the tag.\n<score_A>"
    )
    payload: dict[str, object] = {
        "model": model_id if isinstance(model_id, str) else "qwen3.8-27b-sglang",
        "messages": [
            {"role": "system", "content": "You are an expert reviewer."},
            {"role": "user", "content": prompt_tail},
        ],
        "max_tokens": 4,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": max(20, len(labels)),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        resp = _post_json(base, "/chat/completions", payload)
    except Exception as exc:  # noqa: BLE001
        results.append(DoctorResult("logprob access", False, _is_http_error(exc)[1]))
        return results

    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        results.append(DoctorResult("logprob access", False, "no choices in response"))
        return results

    logprobs = choices[0].get("logprobs")
    if not isinstance(logprobs, dict) or not logprobs.get("content"):
        results.append(
            DoctorResult(
                "logprob access",
                False,
                "server did not return per-token logprobs (requires a logprob-capable "
                "serving mode such as MTP; DSpark does not support return_logprob)",
            )
        )
        return results

    results.append(DoctorResult("logprob access", True, "server returned per-token logprobs"))

    # Require a SINGLE content position whose top-logprobs contains every label
    # together. Unioning alternatives across positions would let A..E pass even
    # if no decoder position could actually produce all of them as one
    # token-level choice, so we check per position.
    matched_position: int | None = None
    for idx, item in enumerate(logprobs["content"]):
        if not isinstance(item, dict):
            continue
        tops = item.get("top_logprobs")
        if not isinstance(tops, list):
            continue
        toks = {t["token"] for t in tops if isinstance(t, dict) and isinstance(t.get("token"), str)}
        if all(lab in toks for lab in labels):
            matched_position = idx
            break

    if matched_position is None:
        results.append(
            DoctorResult(
                "score-token single-token property",
                False,
                f"no single content position contained all labels {list(labels)} "
                "together in top-logprobs (raise top_logprobs or reduce it to >= len(labels))",
            )
        )
    else:
        results.append(
            DoctorResult(
                "score-token single-token property",
                True,
                f"all labels {list(labels)} occur together as distinct tokens at "
                f"content position {matched_position}",
            )
        )
    return results


def format_results(results: list[DoctorResult]) -> str:
    return "\n".join(str(r) for r in results)
