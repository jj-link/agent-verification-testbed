"""Discrete Qwen judge (plan section 13.1-13.3).

One model request per (pair, criterion) reads the probability the model places
on the single-token score labels for trajectory A and trajectory B from the
response ``top_logprobs``. The discrete judge scores each trajectory as the
numeric value of its highest-probability score token (G=5 for the smoke test).

Leakage: the request contains only the public task description and the two
rendered trajectories; grader outcome, hidden tests, reference solutions, and
candidate pass/fail labels are never included.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.request import Request, urlopen

from avt.config import Config
from avt.doctor import G5_LABELS
from avt.frontier import (
    FrontierClient,
    FrontierError,
    build_frontier_messages,
)
from avt.pairs import display_order
from avt.rendering import render_trajectory
from avt.storage.artifacts import ArtifactStore
from avt.storage.catalog import Catalog
from avt.storage.ids import experiment_id, verification_id

__all__ = [
    "DiscreteJudge",
    "MalformedVerifier",
    "MissingLabelError",
    "SCORE_LABELS",
    "build_messages",
    "build_single_messages",
    "build_frontier_single_messages",
    "normalize_endpoint",
    "labels_for_granularity",
    "scores_from_logprobs",
]

# Frozen granularity-5 labels from Stage 4 (`doctor.G5_LABELS`): letters tokenize
# to a single token with a stable ordering. A < B < C < D < E maps to 1..5.
SCORE_LABELS = tuple(G5_LABELS)
_LABEL_VALUE = {label: i + 1 for i, label in enumerate(G5_LABELS)}


def labels_for_granularity(granularity: int) -> tuple[str, ...]:
    """The first ``granularity`` uppercase letters as single-token score labels."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if granularity < 1 or granularity > len(letters):
        raise ValueError(f"granularity {granularity} out of range 1..{len(letters)}")
    return tuple(letters[:granularity])


_TOP_LOGPROBS = 100
# Delimiter/whitespace characters allowed around a single letter token.
_DELIMS = set("<>(),.:;") | {chr(32), chr(9), chr(10), chr(13), chr(39), chr(34)}

CRITERIA: dict[str, str] = {
    "specification": "Does the approach satisfy ALL of the explicit task requirements?",
    "output": "Does it produce the expected output, artifacts, or format?",
    "errors": "Does it avoid meaningful failure signals in logs and tool output?",
}

_PROMPT_VERSION = "score-label-map-v2"
_SINGLE_TARGET_PROMPT_VERSION = "single-target-v1"
_FRONTIER_SINGLE_PROMPT_VERSION = "single-target-frontier-v1"


def _score_mapping(labels: tuple[str, ...]) -> str:
    return ", ".join(f"{label}={value}" for value, label in enumerate(labels, start=1))


def _domain_instruction(labels: tuple[str, ...]) -> str:
    return (
        "You are an expert reviewer evaluating two AI coding-agent trajectories for "
        "the same software engineering task. Score each trajectory from 1 (poor) to "
        f"{len(labels)} (excellent) for the stated criterion. Higher is better. "
        f"Use this exact score-to-letter mapping: {_score_mapping(labels)}."
    )


# Plan 15: a malformed (unparsable) output is retried exactly once with this
# strict format reminder appended to the user message.
def _malformed_reminder(labels: tuple[str, ...]) -> str:
    """Strict-format retry reminder scoped to the frozen score-label set."""
    label_str = "{" + ",".join(labels) + "}"
    return (
        "\n\nYour previous response was not exactly two valid score letters. Output "
        "ONLY two single letters separated by one space: the first scores trajectory "
        f"A and the second scores trajectory B. Valid letters are {label_str}, with "
        f"this exact mapping: {_score_mapping(labels)}. Any other letter is invalid. "
        "No other text or explanation."
    )


# host.docker.internal is a container-only alias; the judge runs host-side (AVT
# orchestration on the workstation), so the model is reached on host loopback.
# Mapping it here keeps the frozen config (and hence the experiment id) stable.
_HOST_INTERNAL = "host.docker.internal"
_HOST_LOOPBACK = "127.0.0.1"


class MalformedVerifier(Exception):
    """The model response could not be parsed into two score tokens."""


class MissingLabelError(Exception):
    """The endpoint did not return a logprob for every configured score label."""


def normalize_endpoint(endpoint: str) -> str:
    return endpoint.replace(_HOST_INTERNAL, _HOST_LOOPBACK)


def build_messages(
    task_description: str,
    trajectory_a: str,
    trajectory_b: str,
    criterion: str,
    labels: tuple[str, ...] = SCORE_LABELS,
    prompt_version: str = _PROMPT_VERSION,
) -> list[dict[str, str]]:
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion: {criterion!r}")
    if prompt_version != _PROMPT_VERSION:
        raise ValueError(f"unknown verifier prompt version: {prompt_version!r}")
    label_str = "{" + ",".join(labels) + "}"
    user_msg = (
        f"{CRITERIA[criterion]}\n\n"
        f"TASK:\n{task_description}\n\n"
        f"TRAJECTORY A:\n{trajectory_a}\n\n"
        f"TRAJECTORY B:\n{trajectory_b}\n\n"
        "Output ONLY two single score letters separated by one space. The first "
        "letter scores trajectory A; the second scores trajectory B. "
        f"Valid letters are {label_str}, with this exact mapping: "
        f"{_score_mapping(labels)}. Any other letter is invalid. No other text."
    )
    return [
        {"role": "system", "content": _domain_instruction(labels)},
        {"role": "user", "content": user_msg},
    ]


def build_single_messages(
    task_description: str,
    trajectory_a: str,
    trajectory_b: str,
    criterion: str,
    target: str,
    labels: tuple[str, ...] = SCORE_LABELS,
    prompt_version: str = _SINGLE_TARGET_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """Single-trajectory scoring prompt (forced-token G>5 path).

    Both trajectories are included so the pair context is identical to the
    two-token path; only the score target differs. The model is asked for ONE
    score letter, so the single forced score token is conditioned on a clean
    ``...TRAJECTORY X`` context rather than on a glued two-letter continuation
    with a suppressed separator (plan 13.2 fallback #4).
    """
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion: {criterion!r}")
    if prompt_version != _SINGLE_TARGET_PROMPT_VERSION:
        raise ValueError(f"unknown single-target prompt version: {prompt_version!r}")
    if target not in ("A", "B"):
        raise ValueError(f"single-target must be 'A' or 'B', got {target!r}")
    label_str = "{" + ",".join(labels) + "}"
    user_msg = (
        f"{CRITERIA[criterion]}\n\n"
        f"TASK:\n{task_description}\n\n"
        f"TRAJECTORY A:\n{trajectory_a}\n\n"
        f"TRAJECTORY B:\n{trajectory_b}\n\n"
        f"Score ONLY TRAJECTORY {target}. Output ONLY one score letter from "
        f"{label_str}, using this exact mapping: {_score_mapping(labels)}. "
        "Any other letter is invalid. No other text."
    )
    return [
        {"role": "system", "content": _domain_instruction(labels)},
        {"role": "user", "content": user_msg},
    ]


def build_frontier_single_messages(
    task_description: str,
    trajectory_a: str,
    trajectory_b: str,
    frontier_analysis: str,
    criterion: str,
    target: str,
    labels: tuple[str, ...] = SCORE_LABELS,
    prompt_version: str = _FRONTIER_SINGLE_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """Single-trajectory scoring prompt conditioned on frontier analysis.

    Stage-2 of plan 24: the frontier model's comparative reasoning / coarse
    1-10 scores (``frontier_analysis``) are folded into the prompt context, then
    the local single-trajectory forced-token scorer outputs one score letter.
    The frontier analysis is placed as neutral evidence, never as a verdict the
    judge must copy.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion: {criterion!r}")
    if prompt_version != _FRONTIER_SINGLE_PROMPT_VERSION:
        raise ValueError(f"unknown single-target prompt version: {prompt_version!r}")
    if target not in ("A", "B"):
        raise ValueError(f"single-target must be 'A' or 'B', got {target!r}")
    label_str = "{" + ",".join(labels) + "}"
    user_msg = (
        f"{CRITERIA[criterion]}\n\n"
        f"TASK:\n{task_description}\n\n"
        f"TRAJECTORY A:\n{trajectory_a}\n\n"
        f"TRAJECTORY B:\n{trajectory_b}\n\n"
        "THIRD-PARTY REVIEW (for context only; do not treat as authoritative):\n"
        f"{frontier_analysis}\n\n"
        f"Score ONLY TRAJECTORY {target}. Output ONLY one score letter from "
        f"{label_str}, using this exact mapping: {_score_mapping(labels)}. "
        "Any other letter is invalid. No other text."
    )
    return [
        {"role": "system", "content": _domain_instruction(labels)},
        {"role": "user", "content": user_msg},
    ]


def _label_of(token: str, labels: tuple[str, ...]) -> str | None:
    """Return the single score letter a generated token denotes, or None.

    The frozen single-token labels frequently surface wrapped in delimiter or
    whitespace tokens (``" A"``, ``"E>"``, ``"<E>"``, ``"E,"``). A token maps
    to a letter only if it contains exactly one label letter and no other
    non-delimiter content.
    """
    letters = [ch for ch in token if ch in labels]
    if len(letters) != 1:
        return None
    for ch in token:
        if ch not in labels and ch not in _DELIMS:
            return None
    return letters[0]


def _label_probs(
    top_logprobs: list[dict[str, object]], labels: tuple[str, ...]
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in top_logprobs:
        token = str(item.get("token"))
        label = _label_of(token, labels)
        if label is not None:
            lp = item.get("logprob")
            if isinstance(lp, (int, float)):
                weights[label] = weights.get(label, 0.0) + math.exp(float(lp))
    missing = [lab for lab in labels if lab not in weights]
    if missing:
        # Plan 13.2: missing score-token probabilities are a configuration
        # failure; never silently assign probability zero. A label absent from
        # the returned top-logprobs (e.g. a rare letter at G=20) is treated as
        # missing data and fails visibly, per the frozen plan.
        raise MissingLabelError(f"endpoint omitted logprob for labels {missing}")
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0.0:
        # Plan 13.4: p(score) must be a valid probability over the G labels.
        raise MissingLabelError(f"non-finite/zero label mass at a score position: {weights}")
    return {label: weight / total for label, weight in weights.items()}


def _discrete(probs: dict[str, float], labels: tuple[str, ...]) -> int:
    label_value = {label: i + 1 for i, label in enumerate(labels)}
    best = max(probs.items(), key=lambda kv: kv[1])[0]
    return label_value[best]


def _expected(probs: dict[str, float], labels: tuple[str, ...]) -> float:
    label_value = {label: i + 1 for i, label in enumerate(labels)}
    return sum(label_value[label] * prob for label, prob in probs.items())


def _both_label_probs(
    content_logprobs: list[dict[str, object]], labels: tuple[str, ...]
) -> tuple[dict[str, float], dict[str, float]]:
    """Return the A and B score-token label distributions.

    The two scores are the first two generated tokens whose sampled token maps
    to a score label (a separator/whitespace token is skipped).
    """
    label_indices = [
        i
        for i, c in enumerate(content_logprobs)
        if isinstance(c.get("token"), str) and _label_of(str(c["token"]), labels) in labels
    ]
    if len(label_indices) < 2:
        raise MalformedVerifier(f"expected >= 2 score-token positions, found {len(label_indices)}")
    top0 = content_logprobs[label_indices[0]].get("top_logprobs")
    top1 = content_logprobs[label_indices[1]].get("top_logprobs")
    return (
        _label_probs(list(top0) if isinstance(top0, list) else [], labels),
        _label_probs(list(top1) if isinstance(top1, list) else [], labels),
    )


def scores_from_logprobs(
    content_logprobs: list[dict[str, object]], labels: tuple[str, ...] = SCORE_LABELS
) -> tuple[int, int]:
    """Discrete A/B scores: each trajectory's highest-probability label value."""
    pa, pb = _both_label_probs(content_logprobs, labels)
    return _discrete(pa, labels), _discrete(pb, labels)


def expected_scores_from_logprobs(
    content_logprobs: list[dict[str, object]], labels: tuple[str, ...] = SCORE_LABELS
) -> tuple[float, float]:
    """Continuous A/B expected scores: label-value weighted probability sums."""
    pa, pb = _both_label_probs(content_logprobs, labels)
    return _expected(pa, labels), _expected(pb, labels)


def _post_json(url: str, payload: dict[str, object], timeout: int = 120) -> dict[str, Any]:
    req = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())  # type: ignore[no-any-return]


@dataclass(frozen=True)
class PairScore:
    verification_id: str
    pair_id: str
    criterion: str
    repetition: int
    display_order: tuple[str, str]
    score_a: int
    score_b: int
    status: str = "SUCCEEDED"
    malformed_attempts: int = 0


class DiscreteJudge:
    def __init__(self, config: Config, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self.catalog = Catalog(Path(config.storage.metadata_db))
        self.artifacts = ArtifactStore(Path(config.storage.root))
        self.exp = experiment_id(config.raw)
        self.endpoint = normalize_endpoint(config.verifier.endpoint).rstrip("/")
        self._max_tokens = int(getattr(config.verifier, "max_tokens", 16) or 16)
        self._top_logprobs = int(getattr(config.verifier, "top_logprobs", 300) or 300)
        self._labels = labels_for_granularity(int(config.verifier.granularity))
        self._prompt_version = str(
            getattr(config.verifier, "prompt_version", _PROMPT_VERSION) or _PROMPT_VERSION
        )
        # Plan 13.2 fallback #4: forced-token scoring. At G>5 the endpoint does
        # not reliably surface every score label in the natural top-logprobs, so
        # we bias the single-token label logits to force them into the returned
        # top-k. A uniform bias across all labels preserves relative probability
        # (constant cancels in softmax) and guarantees every label is present.
        # An explicit `forced_token_scoring` flag enables the same protocol at
        # G=5, isolating granularity from call protocol in the G5-vs-G20
        # ablation. It cannot disable the required G>5 fallback.
        self._forced_tokens = int(config.verifier.granularity) > 5 or bool(
            config.verifier.forced_token_scoring
        )
        self._forced_bias = 100.0
        # Validated against the served tokenizer: /v1/tokenize maps each single
        # uppercase letter to one token with id ord(ch)-33 (A=32 .. T=51).
        self._label_token_ids = (
            {ord(ch) - 33 for ch in self._labels} if self._forced_tokens else set()
        )
        self._frontier: FrontierClient | None = None
        frontier_cfg = config.frontier
        if frontier_cfg:
            api_key_env = str(frontier_cfg.get("api_key_env") or "")
            endpoint = str(frontier_cfg.get("endpoint") or "").rstrip("/")
            model = str(frontier_cfg.get("model") or "")
            spend_cap_val: object = frontier_cfg.get("spend_cap_usd")
            if not isinstance(spend_cap_val, (int, float)):
                raise FrontierError("frontier configured but `spend_cap_usd` must be a number")
            spend_cap = spend_cap_val
            if spend_cap is None:
                raise FrontierError("frontier configured but missing `spend_cap_usd`")
            if not api_key_env:
                raise FrontierError("frontier configured but missing `api_key_env`")
            if not endpoint or not model:
                raise FrontierError("frontier configured but missing `endpoint`/`model`")
            self._frontier = FrontierClient(
                model=model,
                endpoint=endpoint,
                api_key_env=api_key_env,
                spend_cap_usd=float(spend_cap),
                pricing=(
                    cast("dict[str, object]", frontier_cfg["pricing"])
                    if isinstance(frontier_cfg.get("pricing"), dict)
                    else None
                ),
            )

    def _body_for(self, candidate_id_: str, task_id: str) -> str:
        with self.catalog.connect() as scoped:
            row = next(
                (
                    r
                    for r in scoped.list_candidates(self.exp, task_id)
                    if r["candidate_id"] == candidate_id_
                ),
                None,
            )
        if row is None or not row.get("artifact_path"):
            raise MalformedVerifier(f"{candidate_id_}: no artifact for candidate")
        tj = Path(str(row["artifact_path"])) / "trajectory.json"
        meta = json.loads(tj.read_text(encoding="utf-8"))
        trial_result = meta.get("trial_result")
        atif = Path(str(trial_result)).parent / "agent" / "trajectory.json"
        raw_budget = self.config.rendering.get("max_pair_context_tokens", 120_000)
        body_budget = int(raw_budget if isinstance(raw_budget, (int, float)) else 120_000) // 2
        rendered = render_trajectory(candidate_id_, task_id, atif, body_budget)
        return rendered.body_text

    def _verifier_identity(self) -> dict[str, object]:
        forced = (
            {
                "scoring": "forced_token_single_target",
                "forced_label_token_ids": sorted(self._label_token_ids),
                "forced_bias": self._forced_bias,
                "single_target_prompt_version": _SINGLE_TARGET_PROMPT_VERSION,
            }
            if self._forced_tokens
            else {}
        )
        frontier = (
            {
                "scoring": "frontier_assisted_single_target",
                "frontier_model": self.config.frontier.get("model"),
                "single_target_prompt_version": _FRONTIER_SINGLE_PROMPT_VERSION,
                "spend_cap_usd": self._frontier.ledger.cap,
            }
            if self._frontier is not None
            else {}
        )
        return {
            "model": self.config.verifier.model,
            "endpoint": self.config.verifier.endpoint,
            "criteria": list(self.config.verifier.criteria),
            "granularity": self.config.verifier.granularity,
            "repetitions": self.config.verifier.repetitions,
            "labels": list(self._labels),
            "max_tokens": self._max_tokens,
            "top_logprobs": self._top_logprobs,
            "prompt_version": self._prompt_version,
            **forced,
            **frontier,
        }

    def _persist(
        self,
        pair_id_: str,
        task_id: str,
        criterion: str,
        repetition: int,
        disp_a: str,
        disp_b: str,
        request: object,
        response: object,
        score_a: int,
        score_b: int,
        malformed_attempts: int = 0,
    ) -> PairScore:
        verifier_cfg = self._verifier_identity()
        vid = verification_id(pair_id_, verifier_cfg, criterion, repetition, disp_a + "+" + disp_b)
        scores: dict[str, object] = {
            "score_a": score_a,
            "score_b": score_b,
            "criterion": criterion,
            "repetition": repetition,
        }
        artifact_paths = self.artifacts.write_verification(vid, request, response, scores)
        with self.catalog.connect() as scoped:
            scoped.record_verification(
                vid,
                pair_id_,
                criterion,
                repetition,
                f"{disp_a}+{disp_b}",
                "SUCCEEDED",
                str(artifact_paths["request"]),
                str(artifact_paths["response"]),
                str(artifact_paths["scores"]),
                malformed_attempts,
            )
        return PairScore(vid, pair_id_, criterion, repetition, (disp_a, disp_b), score_a, score_b)

    def _persist_failed(
        self,
        pair_id_: str,
        task_id: str,
        criterion: str,
        repetition: int,
        disp_a: str,
        disp_b: str,
        request: object,
        response: object,
        malformed_attempts: int,
    ) -> PairScore:
        """Record a permanently-malformed verification as FAILED (plan 21 metric).

        The twice-malformed raw response and request are retained as artifacts so
        the malformed-output rate is measurable rather than lost.
        """
        verifier_cfg = self._verifier_identity()
        vid = verification_id(pair_id_, verifier_cfg, criterion, repetition, disp_a + "+" + disp_b)
        scores: dict[str, object] = {
            "status": "FAILED",
            "reason": "malformed_response",
            "malformed_attempts": malformed_attempts,
        }
        artifact_paths = self.artifacts.write_verification(vid, request, response, scores)
        with self.catalog.connect() as scoped:
            scoped.record_verification(
                vid,
                pair_id_,
                criterion,
                repetition,
                f"{disp_a}+{disp_b}",
                "FAILED",
                str(artifact_paths["request"]),
                str(artifact_paths["response"]),
                str(artifact_paths["scores"]),
                malformed_attempts,
            )
        return PairScore(
            vid,
            pair_id_,
            criterion,
            repetition,
            (disp_a, disp_b),
            0,
            0,
            status="FAILED",
            malformed_attempts=malformed_attempts,
        )

    def _verify_pair(
        self,
        pair: dict[str, object],
        task_id: str,
        criterion: str,
        repetition: int,
    ) -> PairScore:
        if self._frontier is not None:
            return self._verify_pair_frontier(pair, task_id, criterion, repetition)
        if self._forced_tokens:
            return self._verify_pair_forced(pair, task_id, criterion, repetition)
        ca = str(pair["candidate_a"])
        cb = str(pair["candidate_b"])
        disp_a, disp_b = display_order(ca, cb, self.config.experiment.seed)
        with self.catalog.connect() as scoped:
            task_description = scoped.get_task_instruction(self.exp, task_id) or ""
        body_a = self._body_for(disp_a, task_id)
        body_b = self._body_for(disp_b, task_id)
        messages = build_messages(
            task_description,
            body_a,
            body_b,
            criterion,
            self._labels,
            self._prompt_version,
        )
        base_payload: dict[str, object] = {
            "model": self.config.verifier.model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "logprobs": True,
            "top_logprobs": self._top_logprobs,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        # Plan 15 retry policy: a malformed output is retried exactly once with a
        # strict format reminder; missing score-token probabilities are a
        # configuration failure and fail this job immediately (never retried).
        reminder_messages = [
            dict(m, content=m["content"] + _malformed_reminder(self._labels))
            if m["role"] == "user"
            else m
            for m in messages
        ]
        score_a = score_b = 0
        malformed_count = 0
        last_error: Exception | None = None
        raw: dict[str, Any] = {}
        for attempt in range(2):  # initial draw plus one malformed retry
            # Reminder applies only on the retry, keeping the frozen prompt clean.
            payload = dict(base_payload, messages=reminder_messages if attempt else messages)
            raw = _post_json(f"{self.endpoint}/chat/completions", payload)
            choices = raw.get("choices") or [{}]
            lp = (choices[0].get("logprobs") or {}) if isinstance(choices[0], dict) else {}
            content_lp: list[dict[str, Any]] = list(lp.get("content") or [])
            try:
                score_a, score_b = scores_from_logprobs(content_lp, self._labels)
                last_error = None
                break
            except MissingLabelError:
                # Configuration failure: never retry, fail this job visibly.
                raise
            except MalformedVerifier as exc:
                malformed_count += 1
                last_error = exc
                continue
        if last_error is not None:
            # Persistently malformed after the single retry: record a FAILED row
            # (plan 21 malformed-rate metric) and continue, rather than aborting
            # the whole run. MissingLabelError is a config failure and already
            # raised above.
            return self._persist_failed(
                str(pair["pair_id"]),
                task_id,
                criterion,
                repetition,
                disp_a,
                disp_b,
                payload,
                raw,
                malformed_count,
            )
        return self._persist(
            str(pair["pair_id"]),
            task_id,
            criterion,
            repetition,
            disp_a,
            disp_b,
            payload,
            raw,
            score_a,
            score_b,
            malformed_count,
        )

    def _verify_pair_frontier(
        self,
        pair: dict[str, object],
        task_id: str,
        criterion: str,
        repetition: int,
    ) -> PairScore:
        """Frontier-assisted two-stage scoring (plan 24).

        Stage 1 asks the authorized frontier model for comparative reasoning and
        coarse 1-10 scores for the pair once per (pair, criterion). Stage 2
        conditions the local forced single-target scorer on that analysis and
        combines the two single-position distributions into one synthetic
        response for the unchanged discrete+continuous pipeline. Frontier model,
        prompt, and spend-cap settings are part of the verification identity, so
        a frontier run never collides with the plain local run. Every paid call
        is gated behind ``FrontierClient.authorized()`` and the spend cap; with
        no key/cap configured this raises rather than calling out.
        """
        assert self._frontier is not None
        ca = str(pair["candidate_a"])
        cb = str(pair["candidate_b"])
        disp_a, disp_b = display_order(ca, cb, self.config.experiment.seed)
        with self.catalog.connect() as scoped:
            task_description = scoped.get_task_instruction(self.exp, task_id) or ""
        body_a = self._body_for(disp_a, task_id)
        body_b = self._body_for(disp_b, task_id)

        # Stage 1: frontier comparative analysis (once per pair + criterion).
        if not self._frontier.authorized():
            raise MissingLabelError(
                f"frontier {self.config.frontier.get('model')!r} not authorized "
                f"(set {self.config.frontier.get('api_key_env')!r} and a spend cap)"
            )
        frontier_msgs = build_frontier_messages(
            task_description, body_a, body_b, CRITERIA[criterion]
        )
        analysis = self._frontier.analyze(frontier_msgs)

        # Stage 2: local forced single-target scorer conditioned on analysis.
        contents: dict[str, dict[str, Any]] = {}
        raw_calls: dict[str, object] = {"frontier": analysis.raw}
        payloads: dict[str, object] = {"frontier": frontier_msgs}
        for target in ("A", "B"):
            messages = build_frontier_single_messages(
                task_description,
                body_a,
                body_b,
                analysis.text,
                criterion,
                target,
                self._labels,
                _FRONTIER_SINGLE_PROMPT_VERSION,
            )
            payload: dict[str, object] = {
                "model": self.config.verifier.model,
                "messages": messages,
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": len(self._labels),
                "logit_bias": {str(tok): self._forced_bias for tok in self._label_token_ids},
                "chat_template_kwargs": {"enable_thinking": False},
            }
            payloads[target] = payload
            raw = _post_json(f"{self.endpoint}/chat/completions", payload)
            raw_calls[target] = raw
            choices = raw.get("choices") or [{}]
            lp = (choices[0].get("logprobs") or {}) if isinstance(choices[0], dict) else {}
            content = list(lp.get("content") or [])
            first = content[0] if content else None
            if not isinstance(first, dict):
                raise MissingLabelError(
                    f"frontier-assisted forced call (criterion {criterion}, {target}) "
                    "had no valid content position"
                )
            contents[target] = first

        synthetic_content: list[dict[str, Any]] = [contents["A"], contents["B"]]
        score_a, score_b = scores_from_logprobs(synthetic_content, self._labels)
        combined_response: dict[str, object] = {
            "frontier_assisted": True,
            "frontier_model": self.config.frontier.get("model"),
            "frontier_analysis": analysis.text,
            "frontier_usage": {
                "prompt_tokens": analysis.prompt_tokens,
                "completion_tokens": analysis.completion_tokens,
                "latency_s": analysis.latency_s,
                "cost_usd": analysis.cost_usd,
                "spend_cap_usd": self._frontier.ledger.cap,
                "cumulative_spent_usd": self._frontier.ledger.spent,
            },
            "granularity": len(self._labels),
            "calls": raw_calls,
            "choices": [{"logprobs": {"content": synthetic_content}}],
        }
        request_artifact: dict[str, object] = {
            "frontier_assisted": True,
            "frontier_model": self.config.frontier.get("model"),
            "calls": payloads,
        }
        return self._persist(
            str(pair["pair_id"]),
            task_id,
            criterion,
            repetition,
            disp_a,
            disp_b,
            request_artifact,
            combined_response,
            score_a,
            score_b,
            0,
        )

    def _verify_pair_forced(
        self,
        pair: dict[str, object],
        task_id: str,
        criterion: str,
        repetition: int,
    ) -> PairScore:
        """Forced-token G>5 scoring (plan 13.2 fallback #4).

        Two separate single-token calls, one per trajectory, with identical
        full-pair context and an explicit single-trajectory score target. A
        uniform logit bias forces the G label tokens into the returned top-k at
        that one position, so every configured label's probability is present
        (no silent zero-fill). The two single-position distributions are
        combined into one synthetic two-position response so all downstream
        scoring (discrete + continuous) is unchanged.
        """
        ca = str(pair["candidate_a"])
        cb = str(pair["candidate_b"])
        disp_a, disp_b = display_order(ca, cb, self.config.experiment.seed)
        with self.catalog.connect() as scoped:
            task_description = scoped.get_task_instruction(self.exp, task_id) or ""
        body_a = self._body_for(disp_a, task_id)
        body_b = self._body_for(disp_b, task_id)

        contents: dict[str, dict[str, Any]] = {}
        raw_calls: dict[str, object] = {}
        payloads: dict[str, object] = {}
        for target in ("A", "B"):
            messages = build_single_messages(
                task_description,
                body_a,
                body_b,
                criterion,
                target,
                self._labels,
                _SINGLE_TARGET_PROMPT_VERSION,
            )
            payload: dict[str, object] = {
                "model": self.config.verifier.model,
                "messages": messages,
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": len(self._labels),
                "logit_bias": {str(tok): self._forced_bias for tok in self._label_token_ids},
                "chat_template_kwargs": {"enable_thinking": False},
            }
            payloads[target] = payload
            raw = _post_json(f"{self.endpoint}/chat/completions", payload)
            raw_calls[target] = raw
            choices = raw.get("choices") or [{}]
            lp = (choices[0].get("logprobs") or {}) if isinstance(choices[0], dict) else {}
            content = list(lp.get("content") or [])
            first = content[0] if content else None
            if not isinstance(first, dict):
                raise MissingLabelError(
                    f"forced single-token call (criterion {criterion}, {target}) "
                    "had no valid content position"
                )
            contents[target] = first

        synthetic_content: list[dict[str, Any]] = [contents["A"], contents["B"]]
        # Discrete plan-13.3 scores from the two single-position distributions.
        score_a, score_b = scores_from_logprobs(synthetic_content, self._labels)
        combined_response: dict[str, object] = {
            "forced_token_scoring": True,
            "granularity": len(self._labels),
            "calls": raw_calls,
            "choices": [{"logprobs": {"content": synthetic_content}}],
        }
        request_artifact: dict[str, object] = {
            "forced_token_scoring": True,
            "granularity": len(self._labels),
            "calls": payloads,
        }
        return self._persist(
            str(pair["pair_id"]),
            task_id,
            criterion,
            repetition,
            disp_a,
            disp_b,
            request_artifact,
            combined_response,
            score_a,
            score_b,
            0,
        )

    def verify_all(self) -> list[PairScore]:
        tasks = [
            line.strip()
            for line in Path(self.config.experiment.task_file)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        results: list[PairScore] = []
        expected_count = 0
        succeeded_count = 0
        verifier_cfg = self._verifier_identity()
        for task in tasks:
            with self.catalog.connect() as scoped:
                pairs = scoped.list_pairs(self.exp, task)
            for pair in pairs:
                pair_id_ = str(pair["pair_id"])
                ca = str(pair["candidate_a"])
                cb = str(pair["candidate_b"])
                disp_a, disp_b = display_order(ca, cb, self.config.experiment.seed)
                current_ids: dict[tuple[str, int], str] = {}
                for criterion in self.config.verifier.criteria:
                    for rep in range(self.config.verifier.repetitions):
                        current_ids[(criterion, rep)] = verification_id(
                            pair_id_,
                            verifier_cfg,
                            criterion,
                            rep,
                            disp_a + "+" + disp_b,
                        )
                expected_count += len(current_ids)
                # Clean cutover: stale prompt/model/output-policy identities must
                # not satisfy resume or contaminate downstream aggregation. The
                # content-addressed artifacts remain on disk for audit.
                with self.catalog.connect() as scoped:
                    scoped.remove_superseded_verifications(pair_id_, set(current_ids.values()))
                for (criterion, rep), current_vid in current_ids.items():
                    with self.catalog.connect() as scoped:
                        status = scoped.verification_status(current_vid)
                    if status == "SUCCEEDED":
                        succeeded_count += 1
                        continue
                    if status == "FAILED":
                        continue
                    result = self._verify_pair(pair, task, criterion, rep)
                    results.append(result)
                    if result.status == "SUCCEEDED":
                        succeeded_count += 1
        verified = expected_count > 0 and succeeded_count == expected_count
        with self.catalog.connect() as scoped:
            scoped.set_experiment_stage(self.exp, "VERIFIED" if verified else "VERIFYING")
        return results
