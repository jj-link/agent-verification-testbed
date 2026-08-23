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
from typing import Any
from urllib.request import Request, urlopen

from avt.config import Config
from avt.doctor import G5_LABELS
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
    "normalize_endpoint",
    "scores_from_logprobs",
]

# Frozen granularity-5 labels from Stage 4 (`doctor.G5_LABELS`): letters tokenize
# to a single token with a stable ordering. A < B < C < D < E maps to 1..5.
SCORE_LABELS = tuple(G5_LABELS)
_LABEL_VALUE = {label: i + 1 for i, label in enumerate(G5_LABELS)}
_TOP_LOGPROBS = 100
# Delimiter/whitespace characters allowed around a single letter token.
_DELIMS = set("<>(),.:;") | {chr(32), chr(9), chr(10), chr(13), chr(39), chr(34)}

CRITERIA: dict[str, str] = {
    "specification": "Does the approach satisfy ALL of the explicit task requirements?",
    "output": "Does it produce the expected output, artifacts, or format?",
    "errors": "Does it avoid meaningful failure signals in logs and tool output?",
}

_DOMAIN_INSTRUCTION = (
    "You are an expert reviewer evaluating two AI coding-agent trajectories for "
    "the same software engineering task. Score each trajectory from 1 (poor) to "
    "5 (excellent) for the stated criterion. Higher is better."
)
# Plan 15: a malformed (unparsable) output is retried exactly once with this
# strict format reminder appended to the user message.
_MALFORMED_REMINDER = (
    "\n\nYour previous response was not exactly two score letters. Output ONLY "
    "two single letters from {A,B,C,D,E} separated by a space - the first scores "
    "trajectory A, the second scores trajectory B. No other text, no explanation."
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
    task_description: str, trajectory_a: str, trajectory_b: str, criterion: str
) -> list[dict[str, str]]:
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion: {criterion!r}")
    user_msg = (
        f"{CRITERIA[criterion]}\n\n"
        f"TASK:\n{task_description}\n\n"
        f"TRAJECTORY A:\n{trajectory_a}\n\n"
        f"TRAJECTORY B:\n{trajectory_b}\n\n"
        f"Output ONLY: <A> <B>, where <A> is a single letter from "
        f"{{{','.join(G5_LABELS)}}} scoring trajectory A and <B> a single letter "
        f"from {{{','.join(G5_LABELS)}}} scoring trajectory B. No other text."
    )
    return [
        {"role": "system", "content": _DOMAIN_INSTRUCTION},
        {"role": "user", "content": user_msg},
    ]


def _label_of(token: str) -> str | None:
    """Return the single score letter a generated token denotes, or None.

    The frozen single-token labels frequently surface wrapped in delimiter or
    whitespace tokens (``" A"``, ``"E>"``, ``"<E>"``, ``"E,"``). A token maps
    to a letter only if it contains exactly one A-E letter and no other
    non-delimiter content.
    """
    letters = [ch for ch in token if ch in "ABCDE"]
    if len(letters) != 1:
        return None
    for ch in token:
        if ch not in "ABCDE" and ch not in _DELIMS:
            return None
    return letters[0]


def _label_probs(top_logprobs: list[dict[str, object]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in top_logprobs:
        token = str(item.get("token"))
        label = _label_of(token)
        if label is not None:
            lp = item.get("logprob")
            if isinstance(lp, (int, float)):
                weights[label] = weights.get(label, 0.0) + math.exp(float(lp))
    missing = [lab for lab in SCORE_LABELS if lab not in weights]
    if missing:
        # Plan 13.2: missing score-token probabilities are a configuration
        # failure; never silently assign probability zero.
        raise MissingLabelError(f"endpoint omitted logprob for labels {missing}")
    total = sum(weights.values())
    if not math.isfinite(total) or total <= 0.0:
        # Plan 13.4: p(score) must be a valid probability over the G labels.
        raise MissingLabelError(f"non-finite/zero label mass at a score position: {weights}")
    return {label: weight / total for label, weight in weights.items()}


def _discrete(probs: dict[str, float]) -> int:
    best = max(probs.items(), key=lambda kv: kv[1])[0]
    return _LABEL_VALUE[best]


def _expected(probs: dict[str, float]) -> float:
    return sum(_LABEL_VALUE[label] * prob for label, prob in probs.items())


def _both_label_probs(
    content_logprobs: list[dict[str, object]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Return the A and B score-token label distributions.

    The two scores are the first two generated tokens whose sampled token maps
    to a score label (a separator/whitespace token is skipped).
    """
    label_indices = [
        i
        for i, c in enumerate(content_logprobs)
        if isinstance(c.get("token"), str) and _label_of(str(c["token"])) in SCORE_LABELS
    ]
    if len(label_indices) < 2:
        raise MalformedVerifier(f"expected >= 2 score-token positions, found {len(label_indices)}")
    top0 = content_logprobs[label_indices[0]].get("top_logprobs")
    top1 = content_logprobs[label_indices[1]].get("top_logprobs")
    return (
        _label_probs(list(top0) if isinstance(top0, list) else []),
        _label_probs(list(top1) if isinstance(top1, list) else []),
    )


def scores_from_logprobs(content_logprobs: list[dict[str, object]]) -> tuple[int, int]:
    """Discrete A/B scores: each trajectory's highest-probability label value."""
    pa, pb = _both_label_probs(content_logprobs)
    return _discrete(pa), _discrete(pb)


def expected_scores_from_logprobs(
    content_logprobs: list[dict[str, object]],
) -> tuple[float, float]:
    """Continuous A/B expected scores: label-value weighted probability sums."""
    pa, pb = _both_label_probs(content_logprobs)
    return _expected(pa), _expected(pb)


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
        return {
            "model": self.config.verifier.model,
            "endpoint": self.config.verifier.endpoint,
            "criteria": list(self.config.verifier.criteria),
            "granularity": self.config.verifier.granularity,
            "repetitions": self.config.verifier.repetitions,
            "labels": list(SCORE_LABELS),
            "max_tokens": self._max_tokens,
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
        ca = str(pair["candidate_a"])
        cb = str(pair["candidate_b"])
        disp_a, disp_b = display_order(ca, cb, self.config.experiment.seed)
        with self.catalog.connect() as scoped:
            task_description = scoped.get_task_instruction(self.exp, task_id) or ""
        body_a = self._body_for(disp_a, task_id)
        body_b = self._body_for(disp_b, task_id)
        messages = build_messages(task_description, body_a, body_b, criterion)
        base_payload: dict[str, object] = {
            "model": self.config.verifier.model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "logprobs": True,
            "top_logprobs": _TOP_LOGPROBS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        # Plan 15 retry policy: a malformed output is retried exactly once with a
        # strict format reminder; missing score-token probabilities are a
        # configuration failure and fail this job immediately (never retried).
        reminder_messages = [
            dict(m, content=m["content"] + _MALFORMED_REMINDER) if m["role"] == "user" else m
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
                score_a, score_b = scores_from_logprobs(content_lp)
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

    def verify_all(self) -> list[PairScore]:
        tasks = [
            line.strip()
            for line in Path(self.config.experiment.task_file)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        results: list[PairScore] = []
        for task in tasks:
            with self.catalog.connect() as scoped:
                pairs = scoped.list_pairs(self.exp, task)
            for pair in pairs:
                # Resume: only make model calls for keys without a recorded
                # terminal (SUCCEEDED or FAILED) outcome.
                with self.catalog.connect() as scoped:
                    done = scoped.terminal_verification_keys(str(pair["pair_id"]))
                for criterion in self.config.verifier.criteria:
                    for rep in range(self.config.verifier.repetitions):
                        if (criterion, rep) in done:
                            continue
                        results.append(self._verify_pair(pair, task, criterion, rep))
        # VERIFIED only with full usable coverage. A FAILED (persistently
        # malformed) verification anywhere in this experiment's pairs removes
        # coverage; stay VERIFYING so the gap stays visible (plan 14: fail
        # visibly, never silently degrade coverage). Pool-wide count is used,
        # not this run's `results`, so a resumed run that executes no new keys
        # still sees a pre-existing FAILED row.
        with self.catalog.connect() as scoped:
            failed = scoped.count_failed_verifications(self.exp)
        with self.catalog.connect() as scoped:
            scoped.set_experiment_stage(self.exp, "VERIFIED" if failed == 0 else "VERIFYING")
        return results
