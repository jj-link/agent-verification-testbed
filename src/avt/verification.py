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
    probs: dict[str, float] = {}
    for item in top_logprobs:
        token = str(item.get("token"))
        label = _label_of(token)
        if label is not None:
            lp = item.get("logprob")
            if isinstance(lp, (int, float)):
                probs[label] = probs.get(label, 0.0) + math.exp(float(lp))
    missing = [lab for lab in SCORE_LABELS if lab not in probs]
    if missing:
        # Plan 13.2: missing score-token probabilities are a configuration
        # failure; never silently assign probability zero.
        raise MissingLabelError(f"endpoint omitted logprob for labels {missing}")
    return probs


def _discrete(probs: dict[str, float]) -> int:
    best = max(probs.items(), key=lambda kv: kv[1])[0]
    return _LABEL_VALUE[best]


def scores_from_logprobs(content_logprobs: list[dict[str, object]]) -> tuple[int, int]:
    """Extract discrete A/B scores from an OpenAI logprob response.

    The two scores are the first two generated tokens whose sampled token is a
    score label (a separator token like a space is skipped).
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
    pa = _label_probs(list(top0) if isinstance(top0, list) else [])
    pb = _label_probs(list(top1) if isinstance(top1, list) else [])
    return _discrete(pa), _discrete(pb)


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


class DiscreteJudge:
    def __init__(self, config: Config, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self.catalog = Catalog(Path(config.storage.metadata_db))
        self.artifacts = ArtifactStore(Path(config.storage.root))
        self.exp = experiment_id(config.raw)
        self.endpoint = normalize_endpoint(config.verifier.endpoint).rstrip("/")

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
    ) -> PairScore:
        verifier_cfg = {
            "model": self.config.verifier.model,
            "endpoint": self.config.verifier.endpoint,
            "criteria": list(self.config.verifier.criteria),
            "granularity": self.config.verifier.granularity,
            "repetitions": self.config.verifier.repetitions,
            "labels": list(SCORE_LABELS),
        }
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
            )
        return PairScore(vid, pair_id_, criterion, repetition, (disp_a, disp_b), score_a, score_b)

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
        payload: dict[str, object] = {
            "model": self.config.verifier.model,
            "messages": messages,
            "max_tokens": 16,
            "logprobs": True,
            "top_logprobs": _TOP_LOGPROBS,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        # The endpoint is stochastic: a single draw occasionally returns prose
        # with no isolated score letter (malformed) or omits a label. Bounded
        # retry mirrors the plan's "malformed verifier output -> retry" policy.
        score_a = score_b = 0
        last_issue: Exception | None = None
        for _attempt in range(3):
            raw = _post_json(f"{self.endpoint}/chat/completions", payload)
            choices = raw.get("choices") or [{}]
            lp = (choices[0].get("logprobs") or {}) if isinstance(choices[0], dict) else {}
            content_lp: list[dict[str, Any]] = list(lp.get("content") or [])
            try:
                score_a, score_b = scores_from_logprobs(content_lp)
            except (MalformedVerifier, MissingLabelError) as exc:
                last_issue = exc
                continue
            last_issue = None
            break
        if last_issue is not None:
            raise last_issue
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
                for criterion in self.config.verifier.criteria:
                    for rep in range(self.config.verifier.repetitions):
                        results.append(self._verify_pair(pair, task, criterion, rep))
        with self.catalog.connect() as scoped:
            scoped.set_experiment_stage(self.exp, "VERIFIED")
        return results
