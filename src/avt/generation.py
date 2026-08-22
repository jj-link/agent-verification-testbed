"""Candidate generation: resumable, retryable Harbor actor runs.

Each (task, attempt) is a persisted job in the catalog. A successfully graded
candidate is recorded (artifacts + ground truth); infrastructure failures are
retried a bounded number of times; quality is never a reason to discard.

The Qwen Code settings mounted into Harbor derive from the frozen generator
config (temperature, max_tokens), so candidate pools are configuration-bound.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from avt.config import Config
from avt.storage import ArtifactStore, Catalog, GroundTruth
from avt.storage.ids import candidate_id, experiment_id

__all__ = ["GenerationError", "InfrastructureFailure", "GenerationService"]

_JOB_TYPE = "candidate"
_INFRA_RETRIES = 2
_OPENAI_KEY = "sk-local"  # local endpoint performs no auth
_HARBOR_DATASET = "terminal-bench@2.0"
_MOUNT_TARGET = "/mnt/qwen-settings.json"


class GenerationError(Exception):
    """Base for generation failures."""


class InfrastructureFailure(GenerationError):
    """A retryable infrastructure failure (container/endpoint)."""


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    task_id: str
    attempt_index: int
    reward: float | None
    job_dir: Path
    exception: str | None = None


def qwen_settings_for(config: Config) -> dict[str, object]:
    """Qwen Code settings derived from the frozen generator config."""
    sampling: dict[str, object] = {
        "temperature": config.generator.temperature,
        "max_tokens": config.generator.max_tokens,
    }
    return {
        "model": {
            "generationConfig": {
                "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                "samplingParams": sampling,
            }
        }
    }


class _HarborRunner:
    """Invokes the pinned Harbor CLI for a single candidate (one attempt)."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def _write_qwen_settings(self, config: Config) -> Path:
        """Write the config-derived Qwen settings and return the path to mount."""
        settings_dir = self.repo_root / Path(config.storage.root) / "qwen-settings"
        settings_dir.mkdir(parents=True, exist_ok=True)
        path = settings_dir / f"{config.experiment.name}.json"
        path.write_text(
            json.dumps(qwen_settings_for(config), sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        return path

    def run(
        self,
        config: Config,
        task_id: str,
        attempt: int,
        out_dir: Path,
        job_name: str,
    ) -> Path:
        if shutil.which("harbor") is None:
            raise GenerationError("harbor not on PATH; install it with `uv sync --extra tbench`")
        qwen_settings = self._write_qwen_settings(config)
        mounts = json.dumps(
            [
                {
                    "type": "bind",
                    "source": str(qwen_settings.resolve()),
                    "target": _MOUNT_TARGET,
                    "read_only": True,
                }
            ]
        )
        cmd = [
            "harbor",
            "run",
            "-d",
            _HARBOR_DATASET,
            "-a",
            config.generator.harness,
            "-m",
            config.generator.model,
            "-i",
            task_id,
            "-n",
            "1",
            "-k",
            "1",
            "--mounts",
            mounts,
            "--ae",
            f"QWEN_CODE_SYSTEM_SETTINGS_PATH={_MOUNT_TARGET}",
            "--ae",
            f"OPENAI_BASE_URL={config.generator.endpoint}",
            "--ae",
            f"OPENAI_API_KEY={_OPENAI_KEY}",
            "-o",
            str(out_dir),
            "--job-name",
            job_name,
            "--yes",
        ]
        result = subprocess.run(cmd, cwd=self.repo_root, capture_output=True, text=True)
        job_dir = out_dir / job_name
        if result.returncode != 0 or not job_dir.is_dir():
            detail = (result.stderr or result.stdout or "")[-2000:]
            raise InfrastructureFailure(f"harbor run failed rc={result.returncode}: {detail}")
        return job_dir


def _find_trial_result(job_dir: Path) -> tuple[Path, float | None, str | None]:
    """Locate the trial result, reward, and any agent exception in a job dir."""
    for candidate in job_dir.iterdir():
        if not candidate.is_dir():
            continue
        res = candidate / "result.json"
        if res.exists():
            data = json.loads(res.read_text(encoding="utf-8"))
            reward: float | None = None
            vr = data.get("verifier_result")
            if isinstance(vr, dict):
                rewards = vr.get("rewards")
                if isinstance(rewards, dict) and "reward" in rewards:
                    reward = float(rewards["reward"])
            exc = data.get("exception_info")
            exc_name: str | None = None
            if isinstance(exc, dict):
                exc_name = str(exc.get("exception_type") or "")
            elif isinstance(exc, str) and exc:
                exc_name = exc
            return res, reward, exc_name
    return job_dir, None, "no trial result found"


class GenerationService:
    def __init__(self, config: Config, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self.catalog = Catalog(Path(config.storage.metadata_db))
        self.ground_truth = GroundTruth(Path(config.storage.ground_truth_db))
        self.artifacts = ArtifactStore(Path(config.storage.root))
        self.runner = _HarborRunner(repo_root)

    def experiment_id(self) -> str:
        return experiment_id(self.config.raw)

    def _task_attempts(self) -> list[tuple[str, int]]:
        tasks = [
            line.strip()
            for line in Path(self.config.experiment.task_file)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        return [
            (t, attempt)
            for t in tasks
            for attempt in range(self.config.experiment.candidates_per_task)
        ]

    def generate_one(self, task_id: str, attempt: int) -> CandidateResult:
        exp = self.experiment_id()
        cand = candidate_id(exp, task_id, attempt)
        with self.catalog.connect() as scoped:
            job = scoped.enqueue_or_claim(_JOB_TYPE, cand, {"task": task_id, "attempt": attempt})
            if job is None:
                # Already SUCCEEDED (skip) or claimed concurrently.
                return CandidateResult(cand, task_id, attempt, None, self.repo_root)
            # Keep the job RUNNING across internal retry rounds; only mark
            # terminal state (SUCCEEDED / PERMANENT_FAILED) after the loop.

        max_rounds = 1 + _INFRA_RETRIES
        last_exc: str | None = None
        for round_no in range(max_rounds):
            out_dir = (
                self.repo_root
                / Path(self.config.storage.root)
                / "generation"
                / task_id
                / str(attempt)
                / str(round_no)
            )
            job_name = f"gen-{task_id}-{attempt}-{round_no}"
            job_dir = out_dir / job_name

            # A prior graded round (from a crash) is reused; a prior *failed*
            # round advances to the next round slot rather than overwriting.
            if job_dir.is_dir():
                trial, reward, exc_name = _find_trial_result(job_dir)
                if trial is not None:
                    if exc_name is not None:
                        last_exc = exc_name
                        continue
                    self._persist(cand, task_id, attempt, trial, reward, job_dir)
                    with self.catalog.connect() as scoped:
                        scoped.mark_succeeded(_JOB_TYPE, cand)
                    return CandidateResult(cand, task_id, attempt, reward, job_dir)

            try:
                job_dir = self.runner.run(self.config, task_id, attempt, out_dir, job_name)
            except InfrastructureFailure as exc:
                last_exc = str(exc)
                continue  # stay RUNNING; retry next round

            trial, reward, exc_name = _find_trial_result(job_dir)
            if exc_name:
                last_exc = exc_name
                continue

            # Graded (reward may be 0.0 for a valid failed candidate).
            self._persist(cand, task_id, attempt, trial, reward, job_dir)
            with self.catalog.connect() as scoped:
                scoped.mark_succeeded(_JOB_TYPE, cand)
            return CandidateResult(cand, task_id, attempt, reward, job_dir)

        with self.catalog.connect() as scoped:
            scoped.mark_failed(_JOB_TYPE, cand, permanent=True)
        raise InfrastructureFailure(
            f"candidate {cand} failed after {max_rounds} rounds: {last_exc}"
        )

    def _persist(
        self,
        cand: str,
        task_id: str,
        attempt: int,
        trial: Path,
        reward: float | None,
        job_dir: Path,
    ) -> None:
        self.artifacts.write_candidate(
            cand,
            {"task": task_id, "attempt": attempt},
            {"trial_result": str(trial), "job_dir": str(job_dir)},
            {"attempt": attempt, "task": task_id, "harness": self.config.generator.harness},
        )
        self.artifacts.write_official_result(cand, {"reward": reward, "source": str(trial)})
        with self.ground_truth.connect() as g:
            g.put(cand, task_id, reward, {"reward": reward})

    def generate_all(self) -> list[CandidateResult]:
        results: list[CandidateResult] = []
        for task, attempt in self._task_attempts():
            try:
                results.append(self.generate_one(task, attempt))
            except InfrastructureFailure:
                results.append(
                    CandidateResult(
                        candidate_id(self.experiment_id(), task, attempt),
                        task,
                        attempt,
                        None,
                        self.repo_root,
                        exception="final infrastructure failure",
                    )
                )
        return results
