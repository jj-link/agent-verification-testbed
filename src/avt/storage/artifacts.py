"""Filesystem artifact store under a runtime root.

Layout (plan section 8)::

    <root>/artifacts/candidates/{candidate_id}/{manifest,trajectory,usage}.json
    <root>/artifacts/verifier-runs/{verification_id}/{request,response,scores}.json
    <root>/artifacts/official-results/{candidate_id}/result.json

Writes are atomic (temp file + rename) so an artifact is never observed
half-written, and are completed before any catalog record is indexed.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

__all__ = ["ArtifactStore"]


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.artifacts_dir = root / "artifacts"

    def _atomic_write_json(self, path: Path, data: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=True, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def write_candidate(
        self,
        candidate_id: str,
        manifest: object,
        trajectory: object,
        usage: object,
    ) -> dict[str, Path]:
        base = self.artifacts_dir / "candidates" / candidate_id
        manifest_p = base / "manifest.json"
        trajectory_p = base / "trajectory.json"
        usage_p = base / "usage.json"
        self._atomic_write_json(manifest_p, manifest)
        self._atomic_write_json(trajectory_p, trajectory)
        self._atomic_write_json(usage_p, usage)
        return {"manifest": manifest_p, "trajectory": trajectory_p, "usage": usage_p}

    def write_verification(
        self,
        verification_id: str,
        request: object,
        response: object,
        scores: object,
    ) -> dict[str, Path]:
        base = self.artifacts_dir / "verifier-runs" / verification_id
        request_p = base / "request.json"
        response_p = base / "response.json"
        scores_p = base / "scores.json"
        self._atomic_write_json(request_p, request)
        self._atomic_write_json(response_p, response)
        self._atomic_write_json(scores_p, scores)
        return {"request": request_p, "response": response_p, "scores": scores_p}

    def write_official_result(self, candidate_id: str, result: object) -> Path:
        base = self.artifacts_dir / "official-results" / candidate_id
        path = base / "result.json"
        self._atomic_write_json(path, result)
        return path
