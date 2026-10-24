"""fedlearn.provenance — MLflow-backed run records with a JSONL fallback.

Every experiment run gets: resolved config, metrics, data hashes, git
revision, and artifact paths. MLflow is used when importable (tracking URI
points into ``results/mlruns``); if it is missing or fails, a JSONL manifest
under ``results/manifests/`` carries the same information so no run is ever
silently unrecorded. Each script calls :func:`snapshot_state` at the start and
:func:`record_run` at the end.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from .config import MANIFESTS_DIR, MLFLOW_DIR
from .log import get_logger

log = get_logger("provenance")


def sha256_of_files(paths: list[Path | str]) -> dict[str, str]:
    """Per-file sha256 hex digests (for data-hash pins in manifests)."""
    import hashlib

    digests: dict[str, str] = {}
    for path in paths:
        p = Path(path)
        if not p.exists():
            digests[str(p)] = "MISSING"
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        digests[str(p)] = h.hexdigest()
    return digests


def git_revision() -> dict[str, str]:
    """Short commit + dirty flag; degrades to 'unknown' outside a repo."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:  # noqa: BLE001 — provenance must never crash the run
        log.warning("could not read git revision (not a repo?)")
        return {"git_commit": "unknown", "git_dirty": True}


class Provenance:
    """One MLflow experiment (or JSONL fallback) per pipeline stage."""

    def __init__(self, experiment_name: str, log_level: bool = True) -> None:
        self.experiment = experiment_name
        self._mlflow = _init_mlflow(experiment_name)
        self._jsonl_path = MANIFESTS_DIR / "run_manifest.jsonl"
        MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)
        self._active_run_id: str | None = None
        if log_level:
            if self._mlflow is not None:
                log.info(f"provenance: MLflow experiment '{experiment_name}' (uri={MLFLOW_DIR})")
            else:
                log.info(f"provenance: MLflow unavailable → JSONL manifest at {self._jsonl_path}")

    def begin(self, params: dict[str, Any]) -> str:
        """Open a run and record starting parameters (config snapshot)."""
        self._active_run_id = str(uuid.uuid4())[:8]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if self._mlflow is not None:
            try:
                run = self._mlflow.start_run(run_name=f"{stamp}-{self.experiment}")
                self._active_run_id = run.info.run_id
                self._mlflow.log_params(_flatten(params))
            except Exception as exc:  # noqa: BLE001
                log.warning(f"mlflow.start_run failed ({exc}) — falling back to JSONL")
                self._mlflow = None
        self._append({
            "record_type": "begin",
            "run_id": self._active_run_id,
            "experiment": self.experiment,
            "params": params,
        })
        return self._active_run_id

    def record(self, metrics: dict[str, Any], artifacts: list[Path | str] | None = None) -> None:
        """Finish the active run with metrics + optional artifact paths."""
        entry = {
            "record_type": "metrics",
            "run_id": self._active_run_id,
            "experiment": self.experiment,
            "metrics": metrics,
        }
        if artifacts:
            entry["artifacts"] = [str(a) for a in artifacts]
            if self._mlflow is not None:
                try:
                    for artifact in artifacts:
                        if Path(artifact).exists():
                            self._mlflow.log_artifact(str(artifact))
                except Exception as exc:  # noqa: BLE001
                    log.warning(f"mlflow.log_artifact failed ({exc})")
        if self._mlflow is not None:
            try:
                self._mlflow.log_metrics(metrics)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"mlflow.log_metrics failed ({exc})")
        self._append(entry)
        if self._mlflow is not None:
            try:
                self._mlflow.end_run()
            except Exception:
                pass

    def _append(self, entry: dict) -> None:
        with open(self._jsonl_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")


def _init_mlflow(experiment: str):
    """Return the mlflow module wired to the local URI, or None."""
    # mlflow >= 2.17 filesystem-trackers gated behind an opt-in env var —
    # set it before any mlflow call so a local results/mlruns still works.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow  # type: ignore

        mlflow.set_tracking_uri(MLFLOW_DIR.as_uri())
        mlflow.set_experiment(experiment)
        return mlflow
    except Exception as exc:  # noqa: BLE001
        log.warning(f"mlflow init failed ({exc}) — using JSONL manifest only")
        return None


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """One-level flatten for mlflow params (nested groups become a.b keys)."""
    flat: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        elif isinstance(value, (list, tuple)):
            flat[name] = json.dumps(value, default=str)
        else:
            flat[name] = value
    return flat


def snapshot_state(config: dict, data_files: list[Path | str] | None = None) -> dict:
    """Assemble the reproducibility header written by every script."""
    state = {"config": config, **git_revision()}
    if data_files:
        state["data_sha256"] = {str(k): v[:16] for k, v in sha256_of_files(data_files).items()}
    return state
