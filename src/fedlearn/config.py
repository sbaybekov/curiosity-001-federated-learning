"""fedlearn.config — paths, endpoints, and the single resolved experiment config.

DEFAULTS below are the source of truth; ``configs/config.yaml`` (if present) is
deep-merged over them, then any CLI overrides win last. Every script echoes the
RESOLVED CONFIG at startup and snapshots it into the run manifest, so any run
can be reproduced from its own log.
"""

from __future__ import annotations

import copy
from pathlib import Path

from .log import get_logger

log = get_logger("config")

# --------------------------------------------------------------------------
# Repo layout
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = ROOT / "docs"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = PROCESSED_DIR / "splits"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
METRICS_DIR = RESULTS_DIR / "metrics"
RUNS_DIR = RESULTS_DIR / "runs"
MANIFESTS_DIR = RESULTS_DIR / "manifests"
MLFLOW_DIR = RESULTS_DIR / "mlruns"
FEATURES_DIR = RESULTS_DIR / "features"

# --------------------------------------------------------------------------
# Endpoints — 2 classification, 2 regression
# --------------------------------------------------------------------------
ENDPOINTS: tuple[str, ...] = ("ames", "herg", "solubility", "logp")
TASK: dict[str, str] = {"ames": "cls", "herg": "cls", "solubility": "reg", "logp": "reg"}
DEFAULT_SEED = 42

# --------------------------------------------------------------------------
# Config defaults — mirrors configs/config.yaml (keep in sync)
# ponytail: configs/config.yaml is the deliverable artifact; DEFAULTS here
# exists only so notebooks/scripts run when configs/ is absent. Drop DEFAULTS
# (require the file) once the repo is stable.
# --------------------------------------------------------------------------
DEFAULTS: dict = {
    "data": {
        "raw_dir": "data/raw",
        "processed_dir": "data/processed",
        "endpoints": list(ENDPOINTS),
        "heldout_frac": 0.2,
    },
    "silos": {"n_partners": 3, "mode": "scaffold", "seed": DEFAULT_SEED},
    "model": {
        "nbits": 2048,
        "radius": 2,
        "epochs_per_round": 3,
        "lr": 0.01,
        "reg_penalty": 0.0001,
    },
    "fl": {"rounds": 20, "backend": "flower", "fraction_fit": 1.0, "seed": DEFAULT_SEED},
    "eval": {
        "bootstrap_reps": 1000,
        "coverage_bins": [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "privacy": {"enabled": False, "clip_C": 1.0, "noise_multiplier": 1.1, "delta": 1e-5},
}


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge ``overrides`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_config(config_file: str | Path | None = None, cli_overrides: dict | None = None) -> dict:
    """Merge configs/config.yaml (if given) + CLI overrides over DEFAULTS.

    The result is logged key-by-key so a run's log alone documents its setup.
    """
    merged = copy.deepcopy(DEFAULTS)
    source = None
    if config_file is not None:
        src = Path(config_file)
        if not src.exists():
            log.warning(f"config file {src} not found — falling back to defaults")
        else:
            import yaml  # pyyaml ships with hydra-core

            with open(src, "r", encoding="utf-8") as fh:
                merged = _deep_merge(merged, yaml.safe_load(fh) or {})
            source = str(src)
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)
    if source:
        log.info(f"config file: {source}")
    log.info("resolved config:")
    _log_config(merged)
    return merged


def _log_config(cfg: dict) -> None:
    for group, values in cfg.items():
        for key, value in values.items():
            log.info(f"    [{group:<9}] {key:<22} = {value}")
