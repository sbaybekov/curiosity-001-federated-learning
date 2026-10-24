#!/usr/bin/env python
"""STAGE 1/5 — download + normalize + scaffold split + partner silos.

Usage:  uv run python scripts/download_data.py [--config configs/config.yaml]
Logs every step: TDC fetch per endpoint, canonicalization drops, scaffold
split sizes, per-partner chemistry stats. Writes data/raw/*.csv and
data/processed/splits/*.parquet + splits.json.
"""

from __future__ import annotations

import argparse
import json

from fedlearn.config import SPLIT_DIR, resolve_config
from fedlearn.data.download import download_all
from fedlearn.data.split import build_silos
from fedlearn.log import banner, get_logger, stage
from fedlearn.provenance import Provenance, snapshot_state

log = get_logger("stage1.download")


def main() -> None:
    parser = argparse.ArgumentParser(description="stage 1 — data & silos")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = resolve_config(args.config)

    stage(log, 1, 5, "DATA & SILOS — four public endpoints, one schema, N non-IID partners")
    banner(log, "FedMol stage 1: download → normalize → scaffold split → silos")
    prove = Provenance("stage1-download")
    state = snapshot_state(config)
    prove.begin(state)

    paths = download_all(config["data"]["raw_dir"])
    summary = build_silos(config)

    prove.record({"n_endpoints": 4}, artifacts=[
        SPLIT_DIR / "splits.json",
        *paths.values(),
    ])
    banner(log, "STAGE 1 COMPLETE — next: bash scripts/run_validation.py")
    log.info("see data/processed/splits/splits.json for the summary table")


if __name__ == "__main__":
    main()
