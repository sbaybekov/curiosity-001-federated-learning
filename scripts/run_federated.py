#!/usr/bin/env python
"""STAGE 3b/5 — the federated loop (Flower FedAvg, manual fallback).

Trains the same per-endpoint heads across the N partner silos via weight
exchange only. Writes: results/models/federated.npz (final global state),
results/metrics/regime_federated.json (held-out metrics + CIs),
results/runs/fed_history.jsonl (per-round loss/wall-time for the timing
analysis). DP switch lives in configs/config.yaml → privacy.enabled.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from fedlearn.config import MODELS_DIR, METRICS_DIR, RUNS_DIR, resolve_config
from fedlearn.data.partitions import load_partitions
from fedlearn.eval.evaluate import state_metrics
from fedlearn.fl.server import run_federated
from fedlearn.log import banner, get_logger, stage, summarize
from fedlearn.provenance import Provenance, snapshot_state
from fedlearn.privacy.dp import log_privacy_config, DPSettings

log = get_logger("stage3.federated")


def main() -> None:
    parser = argparse.ArgumentParser(description="stage 3b — federated training")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--backend", choices=["flower", "manual"], default=None,
                        help="override configs/config.yaml fl.backend")
    args = parser.parse_args()
    config = resolve_config(args.config)
    if args.backend:
        config["fl"]["backend"] = args.backend
        log.info(f"backend overridden → {args.backend}")

    stage(log, 3, 5, "FEDERATED LOOP — FedAvg over partner silos, only weights travel")
    prove = Provenance("stage3-federated")
    prove.begin({"config": config, "stage": "3b"})

    parts = load_partitions(config)
    site_data, eval_data, dims = parts["site_data"], parts["eval_data"], parts["dims"]

    settings = DPSettings(
        enabled=bool(config["privacy"]["enabled"]),
        clip_C=float(config["privacy"]["clip_C"]),
        noise_multiplier=float(config["privacy"]["noise_multiplier"]),
    )
    log_privacy_config(settings, int(config["fl"]["rounds"]))

    result = run_federated(
        config, site_data, eval_data,
        {ep: ("cls" if ep in ("ames", "herg") else "reg") for ep in dims})
    summarize("backend used", result["backend"], log)
    summarize("preliminary held-out loss", result["heldout_loss"], log)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(MODELS_DIR / "federated.npz", state=result["final_state"])
    log.info(f"final global state → {MODELS_DIR / 'federated.npz'}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    federated = state_metrics(result["final_state"], dims, eval_data["X"],
                              eval_data["y"], config, f"federated[{result['backend']}]")
    METRICS_DIR.joinpath("regime_federated.json").write_text(
        json.dumps({"federated": federated}, indent=2))

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / "fed_history.jsonl", "w") as fh:
        for entry in result["history"]:
            fh.write(json.dumps(entry) + "\n")
    log.info(f"per-round history → {RUNS_DIR / 'fed_history.jsonl'} "
             f"({len(result['history'])} rounds)")

    prove.record({"backend": result["backend"]},
                 artifacts=[MODELS_DIR / "federated.npz",
                            METRICS_DIR / "regime_federated.json"])
    banner(log, "STAGE 3b COMPLETE — next: bash scripts/benchmark.py (money plot)")


if __name__ == "__main__":
    main()
