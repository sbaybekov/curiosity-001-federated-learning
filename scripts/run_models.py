#!/usr/bin/env python
"""STAGE 3a/5 — baselines: isolated partners + centralized-oracle.

Isolated: each partner trains its own heads on its own silo, with the SAME
total local work as a federated run (rounds × epochs passes) — the status-quo
comparison, evaluated per-partner then averaged. Centralized-oracle: all silo
data pooled (the privacy-impossible upper bound). Both evaluated on the
never-touched held-out set with bootstrap CIs. Results → results/metrics/.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from fedlearn.config import MODELS_DIR, METRICS_DIR, TASK, resolve_config
from fedlearn.data.partitions import load_partitions
from fedlearn.eval.evaluate import state_metrics
from fedlearn.log import banner, get_logger, section, stage, summarize, timed
from fedlearn.models.heads import pack_heads, init_heads
from fedlearn.provenance import Provenance, snapshot_state

log = get_logger("stage3.baselines")


def main() -> None:
    parser = argparse.ArgumentParser(description="stage 3a — isolated & centralized baselines")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = resolve_config(args.config)
    stage(log, 3, 5, "MODELS & BASELINES — isolated vs centralized-oracle (FL comes next)")
    prove = Provenance("stage3-baselines")
    prove.begin({"config": config})

    parts = load_partitions(config)
    site_data, eval_data, dims = parts["site_data"], parts["eval_data"], parts["dims"]
    rounds = int(config["fl"]["rounds"])
    epochs = int(config["model"]["epochs_per_round"])
    lr = float(config["model"]["lr"])

    # ---- isolated: each partner, its own heads --------------------------------
    section(log, "regime: ISOLATED — each partner trains alone (status quo)")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    partner_metrics: dict[str, dict] = {}
    for cid in sorted(site_data):
        with timed(log, f"isolated training {cid}"):
            heads = init_heads(dims, TASK, seed=0)
            for ep in sorted(dims):
                for _ in range(rounds * epochs):
                    heads[ep].partial_fit(site_data[cid]["X"][ep],
                                          site_data[cid]["y"][ep], 1, lr)
            state, _ = pack_heads(heads)
            np.savez(MODELS_DIR / f"isolated_{cid}.npz", state=state)
            partner_metrics[cid] = state_metrics(
                state, dims, eval_data["X"], eval_data["y"], config, f"isolated {cid}")
    # isolated regime metric = mean of partners (spread logged for honesty)
    isolated: dict = {}
    for ep in sorted(dims):
        keys = [k for k in ("pr_auc", "pr_auc_lo", "pr_auc_hi", "roc_auc", "rmse",
                            "rmse_lo", "rmse_hi", "r2") if k in partner_metrics["p0"][ep]]
        spread = [partner_metrics[cid][ep]["pr_auc"] if "pr_auc" in partner_metrics[cid][ep]
                  else partner_metrics[cid][ep]["rmse"] for cid in sorted(site_data)]
        summarize(f"isolated spread {ep}", f"min={min(spread):.3f} max={max(spread):.3f}", log)
        isolated[ep] = {k: float(np.mean([partner_metrics[c][ep][k] for c in partner_metrics]))
                        for k in keys}
    METRICS_DIR.joinpath("regime_isolated.json").write_text(
        json.dumps({"isolated": isolated}, indent=2))

    # ---- centralized-oracle: pooled data --------------------------------------
    section(log, "regime: CENTRALIZED-ORACLE — all data pooled (privacy-impossible bound)")
    with timed(log, "centralized training"):
        heads = init_heads(dims, TASK, seed=0)
        for ep in sorted(dims):
            X_pool = np.concatenate([site_data[cid]["X"][ep] for cid in sorted(site_data)])
            y_pool = np.concatenate([site_data[cid]["y"][ep] for cid in sorted(site_data)])
            summarize(f"pooled {ep}", f"{len(y_pool)} rows", log)
            for _ in range(rounds * epochs):
                heads[ep].partial_fit(X_pool, y_pool, 1, lr)
        state, _ = pack_heads(heads)
        np.savez(MODELS_DIR / "centralized.npz", state=state)
    centralized = state_metrics(state, dims, eval_data["X"], eval_data["y"], config,
                               "centralized")
    METRICS_DIR.joinpath("regime_centralized.json").write_text(
        json.dumps({"centralized": centralized}, indent=2))

    prove.record({"regimes": ["isolated", "centralized"]},
                 artifacts=[METRICS_DIR / "regime_isolated.json",
                            METRICS_DIR / "regime_centralized.json"])
    banner(log, "STAGE 3a COMPLETE — next: bash scripts/run_federated.py")


if __name__ == "__main__":
    main()
