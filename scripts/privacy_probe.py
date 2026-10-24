#!/usr/bin/env python
"""STAGE 5/5 — privacy audit: membership inference, before/after DP.

Protocol: (1) probe the released federated model — can members (in-silo
molecules) be told apart from non-members (the scaffold-held-out set, trained
under NO regime)? (2) retrain the federation with DP ON (clip + Gaussian
noise), (3) re-probe the DP model. The before/after AUC comparison is the
deliverable — leakage shown, mitigation shown, honestly labeled.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict

import numpy as np

from fedlearn.config import MODELS_DIR, RESULTS_DIR, resolve_config
from fedlearn.data.partitions import load_partitions
from fedlearn.fl.server import run_federated
from fedlearn.log import banner, get_logger, section, stage, summarize
from fedlearn.models.heads import init_heads, unpack_heads
from fedlearn.privacy.dp import log_privacy_config, DPSettings
from fedlearn.privacy.membership import probe_mi
from fedlearn.provenance import Provenance

log = get_logger("stage5.privacy")


def _scores_for(state, endpoint, X, task_map) -> np.ndarray:
    from fedlearn.config import TASK

    heads = init_heads({endpoint: X.shape[1]}, {endpoint: TASK[endpoint]}, seed=0)
    unpack_heads(state.astype("float32"),
                 OrderedDict({endpoint: X.shape[1]}), heads)
    return heads[endpoint].scores(X)


def _member_nonmember_scores(state, parts, endpoint, bootstrap_seed=7):
    """Members = partner training molecules; non-members = global held-out."""
    rng = np.random.default_rng(bootstrap_seed)
    X_all = np.concatenate([parts["site_data"][f"p{p}"]["X"][endpoint]
                            for p in range(len(parts["site_data"]))])
    y_all = np.concatenate([parts["site_data"][f"p{p}"]["y"][endpoint]
                            for p in range(len(parts["site_data"]))])
    n_member = min(len(X_all), 500)
    idx = rng.choice(len(X_all), size=n_member, replace=False)
    member_scores = _scores_for(state, endpoint, X_all[idx], None)
    Xh = parts["eval_data"]["X"][endpoint]
    nonmember_scores = _scores_for(state, endpoint, Xh, None)
    if len(y_all) and Xh.shape[0] and endpoint in ("solubility", "logp"):
        # regression: members have small |error| — invert so 'high = member-ish'
        member_scores = -np.abs(member_scores - y_all[idx])
        nonmember_scores = -np.abs(nonmember_scores - parts["eval_data"]["y"][endpoint])
    return member_scores, nonmember_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="stage 5 — privacy audit")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--endpoint", default="ames",
                        choices=["ames", "herg", "solubility", "logp"])
    parser.add_argument("--dp-config", default="configs/privacy on".split(" ")[0], nargs="?")
    args = parser.parse_args()
    config = resolve_config(args.config)
    stage(log, 5, 5, "PRIVACY AUDIT — membership inference before/after DP")
    prove = Provenance("stage5-privacy")
    prove.begin({"config": config, "endpoint": args.endpoint})

    parts = load_partitions(config)
    endpoint = args.endpoint

    # ---- 1. probe the UNDP'd federated model --------------------------------
    section(log, "phase 1 — membership probe on the plain federated model")
    plain_state = np.load(MODELS_DIR / "federated.npz")["state"]
    m, nm = _member_nonmember_scores(plain_state, parts, endpoint)
    plain = probe_mi(m, nm, label=f"federated[{endpoint}] no-DP")

    # ---- 2. train a DP federated model ---------------------------------------
    dp_cfg = dict(config["privacy"])
    dp_cfg["enabled"] = True
    config["privacy"] = dp_cfg
    settings = DPSettings(**dp_cfg)
    log_privacy_config(settings, int(config["fl"]["rounds"]))
    section(log, "phase 2 — federated retrain WITH DP (clip + Gaussian)")
    result = run_federated(config, parts["site_data"], parts["eval_data"],
                           {ep: ("cls" if ep in ("ames", "herg") else "reg")
                            for ep in parts["dims"]})
    dp_state = result["final_state"]
    np.savez(MODELS_DIR / "federated_dp.npz", state=dp_state)

    # ---- 3. re-probe ----------------------------------------------------------
    section(log, "phase 3 — membership probe on the DP model")
    m2, nm2 = _member_nonmember_scores(dp_state, parts, endpoint)
    privatized = probe_mi(m2, nm2, label=f"federated[{endpoint}] DP")

    delta = plain["auc"] - privatized["auc"]
    verdict = (
        f"DP cut the membership-inference AUC by {delta:.3f} "
        f"({plain['auc']:.3f} → {privatized['auc']:.3f})"
        if plain["auc"] > 0.52 and delta > 0.02
        else f"No strong leakage signal to begin with (AUC {plain['auc']:.3f}) "
             f"or DP change small ({delta:+.3f}) — linear models leak little."
    )
    summarize("before/after", f"{plain['auc']:.3f} → {privatized['auc']:.3f} "
                              f"(Δ {delta:+.3f})", log)
    summarize("verdict", verdict, log)

    (RESULTS_DIR / "privacy_audit.json").write_text(json.dumps({
        "endpoint": endpoint,
        "plain": plain, "dp": privatized, "delta": delta, "verdict": verdict,
        "dp_config": dp_cfg,
    }, indent=2))
    prove.record({"delta_auc": delta},
                 artifacts=[MODELS_DIR / "federated_dp.npz",
                            RESULTS_DIR / "privacy_audit.json"])
    banner(log, "STAGE 5 COMPLETE — honest reading above; see results/privacy_audit.json")


if __name__ == "__main__":
    main()
