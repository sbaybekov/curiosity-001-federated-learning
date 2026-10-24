"""fedlearn.eval.coverage — chemical-space coverage (applicability domain).

Does FL broaden what a partner can confidently predict? For held-out (off-
silo) molecules we compute the Tanimoto distance to the NEAREST training
molecule under each regime's training set, then bin molecules by that
distance and compare per-bin accuracy. If FL's bin-accuracy curve sits above
isolated's in the FAR bins, FL demonstrably extended the applicability domain.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..log import get_logger, section, summarize

log = get_logger("eval.coverage")


def nearest_tanimoto(query: np.ndarray, targets: np.ndarray,
                     batch: int = 1024, log_=log) -> np.ndarray:
    """Max Tanimoto similarity of each query row against all target rows."""
    qry = query.astype(bool)
    tgt = targets.astype(bool)
    n_q, n_t = len(qry), len(tgt)
    out = np.zeros(n_q, dtype=np.float32)
    log_.info(f"  nearest-tanimoto: {n_q} queries × {n_t} targets (batched, {batch}/batch)")
    for i in range(0, n_q, batch):
        q = qry[i:i + batch]
        inter = q @ tgt.T
        q_pop = np.count_nonzero(q, axis=1, keepdims=True).astype(float)
        t_pop = np.count_nonzero(tgt, axis=1, keepdims=True).T.astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = inter / np.where(q_pop + t_pop == 0, 1.0, q_pop + t_pop - inter)
        out[i:i + batch] = sim.max(axis=1)
    return out


def coverage_table(
    heldout_sim: np.ndarray, correct: np.ndarray, bins: list[float],
    sim_label: str, log_=log,
) -> pd.DataFrame:
    """Per-bin (similarity range) coverage summary: n, accuracy."""
    edges = np.asarray(bins)
    digit = np.clip(np.digitize(heldout_sim, edges) - 1, 0, len(edges) - 2)
    rows = []
    for bin_idx in range(len(edges) - 1):
        mask = digit == bin_idx
        n = int(mask.sum())
        acc = float(np.mean(correct[mask])) if n else float("nan")
        rows.append({
            "sim_label": sim_label,
            "sim_bin": f"{edges[bin_idx]:.2f}–{edges[bin_idx + 1]:.2f}",
            "n_molecules": n,
            "accuracy": acc,
        })
    frame = pd.DataFrame(rows)
    for _, row in frame.iterrows():
        summarize(f"sim {row['sim_bin']:>12} [{sim_label}]",
                  f"n={row['n_molecules']:>5} acc={row['accuracy']:.3f}", log_)
    return frame


def coverage_comparison(config, parts, models: dict, log_=log) -> pd.DataFrame:
    """The t21 analysis: per-partner / per-regime accuracy vs training proximity.

    models = {"isolated": {cid: state}, "federated": state, "centralized": state}

    Key subtlety (logged, not hidden): isolated + federated train on the SAME
    molecules — what differs is visibility. So for each held-out molecule we
    measure Tanimoto to (a) partner p's OWN training set for the isolated
    reading, and (b) the UNION of all silos for the federated/centralized
    reading. In the FAR bins (unfamiliar chemistry), FL should beat the
    isolated partners: that is the "broadened applicability domain" claim.
    """
    from collections import OrderedDict

    from ..config import ENDPOINTS, TASK
    from ..models.heads import init_heads, unpack_heads

    bins = config["eval"]["coverage_bins"]
    n_partners = config["silos"]["n_partners"]
    cids = [f"p{p}" for p in range(n_partners)]
    dims = parts["dims"]
    frames: list[pd.DataFrame] = []

    for endpoint in ENDPOINTS:
        Xh, yh = parts["eval_data"]["X"][endpoint], parts["eval_data"]["y"][endpoint]
        section(log_, f"coverage — endpoint {endpoint}")
        # union of partner chemistry for this endpoint (fed + centralized basis)
        union = np.concatenate([parts["site_data"][cid]["X"][endpoint] for cid in cids])

        # ponytail: per-molecule regression "accuracy" = |err| ≤ median-|err|
        # on held-out — a rank proxy, fine for the domain story; upgrade =
        # per-bin RMSE if the coverage plot ever needs the real thing.
        def _acc(state):
            heads = init_heads(dims, TASK, seed=0)
            unpack_heads(state.astype("float32"),
                         OrderedDict((ep, dims[ep]) for ep in sorted(dims)), heads)
            if TASK[endpoint] == "cls":
                return ((heads[endpoint].scores(Xh) > 0).astype(int) == yh.astype(int))
            pred = heads[endpoint].scores(Xh)
            return np.abs(pred - yh) <= np.median(np.abs(pred - yh))

        # isolated: each partner vs its OWN chemistry
        for cid in cids:
            own = parts["site_data"][cid]["X"][endpoint]
            sim = nearest_tanimoto(Xh, own, log_=log_)
            frames.append(coverage_table(sim, _acc(models["isolated"][cid]), bins,
                                         f"isolated.{cid}.{endpoint}", log_))
        # federated + centralized: vs union
        for label in ("federated", "centralized"):
            sim = nearest_tanimoto(Xh, union, log_=log_)
            frames.append(coverage_table(sim, _acc(models[label]), bins,
                                         f"{label}.{endpoint}", log_))

    merged = pd.concat(frames, ignore_index=True)
    merged["regime"] = merged["sim_label"].str.split(".").str[0]
    return merged
