"""fedlearn.privacy.membership — membership-inference probe.

Question: given the released model, can you tell which molecules were in the
training data? Protocol (honest and simple):

  * members    = in-silo training molecules;
  * non-members = the global scaffold-held-out test set — trained under NO
    regime, so it is a clean non-member pool;
  * probe      = AUC of separating the model's score distributions
    (classification: decision-function scores; regression: |error|).

AUC ≈ 0.5 → no detectable leakage. AUC ≫ 0.5 → the model lets membership be
inferred; DP is then re-run and the probe repeated (before/after comparison).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from ..log import get_logger, section, summarize

log = get_logger("privacy.membership")


def probe_mi(member_scores: np.ndarray, nonmember_scores: np.ndarray,
             label: str = "model") -> dict:
    """AUC + bootstrap CI of member-vs-nonmember score separability."""
    y = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    scores = np.concatenate([member_scores, nonmember_scores])
    auc = float(roc_auc_score(y, scores))

    rng = np.random.default_rng(0)
    reps = 500
    boots = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), size=len(y))
        bs = roc_auc_score(y[idx], scores[idx])
        if np.isfinite(bs):
            boots.append(bs)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    reading = (
        "no detectable leakage (AUC≈0.5)"
        if abs(auc - 0.5) < 0.03
        else "weak signal" if auc < 0.6
        else "NOTABLE — membership is inferable"
    )
    section(log, f"membership-inference probe [{label}]")
    summarize(f"AUC (member vs non-member)", f"{auc:.3f} [95% CI {lo:.3f}–{hi:.3f}]", log)
    summarize("reading", reading.upper(), log)
    return {"auc": auc, "ci_lo": float(lo), "ci_hi": float(hi), "reading": reading}
