"""fedlearn.eval.metrics — task metrics with bootstrap confidence intervals.

PR-AUC (classification, the right curve for imbalanced ADMET endpoints) and
RMSE (regression), each with a stratified bootstrap 95% CI. Progress of the
bootstrap is logged — long tails are visible, not silent.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from ..log import get_logger

log = get_logger("eval.metrics")


def pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, scores))


def roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, scores))


def rmse(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - pred) ** 2)))


def r2(y_true: np.ndarray, pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def bootstrap_ci(y_true: np.ndarray, scores: np.ndarray, metric: str,
                 reps: int = 1000, seed: int = 0, verbose: bool = True) -> tuple[float, float, float]:
    """(point, lo, hi) with the same metric as the point; stratified for cls."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 20:
        log.warning(f"  too few samples ({n}) for a stable bootstrap CI")
    if metric == "pr_auc":
        fn = pr_auc
    elif metric == "roc_auc":
        fn = roc_auc
    elif metric == "rmse":
        fn = rmse
    else:
        fn = r2

    point = fn(y_true, scores)
    classes = np.unique(y_true)
    y = np.asarray(y_true)
    s = np.asarray(scores)
    boots = []
    for rep in range(reps):
        idx = np.concatenate([rng.choice(np.where(y == c)[0], size=(y == c).sum(), replace=True)
                              for c in classes]) if len(classes) <= 2 and metric in (
            "pr_auc", "roc_auc") else rng.integers(0, n, size=n)
        try:
            boots.append(fn(y[idx], s[idx]))
        except ValueError:
            continue
        if verbose and (rep + 1) % 250 == 0:
            log.info(f"    bootstrap {rep + 1}/{reps}")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def evaluate_classifier(y_true, scores, reps=1000, log_=log) -> dict:
    """Full classifier summary: PR-AUC + ROC-AUC with CIs, logged."""
    pr, pr_lo, pr_hi = bootstrap_ci(y_true, scores, "pr_auc", reps)
    roc, roc_lo, roc_hi = bootstrap_ci(y_true, scores, "roc_auc", reps)
    log_.info(f"  cls: PR-AUC {pr:.3f} [{pr_lo:.3f}–{pr_hi:.3f}] | "
              f"ROC-AUC {roc:.3f} [{roc_lo:.3f}–{roc_hi:.3f}] | n={len(y_true)}")
    return {"pr_auc": pr, "pr_auc_lo": pr_lo, "pr_auc_hi": pr_hi,
            "roc_auc": roc, "roc_auc_lo": roc_lo, "roc_auc_hi": roc_hi}


def evaluate_regression(y_true, pred, reps=1000, log_=log) -> dict:
    rm, rm_lo, rm_hi = bootstrap_ci(y_true, pred, "rmse", reps)
    rr2 = r2(y_true, pred)
    log_.info(f"  reg: RMSE {rm:.3f} [{rm_lo:.3f}–{rm_hi:.3f}] | R² {rr2:.3f} | n={len(y_true)}")
    return {"rmse": rm, "rmse_lo": rm_lo, "rmse_hi": rm_hi, "r2": rr2}
