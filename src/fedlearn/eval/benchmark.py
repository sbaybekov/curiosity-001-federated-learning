"""fedlearn.eval.benchmark — the money plot & honest summary table.

Assembles the three-regime comparison (isolated / federated / centralized-
oracle) per endpoint: PR-AUC (cls) and RMSE (reg), each with bootstrap CIs,
plus the coverage-curve figure. Writes PNGs + a markdown summary into
results/ — the artifacts the README and walkthrough point at.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import ENDPOINTS, FIGURES_DIR, TASK
from ..log import get_logger, banner, section, summarize

log = get_logger("eval.benchmark")

REGIME_META = {
    "isolated":  {"color": "#c96442", "label": "Isolated (status quo)"},
    "federated": {"color": "#4a7a4a", "label": "Federated (FL)"},
    "centralized": {"color": "#4a6a9a", "label": "Centralized-oracle (upper bound)"},
}


def money_plot(regime_metrics: dict[str, dict], out_dir: Path | str = FIGURES_DIR) -> Path:
    """One panel per endpoint: PR-AUC or RMSE, 3 bars with CI error bars."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    banner(log, "MONEY PLOT — isolated vs federated vs centralized-oracle")
    expected = {"isolated", "federated", "centralized"}
    missing = expected - set(regime_metrics)
    if missing:
        log.warning(f"missing regime metrics: {sorted(missing)} — run the training "
                    f"stages first (scripts/run_models.py + run_federated.py)")
    n_endpoints = len(ENDPOINTS)
    fig, axes = plt.subplots(1, n_endpoints, figsize=(4.2 * n_endpoints, 3.8))
    if n_endpoints == 1:
        axes = [axes]
    for ax, endpoint in zip(axes, ENDPOINTS):
        task = TASK[endpoint]
        regimes = [r for r in ("isolated", "federated", "centralized")
                   if r in regime_metrics and endpoint in regime_metrics[r]]
        values, errs, labels, colors = [], [], [], []
        for regime in regimes:
            m = regime_metrics[regime][endpoint]
            if task == "cls":
                v, lo, hi = m["pr_auc"], m["pr_auc_lo"], m["pr_auc_hi"]
            else:
                v, lo, hi = m["rmse"], m["rmse_lo"], m["rmse_hi"]
            values.append(v)
            errs.append([v - lo, hi - v])
            labels.append(REGIME_META[regime]["label"].split(" (")[0])
            colors.append(REGIME_META[regime]["color"])
        ax.bar(range(len(values)), values, yerr=np.array(errs).T, capsize=5,
               color=colors, alpha=0.85)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f"{endpoint} · {'PR-AUC' if task == 'cls' else 'RMSE'}")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Federated vs isolated vs centralized — held-out (scaffold) performance",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = out_dir / "money_plot.png"
    fig.savefig(out, dpi=150)
    log.info(f"money plot → {out}")
    return out


def write_summary(regime_metrics: dict[str, dict], extra_sections: list[tuple[str, str]],
                  out: Path | str) -> Path:
    """Markdown summary with expectation-vs-result honesty bullets."""
    out = Path(out)
    lines = ["# FedMol — benchmark summary", ""]
    for task_name, task_eps in (("classification", ["ames", "herg"]),
                                ("regression", ["solubility", "logp"])):
        lines += [f"## {task_name}", ""]
        lines += ["| regime | endpoint | metric | value [95% CI] |", "|---|---|---|---|"]
        for endpoint in task_eps:
            for regime in ("isolated", "federated", "centralized"):
                if regime not in regime_metrics or endpoint not in regime_metrics[regime]:
                    continue
                m = regime_metrics[regime][endpoint]
                key = "pr_auc" if task_name == "classification" else "rmse"
                lines.append(f"| {regime} | {endpoint} | {key} | "
                             f"{m[key]:.3f} [{m[key + '_lo']:.3f}–{m[key + '_hi']:.3f}] |")
        lines.append("")
    lines += ["## Honest readings", ""]
    for title, body in extra_sections:
        lines += [f"### {title}", "", body, ""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    log.info(f"benchmark summary → {out}")
    return out
