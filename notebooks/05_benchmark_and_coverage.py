import marimo as mo

app = mo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
# 05 — Benchmark: the money plot + coverage

**Requires stages 3a+3b metrics** — `regime_isolated.json`,
`regime_centralized.json` (`run_models.py`) and `regime_federated.json`
(`run_federated.py`). This notebook renders those numbers; it does not train.

Stage 4. Three regimes on the never-touched scaffold-held-out set: isolated
(mean over partners, spread visible), federated, centralized-oracle — per
endpoint with bootstrap CIs, plus the chemical-space coverage comparison
(accuracy vs Tanimoto distance to the nearest training molecule).

**Canonical run:** `bash scripts/run_all.sh --start 4 --end 4` (needs stages
1–3 done). This notebook renders the same artifacts interactively.
        """
    )
    return


@app.cell
def _(mo):
    import json
    from pathlib import Path

    import pandas as pd

    from fedlearn.config import METRICS_DIR, RESULTS_DIR

    regime_metrics = {}
    for path in sorted(Path(METRICS_DIR).glob("regime_*.json")):
        with open(path) as fh:
            for regime, eps in json.load(fh).items():
                regime_metrics[regime] = eps
    rows = []
    for regime, eps in regime_metrics.items():
        for ep, m in eps.items():
            if "pr_auc" in m:
                rows.append({"regime": regime, "endpoint": ep, "metric": "PR-AUC",
                             "value": m["pr_auc"], "lo": m["pr_auc_lo"], "hi": m["pr_auc_hi"]})
            else:
                rows.append({"regime": regime, "endpoint": ep, "metric": "RMSE",
                             "value": m["rmse"], "lo": m["rmse_lo"], "hi": m["rmse_hi"]})
    df = pd.DataFrame(rows)
    df.pivot_table(index="endpoint", columns="regime", values="value").round(3)
    return df, regime_metrics


@app.cell
def _(mo, df):
    import matplotlib.pyplot as plt

    from fedlearn.config import TASK

    colors = {"isolated": "#c96442", "federated": "#4a7a4a", "centralized": "#4a6a9a"}
    eps = sorted(df["endpoint"].unique())
    fig, axes = plt.subplots(1, len(eps), figsize=(13, 3.8))
    for ax, ep in zip(axes, eps):
        sub = df[df["endpoint"] == ep]
        x = range(len(sub))
        ax.bar(list(x), sub["value"], yerr=[sub["value"] - sub["lo"],
                                            sub["hi"] - sub["value"]],
               capsize=5, color=[colors[r] for r in sub["regime"]])
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["regime"])
        ax.set_title(f"{ep} ({'PR-AUC' if TASK[ep] == 'cls' else 'RMSE'}) · lower RMSE better")
    fig.suptitle("isolated vs federated vs centralized-oracle (held-out)", fontsize=12)
    plt.tight_layout()
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo):
    mo.md(
        """
**The three dimensions to read off:**

1. **Performance** — does FL land between isolated and centralized?
2. **Coverage** — on chemistry FAR from a partner's own set (off-silo scaffolds),
   does the FL model beat the isolated partners? (`results/coverage.csv`)
3. **Training time** — compare `results/runs/fed_history.jsonl` (per-round
   wall time) against the isolated/centralized runs' wall time in the logs:
   FL trades wall-time for breadth; say so honestly in the walkthrough.
        """
    )
    return
