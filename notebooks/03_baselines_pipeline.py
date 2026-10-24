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
# 03 — Baselines: isolated & centralized-oracle

**Requires stage-3a metrics** (`uv run python scripts/run_models.py`).
This notebook reads `results/metrics/regime_*.json` — it does not train.

Stage 3a. Each partner trains alone on its own silo
(isolated, the status quo) and a pooled model trains on everything
(centralized-oracle, the privacy-impossible upper bound). Both get the same
total local work as the federated run so the later comparison is fair.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
**Run the canonical baselines** in the terminal (they log heavily by design):

```bash
uv run python scripts/run_models.py
```

This notebook then reads `results/metrics/regime_isolated.json` /
`regime_centralized.json` and shows the held-out numbers with CIs.
        """
    )
    return


@app.cell
def _(mo):
    import json

    from fedlearn.config import METRICS_DIR

    data = {}
    for name in ("regime_isolated.json", "regime_centralized.json"):
        with open(METRICS_DIR / name) as fh:
            data.update(json.load(fh))
    rows = []
    for regime, ep_metrics in data.items():
        for ep, m in ep_metrics.items():
            if "pr_auc" in m:
                rows.append({"regime": regime, "endpoint": ep, "metric": "PR-AUC",
                             "value": m["pr_auc"], "ci": f"[{m['pr_auc_lo']:.3f}–{m['pr_auc_hi']:.3f}]"})
            else:
                rows.append({"regime": regime, "endpoint": ep, "metric": "RMSE",
                             "value": m["rmse"], "ci": f"[{m['rmse_lo']:.3f}–{m['rmse_hi']:.3f}]"})
    import pandas as pd

    df = pd.DataFrame(rows)
    df["value"] = df["value"].round(3)
    df.pivot_table(index="endpoint", columns="regime", values="value", aggfunc="first")
    return


@app.cell
def _(mo):
    mo.md(
        """
**What to look for:** isolated numbers are MEAN-over-partners — the per-partner
spread is what matters (one partner's chemistry may predict Ames well, another
badly). The federated run (`04_federated_training.py`) should land between
these two tables. Centralized is your target to beat, not to match.
        """
    )
    return
