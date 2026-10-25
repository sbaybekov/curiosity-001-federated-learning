import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # consolidated imports — every name used below is defined here, once
    import matplotlib.pyplot as plt
    import pandas as pd

    from fedlearn.config import FIGURES_DIR, SPLIT_DIR, TASK
    from fedlearn.data.split import murcko_key
    return FIGURES_DIR, SPLIT_DIR, TASK, murcko_key, pd, plt


@app.cell
def _(mo):
    mo.md("""
    # 01 — Data & silos

    **Requires stage-1 artifacts** (`bash scripts/run_all.sh --start 1 --end 1`
    or `uv run python scripts/download_data.py`). This notebook reads the silo
    parquets — it does not re-download anything.

    Stage 1: 4 public endpoints → one schema →
    scaffold-held-out test set → 3 non-IID partner silos. Every cell prints its
    own progress; the canonical run is `bash scripts/run_all.sh --start 1 --end 1`.
    """)
    return


@app.cell
def _(SPLIT_DIR, pd):
    splits = {}
    for ep in ("ames", "herg", "solubility", "logp"):
        splits[ep] = {
            "held_out": pd.read_parquet(SPLIT_DIR / f"heldout_{ep}.parquet"),
            "silos": [pd.read_parquet(SPLIT_DIR / f"silo_{p}_{ep}.parquet") for p in range(3)],
        }
    print("loaded splits for", list(splits))

    rows = []
    for ep, sp in splits.items():
        for p, df in enumerate(sp["silos"]):
            rows.append({"endpoint": ep, "split": f"silo{p}", "rows": len(df)})
        rows.append({"endpoint": ep, "split": "held-out", "rows": len(sp["held_out"])})
    df_summary = pd.DataFrame(rows).pivot(index="split", columns="endpoint", values="rows")
    df_summary
    return splits


@app.cell
def _(murcko_key, pd, splits):
    # scaffold overlap between train (all silos) and held-out — must be zero
    print("scaffold-overlap audit (train ∪ silos vs held-out):")
    for ep_name, sp_data in splits.items():
        train_all = pd.concat(sp_data["silos"])
        s_train = set(train_all["smiles"].map(murcko_key))
        s_hold = set(sp_data["held_out"]["smiles"].map(murcko_key))
        overlap = len(s_train & s_hold)
        status = "OK — disjoint" if overlap == 0 else f"LEAK: {overlap} shared scaffolds!"
        print(f"  {ep_name:>5}: train={len(s_train):>4} scaffolds, held-out={len(s_hold):>4}, "
              f"overlap={overlap:>3}  {status}")
    return


@app.cell
def _(FIGURES_DIR, TASK, mo, plt, splits):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, (ep_x, sp_x) in zip(axes.ravel(), splits.items()):
        sizes = [len(sp_x["silos"][p]) for p in range(3)]
        bars = ax.bar([f"partner {p}" for p in range(3)], sizes, color="#c96442")
        ax.set_title(f"{ep_x} — {'cls' if TASK[ep_x] == 'cls' else 'reg'}")
        ax.set_ylabel("training rows (held-out excluded)")
        for b, v in zip(bars, sizes):
            ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom")
    fig.suptitle("Silo sizes — non-IID assignment by scaffold", fontsize=13)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "01_silos.png", dpi=120)
    mo.mpl.interactive(fig)
    return


@app.cell
def _(mo):
    mo.md("""
    **What to look for:** the held-out scaffold audit prints `OK — disjoint` for
    every endpoint (that is the leakage guarantee, and stage-4 metrics depend on
    it). Silo sizes may differ noticeably — expected: scaffold-based assignment is
    non-IID by design. Next notebook → `02_validation_qc.py`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""

    """)
    return


if __name__ == "__main__":
    app.run()