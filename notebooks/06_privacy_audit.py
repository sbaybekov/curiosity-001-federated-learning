import marimo as mo

app = mo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # consolidated imports — every name used below is defined here, once
    import json
    from pathlib import Path

    from fedlearn.config import RESULTS_DIR
    return RESULTS_DIR, Path, json


@app.cell
def _(mo):
    mo.md(
        """
# 06 — Privacy audit: membership inference before/after DP

**Requires stage-5 output** (`uv run python scripts/privacy_probe.py` →
writes `results/privacy_audit.json` AND a DP-trained model). This notebook
reads that file; the probe itself is the script (it needs a federated retrain
with DP, which is a pipeline action).

Stage 5. Protocol: take the released federated model, try
to tell training molecules apart from never-seen held-out molecules (AUC of
separability), then retrain the federation with DP ON (per-client L2 clipping
+ Gaussian noise) and re-probe. The before/after AUC delta is the honest
story: leakage shown, mitigation shown, both labeled.

**Canonical run:** `bash scripts/run_all.sh --start 5 --end 5` (or
`uv run python scripts/privacy_probe.py --endpoint solubility`).
        """
    )
    return


@app.cell
def _(RESULTS_DIR, Path, json):
    if Path(RESULTS_DIR / "privacy_audit.json").exists():
        with open(RESULTS_DIR / "privacy_audit.json") as fh:
            audit = json.load(fh)
        print(f"endpoint: {audit['endpoint']}")
        print(f"  plain model   MI-AUC = {audit['plain']['auc']:.3f} "
              f"[{audit['plain']['ci_lo']:.3f}–{audit['plain']['ci_hi']:.3f}] "
              f"→ {audit['plain']['reading']}")
        print(f"  with DP       MI-AUC = {audit['dp']['auc']:.3f} "
              f"[{audit['dp']['ci_lo']:.3f}–{audit['dp']['ci_hi']:.3f}] "
              f"→ {audit['dp']['reading']}")
        print(f"  Δ = {audit['delta']:+.3f} | verdict: {audit['verdict']}")
        audit
    else:
        print("no results/privacy_audit.json yet — run scripts/privacy_probe.py first")
    return


@app.cell
def _(mo):
    mo.md(
        """
**Reading the numbers honestly:** with fingerprint + linear models, a plain
score-based membership probe is often weak to begin with (AUC ≈ 0.5–0.6) —
that is a property of the model class, not a license to skip DP. The plan's
real deliverable is the *mechanism*: clip → noise → re-probe, with the DP
configuration (C, z, T → ε upper bound under basic composition) logged in the
terminal by `fedlearn.privacy.dp`.
        """
    )
    return


if __name__ == "__main__":
    app.run()