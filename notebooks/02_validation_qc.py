import marimo as mo

app = mo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # consolidated imports — every name used below is defined here, once
    import pandas as pd

    from fedlearn.config import SPLIT_DIR
    from fedlearn.validators.qc import qc_report
    from fedlearn.validators.overlap import pairwise_overlap, site_digests
    from fedlearn.validators.schema import validate_schema
    return SPLIT_DIR, pd, pairwise_overlap, qc_report, site_digests, validate_schema


@app.cell
def _(mo):
    mo.md(
        """
# 02 — Validation & QC

**Requires stage-1 artifacts** (silo parquets from `download_data.py` —
this notebook validates what stage 1 built; it does not run the pipeline).

Stage 2 — the "validators under partial
visibility" story: schema checks return pass/fail verdicts + counts, QC
returns aggregates, cross-site overlap returns counts over hashed Morgan
digests. Nothing here ever prints a molecule row.
        """
    )
    return


@app.cell
def _(SPLIT_DIR, pd, qc_report, validate_schema):
    rows = []
    for ep in ("ames", "herg", "solubility", "logp"):
        df = pd.read_parquet(SPLIT_DIR / f"silo_0_{ep}.parquet")
        check = validate_schema(df, ep)
        qc = qc_report(df, ep)
        rows.append({
            "endpoint": ep,
            "schema_valid": check["valid"],
            "rows": qc["n_rows"],
            "exact_dups": qc["n_exact_duplicates"],
            "near_dups": qc["n_near_duplicates"],
            "mean_MW": round(qc.get("mean_mw", float("nan")), 1),
        })
    summary = pd.DataFrame(rows)
    summary
    return summary


@app.cell
def _(SPLIT_DIR, pd, pairwise_overlap, site_digests):
    # cross-site overlap on hashed digests — server sees counts, never structures
    digests = {}
    for partner in range(3):
        for endpoint in ("ames", "herg", "solubility", "logp"):
            frame = pd.read_parquet(SPLIT_DIR / f"silo_{partner}_{endpoint}.parquet")
            digests[f"silo{partner}.{endpoint}"] = site_digests(frame["smiles"])
    overlap = pairwise_overlap(digests)
    nonzero = overlap[overlap["shared_digests"] > 0]
    print(f"site pairs compared: {len(overlap)} | pairs with shared molecules: {len(nonzero)}")
    overlap.head(10)
    return


@app.cell
def _(mo):
    mo.md(
        """
**Reading the output:** every endpoint should show `schema_valid=True`;
`near_dups > 0` is not an error (public ADMET data has many close analogues —
Tanimoto > 0.9 pairs are flagged for the partner, not rejected). The overlap
table is the privacy-in-action step: identical molecules across "silos"
(potential licensing overlap) get flagged with counts only.

Run the canonical validation + pytest via:
`bash scripts/run_all.sh --start 2 --end 2` — next: `03_baselines_pipeline.py`.
        """
    )
    return


if __name__ == "__main__":
    app.run()