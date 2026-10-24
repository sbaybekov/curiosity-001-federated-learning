# Federated learning for small-molecule property prediction

**curiosity_001** — a self-contained study comparing cross-silo federated learning
(FL) against classical training approaches for a cheminformatics problem:
predicting ADMET / toxicity-related properties from molecular structure.

The driving question isn't just *"can I train a model federated?"* — it's
*"when does FL actually win, where does it lose, and what makes it harder than
ordinary training?"* Everything here is public data and standard tooling, so the
whole thing is reproducible in a weekend on a laptop.

---

## Why this is an interesting question

Drug-discovery ADMET data is fragmented: each organization holds chemistry that
others cannot, by contract, ever see. Classical modelling assumes you can pool
everyone's data into one table. Federated learning rejects that assumption —
the data stays where it is, and only model updates travel. That trade (privacy
and breadth vs. engineering and training cost) is what this project measures.

## The comparison

Three training regimes, same public datasets:

| Regime | What happens to the data | What you get back |
|---|---|---|
| **Isolated** | Each partner trains only on its own silo | The status-quo baseline — strong on familiar chemistry, blind to everyone else's |
| **Federated (FL)** | Local training, only weight updates shared, FedAvg aggregation | Numbers indicate privacy-preserved breadth — reaches chemistry no partner could see alone |
| **Centralized-oracle** | All data pooled into one table | The theoretical upper bound — best possible quality, privacy impossible |

Endpoints are split evenly between task types:
- **Binary classification:** Ames, hERG (mutagenicity / hERG-cardiotoxicity)
- **Continuous regression:** ESOL (aqueous solubility), LogP (lipophilicity)

So the pipeline is exercised on both task types, two endpoints each.

## Models

**Primary — fingerprint + linear model.** ECFP/Morgan fingerprints feed
logistic regression (binary endpoints) and linear regression (regression
endpoints). Logistic regression is the canonical federated learner — FedAvg was
demonstrated on it — so this validates that federated learning works with
*classical* ML, not just neural networks. It trains in seconds, so the whole
regime comparison iterates fast.

**Optional stretch — ChemProp (GNN).** Kept out of the core scope for now, but
the harness is designed so a graph neural network can drop in later on a reduced
endpoint set (one binary + one regression) to test whether the FL conclusion
depends on model class. Add if time allows.

## Method

### Pipeline

```
                 Public ADMET data (Ames · hERG · ESOL · LogP)
                                   │
                                   ▼  scaffold split → never-touched held-out set
                         split remaining into 3–4 non-IID "partner" silos
                                    (distinct chemistry per site)
      ┌────────────┐   ┌────────────┐   ┌────────────┐        ┌────────┐
      │ Partner  A │   │ Partner  B │   │ Partner  C │  ...   │ Server │
      │ local QC ──┼──►│ local QC ──┼──►│ local QC   │        │ FedAvg │
      │ train model│   │ train model│   │ train model│◄───────┤(no data)│
      │  ↑ weights │   │  ↑ weights │   │  ↑ weights │        │        │
      └─────┬──────┘   └─────┬──────┘   └─────┬──────┘        └────────┘
            └─── only weight updates travel — raw molecules never leave
                                   │
                                   ▼
        Benchmark: isolated  vs  federated  vs  centralized-oracle
                                   │
                                   ▼
                Privacy probe: can we tell which molecules were
                used in training? (membership inference)
```

### What we measure

| Dimension | How | Realistic expectation |
|---|---|---|
| **Chemical-space coverage** | Accuracy vs. distance to nearest training molecule (Tanimoto), on off-silo scaffolds | FL broadens the applicability domain beyond isolated |
| **Performance** | PR-AUC / RMSE per endpoint, with confidence intervals | FL ≈ centralized > isolated, most on sparse endpoints |
| **Training time** | Wall-clock of full run; client compute vs. communication | FL is slower — the honest cost of privacy |

The thesis, stated plainly: *FL trades training wall-time for
privacy-preserving breadth* — it reaches chemistry a single partner could never
have access to.

---

## Repository layout

```
src/fedlearn/     importable package (data, splitting, validators, models, FL, eval, privacy)
notebooks/        marimo notebooks — the analysis entry point
scripts/          small CLI tools (dataset download, run FL)
data/             datasets (git-ignored)
results/          figures + model artifacts (git-ignored)
docs/             project notes (git-ignored)
```

## Getting started

Python 3.12 is pinned via `uv`; all dependencies live in a project-local
virtualenv.

```bash
uv sync                 # create .venv + install core deps
uv run marimo edit notebooks   # launch the analysis environment
```

> Notebook tooling: [marimo](https://marimo.io) — reactive Python notebooks,
> tracked as plain `.py` files so they diff cleanly in git.

## License

MIT. This is a learning project on public data — nothing here is proprietary.