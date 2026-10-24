# FedMol — Runbook

Everything the plan (`docs/fedmol_plan_report.html`) asks for, mapped to
commands. Logs are the UI: every stage prints a `STAGE n/5` banner and a
trail of `· key = value` lines, so you always know where the run is and what
happened last.

## 0. Environment (plan t1)

```bash
bash scripts/setup_env.sh          # uv sync + a loud import/version check of every dep
```

Expected: each dependency printed with `✔ <name> <version>`. First run pulls
~1–2 GB of wheels (rdkit + ray via flwr[simulation]) — normal.

```bash
uv run marimo edit notebooks       # interactive analysis layer (see §7)
```

## 1. Data & silos (plan t2–t4)

```bash
uv run python scripts/download_data.py
```

Expected log trail:

```
STAGE 1/5 — DATA & SILOS ...
  PyTDC fetch: endpoint=ames dataset=Ames
  ames: 6512 raw → 6041 usable (471 dropped: bad smiles/dup/NaN)
    label balance: y=0: 3856, y=1: 2185
  scaffolds total=… held-out-scaffolds=…
  ✔ guarantee: no scaffold appears in both train and held-out
  partner 0: rows=… scaffolds=… | bal=…
```

**Caveat — TDC network:** if the fetch fails you get a clear `RuntimeError`
telling you to retry or drop `data/raw/<ep>.csv` yourself with three columns
`unit_id,smiles,y` (SMILES canonicalized by the loader). Re-runs skip
download when the CSV exists.

Outputs: `data/raw/*.csv`, `data/processed/splits/silo_{p}_{ep}.parquet`,
`heldout_{ep}.parquet`, `splits.json` (counts manifest).

## 2. Validators & QC (plan t5–t8)

```bash
uv run python scripts/run_validation.py
```

Expected: per-silo schema verdicts (`PASS`/`FAIL` per check), QC aggregates,
the hashed-digest overlap table, then `pytest`. **Bad schema, empty site, bad
labels, duplicates → loud FAIL** (tests assert it: `tests/test_validators.py`,
`tests/test_splits.py`).

## 3a. Baselines (plan t9–t10)

```bash
uv run python scripts/run_models.py
```

Isolated (per-partner, same total local work as FL) and centralized-oracle
(pooled). Held-out PR-AUC/RMSE with bootstrap CIs per endpoint → `results/metrics/regime_isolated.json`, `regime_centralized.json`, model states in `results/models/`.

## 3b. Federated loop (plan t11)

```bash
uv run python scripts/run_federated.py            # Flower, auto-fallback
uv run python scripts/run_federated.py --backend manual   # force reference loop
```

Expected log trail:

```
FEDERATED LOOP — 20 rounds × 3 partners
  DP=OFF (clip C=1.0, z=1.1)
  backend: FLOWER 1.x — in-process simulation, 3 virtual clients, FedAvg strategy
  round  15/20 | local loss: ames=… herg=… solubility=… logp=… | ‖Δ‖₂=… | 3.2s
```

If flwr/ray misbehaves it logs `Flower path failed (…) — switching to manual
FedAvg loop` and keeps going with identical math — that is by design, not a
bug. Output: `results/models/federated.npz`, `results/metrics/regime_federated.json`,
`results/runs/fed_history.jsonl` (per-round time → the timing dimension).

## 4. Benchmark & coverage (plan t13, t21, t22)

```bash
uv run python scripts/benchmark.py            # full run
uv run python scripts/benchmark.py --skip-coverage   # metrics + plot only (faster)
```

Outputs `results/figures/money_plot.png`, `results/coverage.csv`,
`results/benchmark_summary.md`. The far-bin accuracy comparison (chemistry
0.30–0.50 Tanimoto-far from a partner's own set) is the "FL broadens the
applicability domain" number.

## 5. Privacy audit (plan t15–t16)

```bash
uv run python scripts/privacy_probe.py --endpoint ames    # default
uv run python scripts/privacy_probe.py --endpoint solubility
```

Protocol: probe plain federated model (members = silo molecules, non-members
= scaffold held-out) → retrain with DP ON (config `privacy.*`) → re-probe.
`results/privacy_audit.json` has before/after AUC + ε estimate (labeled an
upper bound under basic composition — tight RDP accounting is a listed
stretch, don't overclaim ε).

## One-shot pipeline (everything in one command)

```bash
bash scripts/run_all.sh               # stages 1→5, stop on first error
bash scripts/run_all.sh --start 3     # resume where you left off
```

## 6. Troubleshooting map

| Symptom | Cause → fix |
|---|---|
| `TDC fetch failed` (stage 1) | Network/proxy — retry; or supply your own CSVs in `data/raw` |
| Flower falls back to manual | flwr version API drift or ray hiccup — **fine**, math identical; check `flwr.__version__` in the log; optionally `uv pip install --upgrade flwr` |
| Ray memory warnings | Normal for in-process simulation on small data; ignore unless OOM |
| `MLflow init failed … JSONL` | mlflow optional; run manifest still recorded in `results/manifests/run_manifest.jsonl` |
| Bootstrap CI slow (1000 reps) | Expected on laptops; reduce `eval.bootstrap_reps` in `configs/config.yaml` |
| pytest fails | Tighten the data path: re-run stage 1 (silos must be disjoint) |

## 7. Notebooks vs scripts — who does what

**Scripts produce artifacts, notebooks explore them — run each script once,
then go notebook-first for the analysis.** No doubling: notebooks read files
from disk; only notebook 04 can also *train* (run-button cell), and it writes
the exact same files `scripts/run_federated.py` writes — pick one path, not both.

| Notebook | Reads (needs from stage) | Satisfy with (one command) |
|---|---|---|
| `01_data_silos.py` | silo/held-out parquets (1) | `bash scripts/run_all.sh --start 1 --end 1` |
| `02_validation_qc.py` | silo parquets (1) | same as above |
| `03_baselines_pipeline.py` | `regime_isolated/centralized.json` (3a) | `uv run python scripts/run_models.py` |
| `04_federated_training.py` | `federated.npz`, `fed_history.jsonl` (3b) — or trains itself | `uv run python scripts/run_federated.py` |
| `05_benchmark_and_coverage.py` | all three `regime_*.json` (3a+3b) | `run_models.py` + `run_federated.py` |
| `06_privacy_audit.py` | `privacy_audit.json` (5) | `uv run python scripts/privacy_probe.py` |

All cells read from disk (no hidden state between cells), so they re-run
standalone. Plot artifacts written by notebooks land next to the script
outputs in `results/figures/`.

## 8. What's intentionally NOT here yet (deferred per plan)

- ChemProp / GNN drop-in (t14, stretch): the harness would accept a new head
  implementing `get_parameters/set_parameters` — `models/heads.py`.
- Tight RDP privacy accounting (t16 is clip+noise + honest upper bound).
- Hydra experiment matrices (configs are hydra-layout; `hydra-core` is
  installed for a later sweep; scripts currently load `configs/config.yaml`
  directly via `fedlearn.config.resolve_config`).

Everything a run produces is recorded through `fedlearn.provenance` — MLflow
(`results/mlruns`) or the JSONL fallback — including data hashes and the git
revision, per `docs/fedmol_plan_report.html` t12.
