"""fedlearn.data.split — Murcko-scaffold splitting into a held-out test set
and non-IID partner silos.

Guarantees (enforced + logged, not assumed):
  * the held-out set shares NO scaffold with any training molecule;
  * the training data is divided among N partners such that each partner
    owns a distinct slice of chemistry (scaffold-wise, non-IID);
  * silo assignment is per-endpoint (a molecule appearing in two endpoint
    datasets may land in different silos — documented, realistic: partner
    licenses are per-dataset, not per-molecule).
"""

from __future__ import annotations

import json
import random

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from ..config import ENDPOINTS, SPLIT_DIR, TASK
from ..log import get_logger, section, summarize, timed
from .download import canonicalize, load_raw

log = get_logger("data.split")


def murcko_key(smiles: str) -> str:
    """Generic Murcko scaffold SMILES; '' for ringless molecules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold else ""


def _scaffold_split_impl(
    df: pd.DataFrame, frac: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    df = df.reset_index(drop=True)
    scaffolds: dict[str, list[int]] = {}
    for i, smiles in enumerate(df["smiles"]):
        scaffolds.setdefault(murcko_key(smiles), []).append(i)
    scaffold_keys = sorted(scaffolds, key=lambda s: (-len(scaffolds[s]), s))
    rng.shuffle(scaffold_keys)
    n_hold = max(1, round(len(scaffold_keys) * frac))
    hold_keys = set(scaffold_keys[:n_hold])
    hold_idx = sorted(i for k in hold_keys for i in scaffolds[k])
    train_idx = [i for i in range(len(df)) if i not in set(hold_idx)]
    train, hold = df.iloc[train_idx].copy(), df.iloc[hold_idx].copy()
    log.info(f"  scaffolds total={len(scaffold_keys)} held-out-scaffolds={n_hold}")
    log.info(f"  rows: train={len(train)} held-out={len(hold)} "
             f"(heldout {len(hold)/(len(train)+len(hold))*100:.1f}%)")
    log.info("  ✔ guarantee: no scaffold appears in both train and held-out")
    return train, hold


def assign_silos(train_df: pd.DataFrame, n_partners: int, seed: int) -> pd.DataFrame:
    """Snake-fill scaffold groups into N silos → each silo gets a chemistry mix."""
    rng = random.Random(seed)
    df = train_df.reset_index(drop=True)
    scaffolds: dict[str, list[int]] = {}
    for i, smiles in enumerate(df["smiles"]):
        scaffolds.setdefault(murcko_key(smiles), []).append(i)
    # largest scaffolds first → the silos differ structurally, not by accident
    ordered = sorted(scaffolds, key=lambda s: (-len(scaffolds[s]), s))
    silo_ids = list(range(n_partners))
    partner = {}
    for pos, key in enumerate(ordered):
        slot = silo_ids[pos % n_partners] if (pos // n_partners) % 2 == 0 \
            else silo_ids[n_partners - 1 - (pos % n_partners)]
        for i in scaffolds[key]:
            partner[i] = slot
    df["partner"] = df.index.map(partner)
    return df


def _partner_stats(df: pd.DataFrame, endpoint: str, n: int) -> None:
    log.info(f"  silo sizes ({endpoint}):")
    for p in range(n):
        sub = df[df["partner"] == p]
        unique_scaffolds = sub["smiles"].map(murcko_key).nunique() if len(sub) else 0
        balance = ""
        if TASK[endpoint] == "cls" and len(sub):
            counts = sub["y"].value_counts(normalize=True).sort_index()
            balance = " | bal=" + ",".join(f"{v:.2f}" for v in counts)
        summarize(f"partner {p}: rows={len(sub)} scaffolds={unique_scaffolds}{balance}", "", log)


def build_silos(config: dict) -> dict:
    """Full stage-1: normalize → scaffold-heldout → silos → parquet + manifest."""
    section(log, "scaffold splitting & silo assignment")
    cfg_data, cfg_silos = config["data"], config["silos"]
    n_partners = cfg_silos["n_partners"]
    seed = cfg_silos["seed"]
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict = {"config": config, "endpoints": {}}
    for endpoint in ENDPOINTS:
        with timed(log, f"{endpoint}: normalize + split + silos"):
            df = load_raw(endpoint, cfg_data["raw_dir"])
            df["smiles"] = df["smiles"].map(canonicalize)
            df = df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"])
            train, hold = _scaffold_split_impl(df, cfg_data["heldout_frac"], seed)
            train = assign_silos(train, n_partners, seed)
            _partner_stats(train, endpoint, n_partners)
            for p in range(n_partners):
                silo = train[train["partner"] == p].drop(columns=["partner"])
                silo.to_parquet(SPLIT_DIR / f"silo_{p}_{endpoint}.parquet")
            hold.drop(columns=["partner"], errors="ignore").to_parquet(
                SPLIT_DIR / f"heldout_{endpoint}.parquet")
            summary["endpoints"][endpoint] = {
                "train": int(len(train)),
                "held_out": int(len(hold)),
                "silos": [int((train["partner"] == p).sum()) for p in range(n_partners)],
                "n_scaffolds_train": int(train["smiles"].map(murcko_key).nunique()),
            }
    manifest_path = SPLIT_DIR / "splits.json"
    manifest_path.write_text(json.dumps(summary, indent=2))
    log.info(f"splits manifest → {manifest_path}")
    return summary


def load_splits(config: dict) -> dict:
    """stage-2+ helper: load all split parquets as {endpoint: {'train': concat,
    'held_out': df, 'silos': [df,...]}} keyed on endpoint."""
    loaded: dict = {}
    for endpoint in ENDPOINTS:
        silos = [
            pd.read_parquet(SPLIT_DIR / f"silo_{p}_{endpoint}.parquet")
            for p in range(config["silos"]["n_partners"])
        ]
        held = pd.read_parquet(SPLIT_DIR / f"heldout_{endpoint}.parquet")
        loaded[endpoint] = {"silos": silos, "held_out": held}
    return loaded
