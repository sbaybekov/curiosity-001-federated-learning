"""fedlearn.validators.qc — per-partner quality-control report.

Aggregates only: missingness, label balance, descriptor means, exact- and
near-duplicate counts. Near-duplicate detection (Tanimoto > 0.9 on Morgan
bits) is capped to a sample for O(n²) safety; any cap is logged explicitly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit.Chem import Crippen, Descriptors
from rdkit.Chem import DataStructs
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit import Chem

from ..config import TASK
from ..log import get_logger, section, summarize, timed
from .schema import site_profile

log = get_logger("validators.qc")

SAMPLE_CAP = 2000  # molecules used for the O(n²) near-dup check
TANIMOTO_CUTOFF = 0.9


def _fingerprints(smiles_list, nbits=1024, radius=2):
    fps = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        fps.append(GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits))
    return fps


def descriptor_stats(smiles_list: pd.Series, cap: int = 500) -> dict:
    """Mean MW / LogP over a capped sample (fast, logged as 'approx')."""
    sample = smiles_list.head(cap)
    mw, logp = [], []
    for smiles in sample:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        mw.append(Descriptors.MolWt(mol))
        logp.append(Crippen.MolLogP(mol))
    return {
        "descriptor_sample_n": len(mw),
        "mean_mw": float(np.mean(mw)) if mw else None,
        "mean_logp": float(np.mean(logp)) if logp else None,
    }


def near_duplicate_count(smiles_list: pd.Series, seed: int = 0) -> tuple[int, int]:
    """Count pairs with Tanimoto > 0.9 on a sampled cap; returns (n_dup_pairs, n_checked)."""
    fps = _fingerprints(smiles_list.head(SAMPLE_CAP).tolist())
    if not fps:
        return 0, 0
    checked = 0
    dup_pairs = 0
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            checked += 1
            if DataStructs.TanimotoSimilarity(fps[i], fps[j]) > TANIMOTO_CUTOFF:
                dup_pairs += 1
    return dup_pairs, checked


def qc_report(df: pd.DataFrame, endpoint: str, log_=log) -> dict:
    """Full QC bundle for one partner site; json-serializable."""
    section(log_, f"QC report — endpoint '{endpoint}'")
    report: dict = {"endpoint": endpoint, "valid": True}
    with timed(log_, f"qc {endpoint}"):
        profile = site_profile(df, endpoint)
        report.update(profile)
        for label, value in profile.items():
            summarize(label, value, log_)

        dup_exact = int(df["smiles"].duplicated().sum())
        report["n_exact_duplicates"] = dup_exact
        if dup_exact:
            log.warning(f"  {dup_exact} exact duplicate SMILES in site")
        summarize("exact duplicates", dup_exact, log_)

        if len(df):
            desc = descriptor_stats(df["smiles"])
            report.update(desc)
            summarize(f"descriptors (sample s={desc['descriptor_sample_n']})",
                      f"MW={desc['mean_mw']:.1f} LogP={desc['mean_logp']:.2f}", log_)

            dup_near, checked = near_duplicate_count(df["smiles"])
            report["n_near_duplicates"] = dup_near
            report["near_dup_pairs_checked"] = checked
            if checked < len(df) * (len(df) - 1) / 2:
                log.info(f"  near-dup check sampled to {SAMPLE_CAP} rows ({checked} pairs) "
                         f"— full check would be O(n²)")
            if dup_near:
                log.warning(f"  {dup_near} near-duplicate pairs (Tanimoto > {TANIMOTO_CUTOFF})")
            summarize("near-duplicate pairs (Tanimoto>0.9)", dup_near, log_)

        if TASK[endpoint] == "cls":
            pos = float(profile.get("class_balance", {}).get(1.0, 0.0))
            if pos < 0.02 or pos > 0.98:
                log.warning("  extreme class imbalance — flag for the partner")
    report["valid"] = report["n_exact_duplicates"] == 0
    summarize("qc verdict", "PASS" if report["valid"] else "review flagged", log_)
    return report
