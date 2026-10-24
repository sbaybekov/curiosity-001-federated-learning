"""fedlearn.data.featurize — Morgan/ECFP fingerprints.

Featurization is the boundary where molecules become numbers; everything else
in the pipeline operates on the returned float32 matrix. There is no cache —
partitions are small (≤12k rows) and featurizing takes seconds; add one only
if the stretch GNN makes it slow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

from ..log import get_logger, section

log = get_logger("data.featurize")


def morgan_bits(smiles: str, radius: int = 2, nbits: int = 2048) -> np.ndarray | None:
    """Morgan (ECFP-like) fingerprint as a dense bit vector, or None if invalid."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return np.array(
        GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits), dtype=np.float32
    )


def featurize_df(
    df: pd.DataFrame,
    radius: int = 2,
    nbits: int = 2048,
    smiles_col: str = "smiles",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """X (N×nbits float32), y, ids — plus how many SMILES failed to parse."""
    X_list, y_list, id_list, bad = [], [], [], 0
    for unit_id, smiles, y in zip(df.index, df[smiles_col], df["y"]):
        bits = morgan_bits(smiles, radius, nbits)
        if bits is None:
            bad += 1
            continue
        X_list.append(bits)
        y_list.append(y)
        id_list.append(unit_id)
    X = np.asarray(X_list, dtype=np.float32) if X_list else np.zeros((0, nbits), dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    ids = np.asarray(id_list, dtype=object)
    if bad:
        log.warning(f"  {bad} SMILES unparseable and dropped")
    return X, y, ids, bad


if __name__ == "__main__":  # one runnable check (ponytail: per-module demo)
    section(log, "featurizer smoke test")
    samples = pd.DataFrame({
        "smiles": ["c1ccccc1", "c1ccc(cc1)Cc1ccccc1", "not-a-molecule"],
        "y": [1.0, 0.0, 1.0],
    })
    X, y, ids, bad = featurize_df(samples)
    assert X.shape == (2, 2048) and bad == 1, "featurizer misbehaved"
    print("OK — X:", X.shape, "| bad:", bad)
