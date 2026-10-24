"""fedlearn.validators.overlap — cross-site overlap WITHOUT sharing structures.

Each site hashes its molecules' Morgan fingerprints (sha256 of the bit-vector
bytes) and ships only the digest set to the server. The server can then report
how many molecules are byte-identical across sites — while never seeing a
structure. This is the "server flags overlap without seeing structures" step
of the plan (t7).
"""

from __future__ import annotations

import hashlib

import pandas as pd
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

from ..log import get_logger, section, summarize

log = get_logger("validators.overlap")


def site_digests(smiles_list: pd.Series, nbits: int = 1024, radius: int = 2) -> set[str]:
    """Set of sha256 digests over Morgan bit-vectors — no structures exit the site."""
    digests: set[str] = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            continue
        bits = GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nbits)
        raw = bytes(bits.ToBitString(), "ascii")
        digests.add(hashlib.sha256(raw).hexdigest())
    return digests


def pairwise_overlap(digest_map: dict[str, set[str]], log_=log) -> pd.DataFrame:
    """Count + fraction of shared digests between every site pair."""
    sites = list(digest_map)
    rows = []
    for i in range(len(sites)):
        for j in range(i + 1, len(sites)):
            a, b = digest_map[sites[i]], digest_map[sites[j]]
            shared = len(a & b)
            rows.append({
                "site_a": sites[i],
                "site_b": sites[j],
                "shared_digests": shared,
                "fraction_of_a": round(shared / len(a), 4) if a else 0.0,
                "fraction_of_b": round(shared / len(b), 4) if b else 0.0,
            })
    frame = pd.DataFrame(rows)
    section(log_, "cross-site overlap (hashed digests only)")
    if frame.empty:
        log_.info("  no site pairs to compare")
        return frame
    for _, row in frame.iterrows():
        marker = "⚠ OVERLAP" if row["shared_digests"] else "✔ clean"
        summarize(f"{row['site_a']} ↔ {row['site_b']}",
                  f"{row['shared_digests']} shared ({marker})", log_)
    return frame
