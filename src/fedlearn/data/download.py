"""fedlearn.data.download — pull the four public endpoints, normalize to one schema.

Sources (PyTDC, public): Ames (mutagenicity), hERG (cardiotoxicity) — binary;
Solubility (AqSolDB, logS), Lipophilicity (LogP) — regression. Each becomes a
canonical table with exactly three columns: ``unit_id``, ``smiles``
(canonicalized), ``y`` (numeric target). Download is skipped when a raw CSV
already exists, so re-runs are fast and offline-friendly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import MolToSmiles

from ..config import ENDPOINTS, RAW_DIR, TASK
from ..log import get_logger, section, timed

log = get_logger("data.download")

# PyTDC dataset names per endpoint key — keys are lowercase in pytdc 0.4.17+;
# candidates list also covers older spellings ('hERG', 'Ames', 'Lipophilicity').
_TDC_CANDIDATES = {
    "ames": ["ames", "Ames"],
    # karim = the canonical 12.6k hERG benchmark; plain 'herg' is a 655-row subset
    "herg": ["herg_karim", "herg", "hERG_Karim", "hERG"],
    "solubility": ["solubility_aqsoldb", "Solubility_AqSolDB", "AqSolDB"],
    "logp": ["lipophilicity_astrazeneca", "Lipophilicity", "logp"],
}
_TDC_GROUP = {"ames": "tox", "herg": "tox", "solubility": "adme", "logp": "adme"}


def canonicalize(smiles: str) -> str | None:
    """Canonical SMILES (isomeric, no stereo canon concerns) or None if invalid."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return MolToSmiles(mol)
    except Exception:  # noqa: BLE001
        return None


def _fetch_tdc(endpoint: str, log_=log) -> pd.DataFrame:
    """Return raw TDC rows (id, smiles, y) for one endpoint.

    Group mapping adapts to pytdc's catalog drift: ames/herg live in Tox,
    logp in ADME (class names ADMET→ADME across versions, dataset keys
    lowercase since 0.4.17) — try candidates until one resolves.
    """
    from tdc.single_pred import ADME, Tox

    group = Tox if _TDC_GROUP[endpoint] == "tox" else ADME
    data = None
    for name in _TDC_CANDIDATES[endpoint]:
        log_.info(f"PyTDC fetch: endpoint={endpoint} candidate={name}")
        try:
            data = group(name=name)
            break
        except ValueError:
            continue
    if data is None:
        raise RuntimeError(
            f"PyTDC has no resolvable dataset for '{endpoint}' "
            f"(tried {_TDC_CANDIDATES[endpoint]})"
        )
    with timed(log_, f"downloading {endpoint}"):
        frame = data.get_data()
    # PyTDC canonical columns: Drug_ID, Drug (SMILES), Y
    frame = frame.rename(columns={"Drug_ID": "unit_id", "Drug": "smiles", "Y": "y"})
    log_.info(f"  raw rows: {len(frame)}")
    return frame[["unit_id", "smiles", "y"]]


def normalize(frame: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    """Canonical schema + dedup, with loud counts for every drop."""
    df = frame.copy()
    before = len(df)
    df = df.dropna(subset=["smiles", "y"])
    df["smiles"] = df["smiles"].astype(str).map(canonicalize)
    df = df.dropna(subset=["smiles"])
    df = df.drop_duplicates(subset=["smiles"])
    df["unit_id"] = df["unit_id"].astype(str)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df = df.dropna(subset=["y"])
    df = df.reset_index(drop=True)
    log.info(f"  {endpoint}: {before} raw → {len(df)} usable "
             f"({before - len(df)} dropped: bad smiles/dup/NaN)")
    if TASK[endpoint] == "cls":
        counts = df["y"].value_counts().sort_index()
        log.info(f"  label balance: " + ", ".join(f"y={k}: {v}" for k, v in counts.items()))
    else:
        log.info(f"  y stats: mean={df['y'].mean():.3f} std={df['y'].std():.3f} "
                 f"min={df['y'].min():.3f} max={df['y'].max():.3f}")
    return df[["unit_id", "smiles", "y"]]


def download_all(raw_dir: Path | str = RAW_DIR) -> dict[str, Path]:
    """Ensure a normalized CSV per endpoint in data/raw; return path map."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    section(log, "endpoint download & normalization")
    paths: dict[str, Path] = {}
    for endpoint in ENDPOINTS:
        out = raw_dir / f"{endpoint}.csv"
        if out.exists():
            df = pd.read_csv(out)
            log.info(f"{endpoint}: cached CSV present — {len(df)} rows (skipping download)")
        else:
            try:
                df = _fetch_tdc(endpoint)
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"failed to fetch {endpoint}: {exc}\n"
                    f"  Check network, or drop your own CSV at {out} with columns "
                    f"unit_id,smiles,y (see download_all docstring)."
                ) from exc
            df = normalize(df, endpoint)
            df.to_csv(out, index=False)
            log.info(f"wrote {out} ({len(df)} rows)")
        paths[endpoint] = out
    return paths


def load_raw(endpoint: str, raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Read a previously normalized endpoint CSV."""
    path = Path(raw_dir) / f"{endpoint}.csv"
    if not path.exists():
        raise FileNotFoundError(f"raw data missing: {path} — run scripts/download_data.py first")
    return pd.read_csv(path)
