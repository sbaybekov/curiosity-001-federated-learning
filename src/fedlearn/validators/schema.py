"""fedlearn.validators.schema — schema- and quality-gate validation.

Returns pass/fail verdicts + counts, never rows. A failing check logs
WARNING/ERROR and flips ``valid`` so downstream stages refuse to proceed —
bad data must fail loudly (and the unit tests enforce exactly that).
"""

from __future__ import annotations

import pandas as pd
from rdkit import Chem

from ..config import TASK
from ..log import get_logger, summarize

log = get_logger("validators.schema")

REQUIRED_COLUMNS = ("unit_id", "smiles", "y")


def validate_schema(df: pd.DataFrame, endpoint: str) -> dict:
    """Run every check; returns {'valid': bool, 'checks': {...}, 'counts': {...}}."""
    checks: dict[str, bool] = {}
    counts: dict[str, int] = {}

    # 1. required columns
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    checks["has_required_columns"] = not missing
    if missing:
        log.error(f"  schema ✗ missing columns: {missing}")
        verdict = all(checks.values())
        for check, ok in checks.items():
            summarize(f"schema[{check}]", "PASS" if ok else "FAIL", log)
        summarize("schema verdict", "VALID" if verdict else "INVALID", log)
        return {"valid": verdict, "checks": checks, "counts": counts}

    # 2. non-empty site
    checks["non_empty"] = len(df) > 0
    if not checks["non_empty"]:
        log.error("  schema ✗ empty site")

    # 3. missingness (aggregate fraction only)
    nulls = int(df["smiles"].isna().sum()) + int(df["y"].isna().sum())
    counts["null_cells"] = nulls
    checks["no_null_labels"] = nulls == 0
    if nulls:
        log.warning(f"  schema ! {nulls} null smiles/y values")

    # 4. SMILES parseability (fraction, not rows)
    n_parse = int(df["smiles"].map(lambda s: Chem.MolFromSmiles(str(s)) is not None).sum())
    counts["parseable"] = n_parse
    counts["total"] = len(df)
    checks["smiles_parseable"] = n_parse == len(df)
    if not checks["smiles_parseable"]:
        log.warning(f"  schema ! {len(df) - n_parse}/{len(df)} SMILES unparseable")

    # 5. label domain — cls: binary 0/1 | reg: finite numeric
    y = pd.to_numeric(df["y"], errors="coerce")
    if TASK[endpoint] == "cls":
        valid_domain = set(y.dropna().unique()) <= {0.0, 1.0}
        counts["label_values"] = sorted(y.dropna().unique().tolist())
        checks["label_domain"] = bool(valid_domain) and not y.isna().any()
    else:
        checks["label_domain"] = bool(y.notna().all() and y.abs().max() < 1e9)
    if not checks["label_domain"]:
        log.error(f"  schema ✗ labels outside {TASK[endpoint]} domain")

    # 6. unique ids
    checks["unique_ids"] = df["unit_id"].is_unique
    if not checks["unique_ids"]:
        log.warning(f"  schema ! {int(df['unit_id'].duplicated().sum())} duplicated ids")

    verdict = all(checks.values())
    for check, ok in checks.items():
        summarize(f"schema[{check}]", "PASS" if ok else "FAIL", log)
    summarize("schema verdict", "VALID" if verdict else "INVALID", log)
    return {"valid": verdict, "checks": checks, "counts": counts}


def site_profile(df: pd.DataFrame, endpoint: str) -> dict:
    """Aggregate statistics a partner is allowed to share pre-flight."""
    y = df["y"]
    profile = {
        "n_rows": int(len(df)),
        "n_unique_smiles": int(df["smiles"].nunique()),
        "y_mean": float(y.mean()),
        "y_std": float(y.std()),
    }
    if TASK[endpoint] == "cls":
        profile["class_balance"] = (
            y.value_counts(normalize=True).sort_index().round(4).to_dict()
        )
    else:
        profile["y_quantiles"] = y.quantile([0.0, 0.25, 0.5, 0.75, 1.0]).round(4).to_dict()
    return profile
