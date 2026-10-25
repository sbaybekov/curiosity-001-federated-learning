"""Unit tests for the validation chain — bad data must fail LOUDLY.

Run with:  uv run pytest tests -q
"""

from __future__ import annotations

import pandas as pd
import pytest

from fedlearn.validators.overlap import pairwise_overlap, site_digests
from fedlearn.validators.qc import near_duplicate_count
from fedlearn.validators.schema import validate_schema


def _df(smiles=("c1ccccc1", "c1ccc(cc1)C"), y=(1, 0)):
    return pd.DataFrame({"unit_id": [f"m{i}" for i in range(len(smiles))],
                         "smiles": list(smiles), "y": list(y)})


def test_happy_path_valid():
    assert validate_schema(_df(), "ames")["valid"] is True


def test_missing_column_fails():
    df = _df().drop(columns=["smiles"])
    result = validate_schema(df, "ames")
    assert result["valid"] is False
    assert result["checks"]["has_required_columns"] is False


def test_empty_site_fails():
    df = _df(smiles=(), y=())
    result = validate_schema(df, "ames")
    assert result["valid"] is False


def test_unparseable_smiles_fails():
    df = _df(smiles=("c1ccccc1", "not-a-molecule"))
    result = validate_schema(df, "ames")
    assert result["checks"]["smiles_parseable"] is False
    assert result["counts"]["parseable"] == 1


def test_label_domain_violation_fails():
    df = _df(y=(1, 2))  # Ames labels must be 0/1
    result = validate_schema(df, "ames")
    assert result["checks"]["label_domain"] is False


def test_duplicate_ids_flagged():
    df = _df()
    df.loc[1, "unit_id"] = "m0"
    result = validate_schema(df, "ames")
    assert result["checks"]["unique_ids"] is False


def test_near_duplicate_detected():
    # stereoisomers are 2D-identical → Morgan Tanimoto = 1.0 (flagged as near-dup)
    dup, checked = near_duplicate_count(pd.Series(["C/C=C/C", "C/C=C\\C", "CCO"]))
    assert checked > 0
    assert dup >= 1


def test_distinct_molecules_not_flagged_as_near():
    # a methyl group drops Morgan Tanimoto to ~0.27 — genuinely distinct
    dup, checked = near_duplicate_count(pd.Series(["c1ccccc1", "Cc1ccccc1"]))
    assert checked > 0
    assert dup == 0


def test_overlap_counts_shared_digests():
    # identical canonical molecule must produce an identical digest across sites
    benz = "c1ccccc1"
    da = {"A": site_digests(pd.Series([benz, "CCO"]))}
    db = {"B": site_digests(pd.Series([benz, "CC(C)=O"]))}
    table = pairwise_overlap({**da, **db})
    shared = table.loc[(table["site_a"] == "A") & (table["site_b"] == "B"),
                       "shared_digests"].iloc[0]
    assert shared == 1


def test_canonical_smiles_gives_same_digest():
    # same molecule, different SMILES spellings → same digest
    da = site_digests(pd.Series(["c1ccccc1"]))
    db = site_digests(pd.Series(["C1=CC=CC=C1"]))
    assert da == db
