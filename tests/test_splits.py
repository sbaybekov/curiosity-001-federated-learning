"""Unit tests for the scaffold splitter — the leakage guarantee.

The core claim of the project: the held-out scaffold set shares NO scaffold
with training, and silos get non-IID chemistry by construction.
"""

from __future__ import annotations

import pandas as pd

from fedlearn.data.split import _scaffold_split_impl, assign_silos, murcko_key

# two aromatic families + one ringless: scaffolds are easy to reason about
_BENZENOIDS = ["c1ccccc1", "Cc1ccccc1", "c1ccc(cc1)C", "c1ccc(cc1)CO"]
_PYRIDINES = ["c1ccncc1", "Cc1ccncc1", "c1cc(ccn1)C"]
_NAPHTHALENES = ["c1ccc2ccccc2c1", "Cc1cc2cccc2c1"]
_ALKANES = ["CCC", "CCCC", "CCO"]


def _df():
    smiles = _BENZENOIDS + _PYRIDINES + _NAPHTHALENES + _ALKANES
    return pd.DataFrame({"unit_id": [f"m{i}" for i in range(len(smiles))],
                         "smiles": smiles, "y": [1, 0] * (len(smiles) // 2)})


def test_ringless_molecules_have_empty_scaffold():
    assert murcko_key("CCC") == ""


def test_heldout_shares_no_scaffold():
    train, hold = _scaffold_split_impl(_df(), frac=0.3, seed=42)
    train_scaffolds = set(train["smiles"].map(murcko_key))
    hold_scaffolds = set(hold["smiles"].map(murcko_key))
    assert train_scaffolds.isdisjoint(hold_scaffolds)


def test_silos_are_nonempty_and_disjoint():
    train, _ = _scaffold_split_impl(_df(), frac=0.0, seed=1)
    silos = assign_silos(train, n_partners=3, seed=7)
    counts = [silos[silos["partner"] == p].shape[0] for p in range(3)]
    assert all(c > 0 for c in counts)
    assert sum(counts) == len(train)


def test_silos_differ_in_scaffold_mix():
    train, _ = _scaffold_split_impl(_df(), frac=0.0, seed=1)
    silos = assign_silos(train, n_partners=3, seed=7)
    mixes = []
    for p in range(3):
        sub = silos[silos["partner"] == p]
        mixes.append(frozenset(sub["smiles"].map(murcko_key)))
    # snake-fill over ordered scaffolds: at least one silo pair differs
    assert len({m for m in mixes}) > 1
