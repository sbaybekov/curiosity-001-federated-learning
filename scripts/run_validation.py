#!/usr/bin/env python
"""STAGE 2/5 — partner-side validation & QC + unit tests.

For every silo × endpoint: schema validation (loud pass/fail), QC report
(missingness, balance, descriptors, exact + near duplicates), then the
cross-site overlap check over hashed Morgan digests — the server learns
overlap counts without ever seeing a structure. Ends by running pytest.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import pandas as pd

from fedlearn.config import RESULTS_DIR, SPLIT_DIR, TASK, resolve_config
from fedlearn.data.split import load_splits
from fedlearn.log import banner, get_logger, section, stage, summarize
from fedlearn.provenance import Provenance, snapshot_state
from fedlearn.validators.overlap import site_digests, pairwise_overlap
from fedlearn.validators.qc import qc_report
from fedlearn.validators.schema import validate_schema

log = get_logger("stage2.validation")


def main() -> None:
    parser = argparse.ArgumentParser(description="stage 2 — validators & QC")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    config = resolve_config(args.config)
    n = config["silos"]["n_partners"]

    stage(log, 2, 5, "VALIDATION & QC — bad data must fail loudly, aggregates only")
    prove = Provenance("stage2-validation")
    prove.begin({"config": config})

    splits = load_splits(config)
    qc_results: dict = {}
    digests: dict[str, set] = {}
    all_valid = True

    for partner in range(n):
        section(log, f"partner {partner}")
        for endpoint, ep_split in splits.items():
            df = ep_split["silos"][partner]
            schema_verdict = validate_schema(df.reset_index(drop=True), endpoint)
            qc = qc_report(df.reset_index(drop=True), endpoint)
            qc_results.setdefault(f"partner{partner}", {})[endpoint] = {
                "schema": schema_verdict, "qc": qc}
            all_valid = all_valid and schema_verdict["valid"]

    section(log, "cross-site near-duplicate check (hashed Morgan digests, no structures)")
    for partner in range(n):
        for endpoint in TASK:
            df = pd.read_parquet(SPLIT_DIR / f"silo_{partner}_{endpoint}.parquet")
            digests[f"silo{partner}.{endpoint}"] = site_digests(df["smiles"])
    overlap = pairwise_overlap(digests)

    qc_report_path = RESULTS_DIR / "qc_report.json"
    qc_report_path.parent.mkdir(parents=True, exist_ok=True)
    qc_report_path.write_text(json.dumps({"qc": qc_results,
                                          "overlap": overlap.to_dict("records")},
                                         indent=2, default=str))
    log.info(f"qc report → {qc_report_path}")
    prove.record({"all_valid": bool(all_valid)}, artifacts=[qc_report_path])

    if not args.skip_tests:
        section(log, "pytest — unit tests for validators & splitter")
        res = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"])
        if res.returncode != 0:
            log.error("unit tests FAILED — fix before moving on")
            sys.exit(res.returncode)

    banner(log, "STAGE 2 COMPLETE — next: bash scripts/run_models.py")


if __name__ == "__main__":
    main()
