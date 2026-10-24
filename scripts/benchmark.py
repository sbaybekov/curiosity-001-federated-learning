#!/usr/bin/env python
"""STAGE 4/5 — benchmark: money plot, coverage analysis, summary table.

Pulls every results/metrics/regime_*.json, renders the money plot
(PR-AUC/RMSE + CI per endpoint: isolated / federated / centralized), runs the
chemical-space coverage comparison, and writes results/benchmark_summary.md
with honest expectation-vs-result readings.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from fedlearn.config import FIGURES_DIR, METRICS_DIR, MODELS_DIR, RESULTS_DIR, resolve_config
from fedlearn.data.partitions import load_partitions
from fedlearn.eval.benchmark import money_plot, write_summary
from fedlearn.eval.coverage import coverage_comparison
from fedlearn.log import banner, get_logger, section, stage, summarize
from fedlearn.provenance import Provenance

log = get_logger("stage4.benchmark")


def _load_regime_metrics() -> dict[str, dict]:
    """All regime_*.json → {regime: {endpoint: metrics}}."""
    regime_metrics: dict[str, dict] = {}
    for path in sorted(METRICS_DIR.glob("regime_*.json")):
        with open(path) as fh:
            data = json.load(fh)
        for regime, endpoints_metrics in data.items():
            regime_metrics[regime] = endpoints_metrics
    summarize("regimes found", sorted(regime_metrics), log)
    return regime_metrics


def _models_for_coverage(config) -> dict:
    """{'isolated': {cid: state}, 'federated': state, 'centralized': state}."""
    n = config["silos"]["n_partners"]
    isolated = {f"p{p}": np.load(MODELS_DIR / f"isolated_p{p}.npz")["state"]
                for p in range(n)}
    return {
        "isolated": isolated,
        "federated": np.load(MODELS_DIR / "federated.npz")["state"],
        "centralized": np.load(MODELS_DIR / "centralized.npz")["state"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="stage 4 — benchmark & coverage")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-coverage", action="store_true")
    args = parser.parse_args()
    config = resolve_config(args.config)
    stage(log, 4, 5, "BENCHMARK — money plot, coverage, summary")
    prove = Provenance("stage4-benchmark")
    prove.begin({"config": config})

    regime_metrics = _load_regime_metrics()
    fig = money_plot(regime_metrics, FIGURES_DIR)

    extra: list[tuple[str, str]] = []
    if not args.skip_coverage:
        section(log, "coverage — accuracy vs distance to nearest training molecule")
        parts = load_partitions(config)
        models = _models_for_coverage(config)
        cov = coverage_comparison(config, parts, models)
        cov.to_csv(RESULTS_DIR / "coverage.csv", index=False)
        log.info(f"coverage table → {RESULTS_DIR / 'coverage.csv'}")
        # headline: the far bins (chemistry unfamiliar to a partner)
        far = cov[cov["sim_bin"].isin(["0.50–0.60", "0.60–0.70"])]
        if len(far):
            iso = far[far["regime"] == "isolated"]["accuracy"].mean()
            fed = far[far["regime"] == "federated"]["accuracy"].mean()
            extra.append((
                "Coverage (applicability domain)",
                f"On molecules 0.30–0.50 Tanimoto-far from a partner's own "
                f"chemistry, isolated partners average {iso:.3f} accuracy vs "
                f"federated {fed:.3f} — "
                f"{'FL broadens the domain' if fed >= iso else 'no clear FL edge here'}.",
            ))
    else:
        log.info("coverage SKIPPED (--skip-coverage)")

    extra.append((
        "Expectation check",
        "Federated should land between isolated and centralized-oracle, with the "
        "largest gain on the sparsest endpoint (LogP, ~4.2k rows). If isolated matches or "
        "beats FL everywhere, reframe: FL matched it while no partner ever saw "
        "the others' chemistry.",
    ))
    write_summary(regime_metrics, extra, RESULTS_DIR / "benchmark_summary.md")
    prove.record({"figures": [str(fig)]}, artifacts=[
        fig, RESULTS_DIR / "benchmark_summary.md", RESULTS_DIR / "coverage.csv"])
    banner(log, "STAGE 4 COMPLETE — next: bash scripts/privacy_probe.py")


if __name__ == "__main__":
    main()
