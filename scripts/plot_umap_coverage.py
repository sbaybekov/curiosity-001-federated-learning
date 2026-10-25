#!/usr/bin/env python
"""UMAP of the chemical space per property, colored by silo (partner) coverage.

For each endpoint, embed the union of every silo's fingerprints with UMAP and
colour points by which partner's silo they belong to. A convex hull per silo
(drawn in UMAP space) visualises how much of the shared chemical space each
partner covers; a coverage table prints each silo's share of the union hull as
a crude but honest "fraction of space explored".

Outputs results/figures/umap_coverage_<endpoint>.png + a markdown summary
(results/umap_coverage.md). Molecule count per plot can be downsampled with
--max-n (per endpoint) to keep UMAP fast on large single-property sets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from sklearn.utils import resample

from fedlearn.config import ENDPOINTS, FIGURES_DIR, RESULTS_DIR, SPLIT_DIR, resolve_config
from fedlearn.data.featurize import featurize_df
from fedlearn.log import banner, get_logger, section, summarize, timed
from fedlearn.provenance import Provenance

log = get_logger("plot.umap_coverage")

SILO_COLORS = ["#4a6a9a", "#c96442", "#4a7a4a"]  # partner p0/p1/p2
HELDOUT_COLOR = "#888888"
SILO_LABELS = {i: f"silo {i}" for i in range(3)}


def _hull_area(points: np.ndarray) -> float:
    """2-D convex-hull area, or 0.0 if fewer than 3 distinct points."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return 0.0
    return float(ConvexHull(pts).volume)  # scipy reports polygon area as .volume


def _hull_xy(points: np.ndarray) -> np.ndarray:
    """Ordered hull vertices for matplotlib Polygon (empty if degenerate)."""
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return np.zeros((0, 2))
    return pts[ConvexHull(pts).vertices]


def _embed(umap, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 2-D UMAP embedding tuned for Morgan fingerprints.

    Default Euclidean distance on 2048-bit ECFP scatters the map; Morgan bits
    want the Jaccard dissimilarity (shared-bit fraction). n_neighbors scales
    with dataset size to avoid fragmenting small silos, and a small min_dist
    packs near-neighbours into tight clusters instead of a diffuse cloud.
    (UMAP forces n_jobs=1 when random_state is set — deterministic over speed.)
    Returns (finite rows, row index keep-mask); Jaccard on an all-zero/duplicate
    fingerprint can yield NaN, which would break the convex hull below.
    """
    umap.set_params(
        n_neighbors=min(30, max(5, len(X) // 150)),
        min_dist=0.05,
        metric="jaccard",
        n_epochs=500,
    )
    embed = umap.fit_transform(X)
    finite = np.isfinite(embed).all(axis=1)
    if not finite.all():
        log.warning(f"  dropping {int((~finite).sum())} NaN/non-finite UMAP rows")
    return embed[finite], finite


def main() -> None:
    parser = argparse.ArgumentParser(description="UMAP chemical-space coverage per silo")
    parser.add_argument("--max-n", type=int, default=0,
                        help="downsample to ≤N molecules per endpoint (0 = no sampling)")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    config = resolve_config(args.config)
    radius = int(config["model"]["radius"])
    nbits = int(config["model"]["nbits"])
    prove = Provenance("plot-umap-coverage")
    prove.begin({"config": config, "max_n": args.max_n})

    banner(log, "UMAP CHEMICAL-SPACE COVERAGE — per property, coloured by silo")
    coverage_rows: list[dict] = []

    for endpoint in ENDPOINTS:
        section(log, f"endpoint: {endpoint}")
        frames, parts = [], []
        for partner in range(config["silos"]["n_partners"]):
            df = pd.read_parquet(SPLIT_DIR / f"silo_{partner}_{endpoint}.parquet")
            frames.append(df)
            parts.append(np.full(len(df), partner, dtype=int))
        df_all = pd.concat(frames, ignore_index=True)
        partner_of = np.concatenate(parts)
        X, y, ids, bad = featurize_df(df_all, radius, nbits)
        if bad:
            log.warning(f"  {endpoint}: {bad} SMILES unparseable and dropped")
        if args.max_n and len(X) > args.max_n:
            keep = resample(np.arange(len(X)), n_samples=args.max_n, random_state=0,
                            stratify=partner_of)
            X, partner_of, y, ids = X[keep], partner_of[keep], y[keep], ids[keep]
        with timed(log, f"UMAP embedding {endpoint} ({len(X)} pts)"):
            import umap
            embed, keep = _embed(umap.UMAP(random_state=0), X)
            partner_of = partner_of[keep]
        section(log, f"UMAP {endpoint} hull coverage")

        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        union_area = _hull_area(embed)
        rows = []
        for partner in sorted(set(partner_of)):
            m = partner_of == partner
            ax.scatter(embed[m, 0], embed[m, 1], s=2, alpha=0.45,
                       color=SILO_COLORS[partner], label=SILO_LABELS[partner])
            hull = _hull_xy(embed[m])
            if len(hull) >= 3:
                ax.add_patch(matplotlib.patches.Polygon(
                    hull, closed=True, fill=True, alpha=0.08,
                    color=SILO_COLORS[partner], linewidth=0))
            area = _hull_area(embed[m])
            share = area / union_area if union_area > 0 else float("nan")
            rows.append({
                "endpoint": endpoint, "silo": partner, "n": int(m.sum()),
                "hull_area": round(area, 4),
                "hull_share_of_union": round(share, 4) if share == share else None,
            })
            summarize(f"  {endpoint} silo {partner}",
                      f"n={int(m.sum()):>6}  hull={area:9.4f}  share={share:6.1%}", log)
        coverage_rows.extend(rows)
        ax.set_title(f"{endpoint} — chemical space by silo")
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("UMAP dim 1")
        ax.set_ylabel("UMAP dim 2")
        ax.set_aspect("equal", adjustable="datalim")
        fig.tight_layout()
        out = FIGURES_DIR / f"umap_coverage_{endpoint}.png"
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150)
        plt.close(fig)
        log.info(f"  → {out}")

    # ---- honest coverage summary ----------------------------------------------
    md_lines = ["| endpoint | silo | molecules | hull area | share of union hull |",
                "|---|---|---|---|---|"]
    for r in coverage_rows:
        share = f"{r['hull_share_of_union']:.1%}" if r["hull_share_of_union"] else "n/a"
        md_lines.append(f"| {r['endpoint']} | {r['silo']} | {r['n']} | "
                        f"{r['hull_area']} | {share} |")
    summary_path = RESULTS_DIR / "umap_coverage.md"
    summary_path.write_text("\n".join(md_lines) + "\n")
    log.info(f"coverage table → {summary_path}")

    prove.record({"endpoints": list(ENDPOINTS), "figures": [
        str(FIGURES_DIR / f"umap_coverage_{ep}.png") for ep in ENDPOINTS]},
        artifacts=[summary_path, *[FIGURES_DIR / f"umap_coverage_{ep}.png" for ep in ENDPOINTS]])
    banner(log, "UMAP COVERAGE COMPLETE — see results/figures/umap_coverage_*.png")


if __name__ == "__main__":
    main()