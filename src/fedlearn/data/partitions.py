"""fedlearn.data.partitions — one loader for every training script.

Turns the silo parquets into the dict shape the FL server + baselines share:
{site_data: {cid: {"X": {ep}, "y": {ep}}}, eval_data: {"X","y"} (held-out),
 dims: {ep: K}}. Uses the results/features npz cache; hit/miss is logged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ENDPOINTS
from ..log import get_logger, section, summarize
from .featurize import featurize_df
from .split import SPLIT_DIR

log = get_logger("data.partitions")


def load_partitions(config: dict) -> dict:
    """Load + featurize every silo and the held-out set for all endpoints."""
    section(log, "loading & featurizing partitions")
    n = config["silos"]["n_partners"]
    radius = config["model"]["radius"]
    nbits = config["model"]["nbits"]

    site_data: dict = {}
    eval_data: dict = {"X": {}, "y": {}}
    dims: dict[str, int] = {}

    for endpoint in ENDPOINTS:
        held = pd.read_parquet(SPLIT_DIR / f"heldout_{endpoint}.parquet")
        Xh, yh, _, bad_h = featurize_df(held, radius, nbits)
        eval_data["X"][endpoint] = Xh
        eval_data["y"][endpoint] = yh
        dims[endpoint] = int(Xh.shape[1]) if Xh.ndim == 2 else 0
        if bad_h:
            log.warning(f"  heldout {endpoint}: {bad_h} unparseable")
        for partner in range(n):
            df = pd.read_parquet(SPLIT_DIR / f"silo_{partner}_{endpoint}.parquet")
            X, y, ids, bad = featurize_df(df, radius, nbits)
            site_data.setdefault(f"p{partner}", {}).setdefault("X", {})[endpoint] = X
            site_data[f"p{partner}"].setdefault("y", {})[endpoint] = y
            if bad:
                log.warning(f"  silo {partner} {endpoint}: {bad} unparseable")

    summarize("endpoint dims (state vector per head)", dims, log)
    for cid in sorted(site_data):
        n_first = int(len(next(iter(site_data[cid]["y"].values()))))
        summarize(f"site {cid} train rows", n_first, log)
    summarize("held-out rows per endpoint",
              {ep: int(len(eval_data["y"][ep])) for ep in ENDPOINTS}, log)
    return {"site_data": site_data, "eval_data": eval_data, "dims": dims}
