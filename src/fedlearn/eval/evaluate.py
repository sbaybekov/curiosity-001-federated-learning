"""fedlearn.eval.evaluate — held-out metrics for a flat state vector.

Shared by every training script so isolated / federated / centralized are
measured with the SAME code path: unpack state → per-endpoint scores →
PR-AUC/RMSE with bootstrap CI, logged + returned.
"""

from __future__ import annotations

from collections import OrderedDict

from ..config import TASK
from ..log import get_logger, section
from ..models.heads import init_heads, unpack_heads
from .metrics import evaluate_classifier, evaluate_regression

log = get_logger("eval.evaluate")


def state_metrics(state, dims: dict[str, int], eval_X: dict, eval_y: dict,
                  config: dict, label: str) -> dict:
    """Score a flattened state on held-out data; return per-endpoint metrics."""
    task_map = TASK
    heads = init_heads(dims, task_map, seed=0)
    lens = OrderedDict((ep, dims[ep]) for ep in sorted(dims))
    unpack_heads(state.astype("float32"), lens, heads)

    section(log, f"held-out metrics [{label}]")
    metrics: dict = {}
    for endpoint in sorted(dims):
        scores = heads[endpoint].scores(eval_X[endpoint])
        if task_map[endpoint] == "cls":
            metrics[endpoint] = evaluate_classifier(
                eval_y[endpoint], scores, reps=config["eval"]["bootstrap_reps"])
        else:
            metrics[endpoint] = evaluate_regression(
                eval_y[endpoint], scores, reps=config["eval"]["bootstrap_reps"])
    return metrics
