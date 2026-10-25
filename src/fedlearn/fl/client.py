"""fedlearn.fl.client — Flower NumPyClient wrapper + shared local trainer.

A "partner" here trains ONE state vector per round: it contains the linear
heads for ALL endpoints (each endpoint's model is independent; they just ride
the same federation orchestration). ``get_parameters`` returns the flat state,
``fit`` trains locally and returns updated state + local-loss metrics,
``evaluate`` scores a local validation split.
"""

from __future__ import annotations

import time

import numpy as np

from ..log import get_logger
from ..models.heads import init_heads, pack_heads, unpack_heads

log = get_logger("fl.client")


class LocalTrainer:
    """Owns per-endpoint heads for one site; trains and exchanges state."""

    def __init__(self, dims: dict[str, int], task_map: dict[str, str], seed: int) -> None:
        self.dims = dims
        # constant schedule — FL server drives lr per round via eta0; must be
        # independent of sklearn's t_ so manual & Flower clients match exactly.
        self.heads = init_heads(dims, task_map, seed, lr_schedule="constant")
        # lens must be known up front so set_state() works regardless of call
        # order — Flower may call evaluate() before any get_state()/fit.
        _, self.lens = pack_heads(self.heads)

    def set_state(self, state: np.ndarray) -> None:
        unpack_heads(state, self.lens, self.heads)

    def get_state(self) -> np.ndarray:
        state, self.lens = pack_heads(self.heads)
        return state

    def fit_round(self, Xs: dict[str, np.ndarray], ys: dict[str, np.ndarray],
                  epochs: int, lr: float) -> dict[str, float]:
        """Local SGD pass over each endpoint's partition; return per-endpoint loss."""
        losses: dict[str, float] = {}
        for endpoint in sorted(Xs):
            losses[endpoint] = self.heads[endpoint].partial_fit(
                Xs[endpoint], ys[endpoint], epochs, lr
            )
        return losses


class FlowerNumPyClient:
    """Flower NumPyClient-style wrapper: identical method signatures, but NOT
    a flwr subclass (flwr is imported lazily; the Flower backend subclasses
    this class right before simulation starts)."""

    def __init__(self, trainer: LocalTrainer, Xs: dict, ys: dict,
                 eval_data: tuple[dict, dict] | None, cid: str) -> None:
        self.trainer = trainer
        self.Xs, self.ys = Xs, ys
        self.eval = eval_data
        self.cid = cid

    def get_parameters(self, config=None):
        return [self.trainer.get_state()]

    def fit(self, parameters, config=None):
        state = np.asarray(parameters[0], dtype=np.float32)
        self.trainer.set_state(state)
        rounds = int(config.get("epochs", 3)) if config else 3
        lr = float(config.get("lr", 0.01)) if config else 0.01
        started = time.perf_counter()
        losses = self.trainer.fit_round(self.Xs, self.ys, rounds, lr)
        elapsed = time.perf_counter() - started
        n_train = int(next(iter(self.ys.values())).size)
        metrics = {"loss/" + k: v for k, v in losses.items()}
        metrics.update({"n_train": n_train, "fit_time_s": round(elapsed, 3)})
        log.info(
            f"    [cid={self.cid}] fit round: loss="
            + ", ".join(f"{k}={v:.4f}" for k, v in losses.items())
            + f" | {elapsed:.2f}s"
        )
        return [self.trainer.get_state()], n_train, metrics

    def evaluate(self, parameters, config=None):
        state = np.asarray(parameters[0], dtype=np.float32)
        self.trainer.set_state(state)
        if self.eval is None:
            return 0.0, 0, {}
        Xv, yv = self.eval
        losses = {}
        for endpoint in sorted(Xv):
            losses[endpoint] = self.trainer.heads[endpoint].loss(Xv[endpoint], yv[endpoint])
        loss = float(np.mean(list(losses.values())))
        n_val = int(next(iter(yv.values())).size)
        log.info(f"    [cid={self.cid}] evaluate: " + ", ".join(
            f"{k}={v:.4f}" for k, v in losses.items()))
        return loss, n_val, {"val_loss/" + k: v for k, v in losses.items()}
