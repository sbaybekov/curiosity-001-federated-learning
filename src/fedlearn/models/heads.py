"""fedlearn.models.heads — per-endpoint linear heads (SGD, warm-continuation).

Parameter exchange convention
------------------------------
A head's state is an OrderedDict {"coef": (K,) float32, "intercept": (1,) float32}.
``pack_heads`` flattens a dict of heads into ONE float32 vector (deterministic
endpoint order) — that vector is the only thing that travels between server
and clients. ``unpack_heads`` splits it back. Norms/shapes are logged at every
exchange so the terminal shows exactly what moved.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from sklearn.linear_model import SGDClassifier, SGDRegressor

from ..log import get_logger, summarize

log = get_logger("models.heads")


class LinearHead:
    """One linear model for one endpoint: cls (logistic) or reg (linear)."""

    def __init__(self, task: str, dim: int, seed: int) -> None:
        if task not in ("cls", "reg"):
            raise ValueError(f"task must be cls|reg, got {task}")
        self.task = task
        self.dim = int(dim)
        kwargs = dict(random_state=seed)
        if task == "cls":
            # logistic regression head (log_loss) — canonical FedAvg learner
            self.est = SGDClassifier(
                loss="log_loss", penalty="l2", average=False, **kwargs
            )
        else:
            self.est = SGDRegressor(
                loss="squared_error", penalty="l2", average=False, **kwargs
            )

    # -- parameter hooks (the FL interface) -----------------------------------
    def get_parameters(self) -> OrderedDict:
        params: OrderedDict = OrderedDict()
        params["coef"] = np.asarray(self.est.coef_, dtype=np.float32).ravel().copy()
        params["intercept"] = np.asarray(self.est.intercept_, dtype=np.float32).ravel().copy()
        return params

    def set_parameters(self, params: OrderedDict) -> None:
        coef = np.asarray(params["coef"], dtype=np.float32).ravel()
        intercept = np.asarray(params["intercept"], dtype=np.float32).ravel()
        if self.task == "cls":
            self.est.coef_ = coef.reshape(1, -1)      # SGDClassifier: (n_classes-1, K)
            self.est.classes_ = np.array([0, 1])
        else:
            self.est.coef_ = coef                     # SGDRegressor: (K,) 1-D
        self.est.intercept_ = intercept
        self.est.n_features_in_ = self.dim

    # -- local training -------------------------------------------------------
    def partial_fit(self, X: np.ndarray, y: np.ndarray, epochs: int, lr: float) -> float:
        """Run ``epochs`` SGD passes starting from current weights; return final loss."""
        for _ in range(epochs):
            if self.task == "cls":
                self.est.partial_fit(X, y, classes=np.array([0, 1]))
            else:
                self.est.partial_fit(X, y)
        loss = self.loss(X, y)
        return loss

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.task == "cls":
            proba = np.clip(self.est.predict_proba(X), 1e-9, 1 - 1e-9)[:, 1]
            return float(-np.mean(y * np.log(proba) + (1 - y) * np.log(1 - proba)))
        pred = self.est.predict(X)
        return float(np.mean((y - pred) ** 2))

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Continuous scores for ranking metrics (AUC-friendly)."""
        if self.task == "cls":
            return self.est.decision_function(X)
        return self.est.predict(X).ravel()


def init_heads(dims: dict[str, int], task_map: dict[str, str], seed: int) -> dict[str, LinearHead]:
    """One fresh head per endpoint (all sites start from the same seed)."""
    heads: dict[str, LinearHead] = {}
    for endpoint, dim in dims.items():
        heads[endpoint] = LinearHead(task_map[endpoint], dim, seed)
    summarize(f"initialized heads for {sorted(heads)}", "", log)
    return heads


def pack_heads(heads: dict[str, LinearHead], log_=log) -> tuple[np.ndarray, OrderedDict]:
    """Flatten heads → (single float32 vector, lens spec {endpoint: n_coef})."""
    parts: list[np.ndarray] = []
    lens: OrderedDict = OrderedDict()
    for endpoint in sorted(heads):
        params = heads[endpoint].get_parameters()
        flat = np.concatenate([params["coef"], params["intercept"]]).astype(np.float32)
        parts.append(flat)
        lens[endpoint] = int(len(params["coef"]))
    state = np.concatenate(parts)
    log_.info(
        f"  pack: {dict(lens)} coefs+intercepts → state shape {state.shape} "
        f"(‖state‖₂={np.linalg.norm(state):.3f})"
    )
    return state, lens


def unpack_heads(
    state: np.ndarray, lens: OrderedDict, heads: dict[str, LinearHead], log_=log
) -> None:
    """Apply a flat state vector back onto the heads (in place)."""
    offset = 0
    for endpoint, n_coef in lens.items():
        coef = state[offset:offset + n_coef]
        intercept = state[offset + n_coef:offset + n_coef + 1]
        offset += n_coef + 1
        heads[endpoint].set_parameters(OrderedDict(coef=coef, intercept=intercept))
    log_.info(
        f"  unpack: restored {sum(lens.values())} coefs + {len(lens)} intercepts "
        f"(‖state‖₂={np.linalg.norm(state):.3f})"
    )
