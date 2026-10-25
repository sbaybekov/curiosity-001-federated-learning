"""fedlearn.fl.server — FedAvg orchestration: Flower (primary) or manual loop.

The manual loop is a reference implementation with identical math, used
whenever flwr/ray is missing or misbehaves — the pipeline never dies on a
framework quirk. Both backends share LocalTrainer (client) and fedavg
(aggregation); the terminal logs state exactly which backend ran.

The server process only ever touches flat parameter vectors — never molecules.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from functools import reduce

import numpy as np

from ..log import banner, get_logger, section, summarize
from ..models.heads import init_heads, pack_heads, unpack_heads
from ..privacy import dp as _dp
from .client import LocalTrainer

log = get_logger("fl.server")


def round_lr(base: float, rnd: int, schedule: str = "invscale") -> float:
    """Explicit per-round learning rate, shared by every backend.

    ``invscale`` mirrors FedAvg-style annealing (eta0/√round) deterministically;
    ``constant`` keeps lr fixed. Server drives this so sklearn's internal t_
    heuristic never affects results — backend-identical and reproducible.
    """
    if schedule == "constant":
        return float(base)
    if schedule == "invscale":
        return float(base) / float(rnd) ** 0.5
    raise ValueError(f"unknown lr schedule: {schedule!r} (use 'constant'|'invscale')")


def run_federated(config: dict, site_data: dict, eval_data: dict, task_map: dict) -> dict:
    """Top-level FedAvg entry used by scripts and notebooks.

    site_data: {cid: {"X": {ep: ndarray}, "y": {ep: ndarray}}}  — local partitions.
    eval_data: {"X": {ep: ndarray}, "y": {ep: ndarray}}          — held-out, no mols.
    Returns {"backend", "history", "final_state", "heldout_loss"}.
    """
    fl_cfg, pr_cfg = config["fl"], config["privacy"]
    cids = sorted(site_data)
    dims = {ep: site_data[cids[0]]["X"][ep].shape[1] for ep in site_data[cids[0]]["X"]}

    banner(log, f"FEDERATED LOOP — {fl_cfg['rounds']} rounds × {len(cids)} partners")
    section(log, f"DP={'ON ' if pr_cfg['enabled'] else 'OFF'} "
                 f"(clip C={pr_cfg['clip_C']}, z={pr_cfg['noise_multiplier']})")
    summarize("backend", f"{fl_cfg['backend']} (falls back to manual on failure)", log)
    summarize("state dims", dims, log)

    settings = _dp.DPSettings(
        enabled=bool(pr_cfg["enabled"]),
        clip_C=float(pr_cfg["clip_C"]),
        noise_multiplier=float(pr_cfg["noise_multiplier"]),
    )
    state0 = pack_heads(init_heads(dims, task_map, int(fl_cfg["seed"])), log)[0]

    if fl_cfg["backend"] == "flower":
        try:
            return _run_flower(config, site_data, eval_data, cids, dims, task_map,
                               state0, settings)
        except Exception as exc:  # noqa: BLE001 — framework quirks must not kill the run
            log.warning(f"Flower path failed ({exc!r}) — switching to manual FedAvg loop")
    return _run_manual(config, site_data, eval_data, cids, dims, task_map, state0, settings)


# ---------------------------------------------------------------------------
# Manual FedAvg — deterministic reference loop (identical math to Flower)
# ponytail: deliberate failover so a run never dies on flwr/ray quirks;
# upgrade path = drop this and require the Flower backend once it's proven
# stable on the host.
# ---------------------------------------------------------------------------
def _run_manual(config, site_data, eval_data, cids, dims, task_map, state0, settings) -> dict:
    log.info("backend: MANUAL FedAvg loop (plain numpy, no server process)")
    rounds = int(config["fl"]["rounds"])
    epochs = int(config["model"]["epochs_per_round"])
    lr = float(config["model"]["lr"])
    schedule = str(config["model"].get("lr_schedule", "invscale"))
    seed = int(config["fl"]["seed"])
    rng = np.random.default_rng(seed + 1)
    trainers = {cid: LocalTrainer(dims, task_map, seed) for cid in cids}
    state = state0.astype(np.float32).copy()
    history: list[dict] = []

    for rnd in range(1, rounds + 1):
        t0 = time.perf_counter()
        updates: list[np.ndarray] = []
        losses: dict[str, list[float]] = {ep: [] for ep in sorted(dims)}
        round_lr_ = round_lr(lr, rnd, schedule)
        for cid in cids:
            tr = trainers[cid]
            tr.set_state(state)
            local = tr.fit_round(site_data[cid]["X"], site_data[cid]["y"], epochs, round_lr_)
            for ep in sorted(dims):
                losses[ep].append(local[ep])
            if settings.enabled:  # clip + noise BEFORE the update leaves the client
                updates.append(settings.sanitize(tr.get_state() - state, rng))
            else:
                updates.append(tr.get_state() - state)
        state = state + reduce(np.add, updates) / len(updates)
        wall = time.perf_counter() - t0
        line = "  ".join(f"{ep}={sum(losses[ep])/len(losses[ep]):.4f}" for ep in sorted(dims))
        log.info(f"  round {rnd:>3}/{rounds} | local loss: {line} | "
                 f"‖Δ‖₂={np.linalg.norm(state - state0):.3f} | {wall:.2f}s")
        history.append({
            "round": rnd,
            "loss": {ep: float(np.mean(losses[ep])) for ep in sorted(dims)},
            "wall_s": round(wall, 3),
        })
    return _finish("manual", history, state, eval_data, dims, task_map)


# ---------------------------------------------------------------------------
# Flower backend — ClientApp simulation with a stashing FedAvg strategy
# ---------------------------------------------------------------------------
def _run_flower(config, site_data, eval_data, cids, dims, task_map, state0, settings) -> dict:
    import flwr
    from flwr.client import NumPyClient as FlwrNumPyClient
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.simulation import start_simulation

    from .client import FlowerNumPyClient

    rounds = int(config["fl"]["rounds"])
    epochs = int(config["model"]["epochs_per_round"])
    lr = float(config["model"]["lr"])
    schedule = str(config["model"].get("lr_schedule", "invscale"))
    seed = int(config["fl"]["seed"])
    log.info(f"backend: FLOWER {flwr.__version__} — in-process simulation, "
             f"{len(cids)} virtual clients, FedAvg strategy")

    # flwr only accepts true NumPyClient instances → subclass at the last moment
    # so the rest of the package stays flwr-free (manual fallback needs nothing).
    class _FlowerClient(FlowerNumPyClient, FlwrNumPyClient):
        pass

    holder: dict = {"state": None, "round": 0, "history": []}

    class StashingFedAvg(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            if not results:
                return None, {}
            # Plain MEAN of client states — identical to the manual reference
            # loop. Flower's default FedAvg weights by num_examples, and since
            # silos are unequal that would diverge from the manual baseline
            # (and weight the whole 4-head vector by ONE endpoint's count).
            states = [np.concatenate([np.asarray(p, dtype=np.float32)
                                      for p in parameters_to_ndarrays(r.parameters)])
                      for _, r in results]
            state = np.mean(np.stack(states), axis=0).astype(np.float32)
            holder["state"], holder["round"] = state, int(server_round)
            # Reconstruct per-endpoint mean loss using the SAME schema as
            # the manual loop, so fed_history.jsonl is backend-agnostic.
            loss: dict[str, float] = {}
            for ep in sorted(dims):
                vals = [r.metrics[f"loss/{ep}"] for _, r in results
                        if f"loss/{ep}" in r.metrics]
                if vals:
                    loss[ep] = float(np.mean(vals))
            wall = [r.metrics["fit_time_s"] for _, r in results
                    if "fit_time_s" in r.metrics]
            holder["history"].append({
                "round": int(server_round),
                "loss": loss,
                "wall_s": round(float(np.mean(wall)) if wall else 0.0, 3),
            })
            log.info(f"  server round {server_round}: averaged {len(results)} "
                     f"updates → ‖state‖₂={np.linalg.norm(state):.3f}")
            return ndarrays_to_parameters([state]), {}

    def client_fn(cid):
        # legacy simulation passes cid as an int (0..N-1); site_data is keyed
        # "p0".."pN-1". Accept either spelling so the lookup is version-proof.
        key = cid if cid in site_data else f"p{cid}"
        data = site_data[key]
        client = _FlowerClient(
            LocalTrainer(dims, task_map, seed),
            data["X"], data["y"],
            (eval_data["X"], eval_data["y"]),   # shared held-out (eval only, never trained on)
            str(key),
        )
        return client.to_client() if hasattr(client, "to_client") else client

    strategy = StashingFedAvg(
        fraction_fit=float(config["fl"]["fraction_fit"]),
        initial_parameters=ndarrays_to_parameters([state0]),
        on_fit_config_fn=lambda rnd: {
            "epochs": epochs,
            "lr": round_lr(lr, int(rnd), schedule),
        },
    )
    server_config = ServerConfig(num_rounds=rounds)

    # flwr 1.42 routes start_simulation to the LEGACY app: it takes
    # client_fn/num_clients (not the client_app/num_supernodes form). Any
    # unsupported variant is caught by run_federated and falls back to the
    # manual FedAvg loop (identical math).
    start_simulation(
        client_fn=client_fn,
        num_clients=len(cids),
        config=server_config,
        strategy=strategy,
        client_resources={"num_cpus": 1},
    )

    state = holder["state"] if holder["state"] is not None else state0
    log.info(f"flower run finished — {len(holder['history'])} rounds, "
             f"final aggregated state recovered from strategy")
    return _finish("flower", holder["history"], state, eval_data, dims, task_map)


def _finish(backend, history, state, eval_data, dims, task_map) -> dict:
    """Preliminary held-out loss of the final global state (full metrics come
    from the eval stage — this keeps the training script self-checking)."""
    heads = init_heads(dims, task_map, seed=0)
    lens = OrderedDict((ep, dims[ep]) for ep in sorted(dims))
    unpack_heads(state.astype(np.float32), lens, heads)
    losses = {ep: heads[ep].loss(eval_data["X"][ep], eval_data["y"][ep])
              for ep in sorted(dims)}
    section(log, "final global model — held-out loss (preliminary)")
    line = "  ".join(f"{ep}={v:.4f}" for ep, v in losses.items())
    log.info(f"  {line}")
    summarize("backend", backend, log)
    return {"backend": backend, "history": history,
            "final_state": state.astype(np.float32), "heldout_loss": losses}
