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
    seed = int(config["fl"]["seed"])
    rng = np.random.default_rng(seed + 1)
    trainers = {cid: LocalTrainer(dims, task_map, seed) for cid in cids}
    state = state0.astype(np.float32).copy()
    history: list[dict] = []

    for rnd in range(1, rounds + 1):
        t0 = time.perf_counter()
        updates: list[np.ndarray] = []
        losses: dict[str, list[float]] = {ep: [] for ep in sorted(dims)}
        for cid in cids:
            tr = trainers[cid]
            tr.set_state(state)
            local = tr.fit_round(site_data[cid]["X"], site_data[cid]["y"], epochs, lr)
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
    from flwr.client import ClientApp, NumPyClient as FlwrNumPyClient
    from flwr.common import parameters_to_ndarrays
    from flwr.server import ServerConfig
    from flwr.server.strategy import FedAvg
    from flwr.simulation import start_simulation

    from .client import FlowerNumPyClient

    rounds = int(config["fl"]["rounds"])
    epochs = int(config["model"]["epochs_per_round"])
    lr = float(config["model"]["lr"])
    seed = int(config["fl"]["seed"])
    log.info(f"backend: FLOWER {flwr.__version__} — in-process simulation, "
             f"{len(cids)} virtual clients, FedAvg strategy")

    # flwr only accepts true NumPyClient instances → subclass at the last moment
    # so the rest of the package stays flwr-free (manual fallback needs nothing).
    class _FlowerClient(FlowerNumPyClient, FlwrNumPyClient):
        pass

    holder: dict = {"state": None, "round": 0}

    class StashingFedAvg(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            agg, metrics = super().aggregate_fit(server_round, results, failures)
            if agg is not None:
                state = np.concatenate([np.asarray(a, dtype=np.float32)
                                        for a in parameters_to_ndarrays(agg)])
                holder["state"], holder["round"] = state, server_round
                log.info(f"  server round {server_round}: aggregated {len(results)} "
                         f"updates → ‖state‖₂={np.linalg.norm(state):.3f}")
            return agg, metrics

    def client_fn(cid: str):
        data = site_data[cid]
        client = _FlowerClient(LocalTrainer(dims, task_map, seed),
                               data["X"], data["y"], None, str(cid))
        return client.to_client() if hasattr(client, "to_client") else client

    client_app = ClientApp(client_fn=client_fn)
    # on_fit_config_fn is what actually carries epochs/lr to each client;
    # without it FlowerNumPyClient.fit uses its own defaults (3 epochs, lr 0.01)
    strategy = StashingFedAvg(
        fraction_fit=float(config["fl"]["fraction_fit"]),
        on_fit_config_fn=lambda rnd: {"epochs": epochs, "lr": lr},
    )
    server_config = ServerConfig(num_rounds=rounds)

    # API drift between flwr releases: num_clients → num_supernodes
    try:
        start_simulation(client_app=client_app, num_supernodes=len(cids),
                         config=server_config, strategy=strategy,
                         backend_config={"init_args": {"num_cpus": 1}})
    except TypeError:
        log.info("  flwr API variant detected — retrying with num_clients=...")
        start_simulation(client_app=client_app, num_clients=len(cids),
                         config=server_config, strategy=strategy)

    state = holder["state"] if holder["state"] is not None else state0
    log.info("flower run finished — final aggregated state recovered from strategy")
    return _finish("flower", [], state, eval_data, dims, task_map)


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
