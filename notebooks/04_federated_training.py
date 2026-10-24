import marimo as mo

app = mo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
# 04 — Federated training (FedAvg)

Stage 3b — the heart of the project. Partners train locally, only flat weight
vectors travel to the server, FedAvg aggregates, repeat for R rounds.

**Two equivalent ways to get the artifacts** — pick ONE, not both:

1. terminal: `uv run python scripts/run_federated.py` (canonical, logged,
   Flower with auto-fallback; this notebook then just *views* convergence);
2. or: the run button below trains in-notebook and writes the SAME files
   (`results/models/federated.npz`, `results/metrics/regime_federated.json`,
   `results/runs/fed_history.jsonl`) — fine as an experiment playground,
   just don't do both if you care about wall-time.
        """
    )
    return


@app.cell
def _(mo):
    rounds = mo.ui.slider(1, 40, value=20, label="FedAvg rounds")
    run_button = mo.ui.run_button(label="Run federated (in this notebook)")
    rounds, run_button
    return rounds, run_button


@app.cell
def _(mo, rounds, run_button):
    import json

    import numpy as np

    from fedlearn.config import MODELS_DIR, RUNS_DIR, resolve_config
    from fedlearn.data.partitions import load_partitions
    from fedlearn.eval.evaluate import state_metrics
    from fedlearn.fl.server import run_federated

    # optional interactive trainer — writes the SAME artifacts as
    # scripts/run_federated.py; skip it if you ran the script (view cell below)
    if run_button.value:
        config = resolve_config("configs/config.yaml")
        config["fl"]["rounds"] = rounds.value
        parts = load_partitions(config)
        result = run_federated(
            config, parts["site_data"], parts["eval_data"],
            {ep: ("cls" if ep in ("ames", "herg") else "reg") for ep in parts["dims"]}
        )
        np.savez(MODELS_DIR / "federated.npz", state=result["final_state"])
        with open(RUNS_DIR / "fed_history.jsonl", "w") as fh:
            for entry in result["history"]:
                fh.write(json.dumps(entry) + "\n")
        metrics = state_metrics(result["final_state"], parts["dims"],
                                parts["eval_data"]["X"], parts["eval_data"]["y"],
                                config, f"federated[{result['backend']}]")
        (__import__("pathlib").Path("results/metrics/regime_federated.json")
         .write_text(json.dumps({"federated": metrics}, indent=2)))
        print(f"done via backend={result['backend']} — see log above for per-round loss")
    else:
        print("press ▶ above to train in-notebook (or run scripts/run_federated.py)")
    return


@app.cell
def _(mo):
    import json
    from pathlib import Path

    import matplotlib.pyplot as plt

    from fedlearn.config import RUNS_DIR

    # view-only: plots the per-round history written by the script OR the
    # run-button cell above — no training happens here
    path = Path(RUNS_DIR) / "fed_history.jsonl"
    if not path.exists():
        print("no fed_history.jsonl yet — run scripts/run_federated.py "
              "(or press ▶ above) once, then re-run this cell")
    else:
        history = [json.loads(line) for line in path.read_text().splitlines()]
        endpoints = sorted(history[0]["loss"]) if history else []
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.6))
        for ep in endpoints:
            ax1.plot([h["round"] for h in history],
                     [h["loss"][ep] for h in history], marker=".", label=ep)
        ax1.set_xlabel("round")
        ax1.set_ylabel("local loss (avg over partners)")
        ax1.legend()
        ax1.set_title("FedAvg convergence")
        ax2.plot([h["round"] for h in history], [h["wall_s"] for h in history],
                 marker="o", color="#c96442")
        ax2.set_xlabel("round")
        ax2.set_ylabel("wall time (s)")
        ax2.set_title("per-round cost → the timing dimension")
        plt.tight_layout()
        fig.savefig("results/figures/04_convergence.png", dpi=120)
        print(f"convergence plot from {len(history)} rounds → "
              f"results/figures/04_convergence.png")
        mo.mpl.interactive(fig)
    return

