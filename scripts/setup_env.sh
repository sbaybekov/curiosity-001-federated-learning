#!/usr/bin/env bash
# FedMol — environment bootstrap (single command, verbose on purpose).
# Usage: bash scripts/setup_env.sh
set -euo pipefail

BANNER="======================================================================"
echo ""
echo "$BANNER"
echo "  SETUP 1/3 — uv + python 3.12 pinned"
echo "$BANNER"
command -v uv >/dev/null || { echo "ERROR: uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
uv --version
cat .python-version | sed 's/^/  python pinned: /'

echo ""
echo "$BANNER"
echo "  SETUP 2/3 — uv venv + pip install (rdkit, flwr[simulation], hydra, mlflow, pytdc)"
echo "              first run downloads ~1-2 GB of wheels (rdkit + ray). Be patient."
echo "              project is installed NON-editable (managed=false) — the editable"
echo "              .pth link is unreliable on macOS, producing 'No module named fedlearn'."
echo "$BANNER"
uv venv 2>/dev/null || true
uv pip install .

echo ""
echo "$BANNER"
echo "  SETUP 3/3 — import & version check (every key dependency)"
echo "$BANNER"
uv run python - <<'PY'
import importlib, sys

print(f"  python {sys.version.split()[0]}")
required = ["fedlearn", "rdkit", "flwr", "hydra", "mlflow", "tdc", "sklearn", "pandas",
            "matplotlib", "scipy", "marimo"]
missing = []
for name in required:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "?")
        print(f"  ✔ {name:<12} {version}")
    except Exception as exc:
        missing.append(name)
        print(f"  ✗ {name:<12} MISSING — {exc}")
if missing:
    print(f"\nERROR: missing dependencies: {missing} — re-run `uv sync`")
    sys.exit(1)
print("\n  env OK — next: bash scripts/run_all.sh  (or scripts/download_data.py alone)")
PY
