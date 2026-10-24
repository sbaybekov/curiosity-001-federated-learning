#!/usr/bin/env bash
# FedMol — one-command pipeline runner.
# Usage:
#   bash scripts/run_all.sh            # stages 1-5, stop on first error
#   bash scripts/run_all.sh --start 3  # resume from stage 3 (data+splits exist)
#   bash scripts/run_all.sh --end 3    # run through stage 3 only
set -euo pipefail

START=1; END=5
[[ "${1:-}" == "--start" ]] && START="$2"
[[ "${1:-}" == "--end" ]]   && END="$2"

UV=${UV:-uv}
sep(){ printf '%s\n' "────────────────────────────────────────────────────────────────"; }
run(){ echo; sep; echo ">>> STAGE $1/5 — $2"; sep; "$UV" run python "scripts/$3"; }
if [ "$START" -le 1 ] && [ "$END" -ge 1 ]; then run 1 "DATA & SILOS" download_data.py; fi
if [ "$START" -le 2 ] && [ "$END" -ge 2 ]; then run 2 "VALIDATION & QC" run_validation.py; fi
if [ "$START" -le 3 ] && [ "$END" -ge 3 ]; then run 3 "MODELS & BASELINES" run_models.py; fi
if [ "$START" -le 3 ] && [ "$END" -ge 3 ]; then run 3 "FEDERATED LOOP" run_federated.py; fi
if [ "$START" -le 4 ] && [ "$END" -ge 4 ]; then run 4 "BENCHMARK" benchmark.py; fi
if [ "$START" -le 5 ] && [ "$END" -ge 5 ]; then run 5 "PRIVACY AUDIT" privacy_probe.py; fi

echo; sep; echo ">>> ALL STAGES DONE — view results/: money_plot.png, benchmark_summary.md"; sep
