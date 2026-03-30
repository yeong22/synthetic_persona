#!/usr/bin/env bash
# Compute all metrics (GT-based + GT-free) for all conditions
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Pre-check: survey results must exist
if [ ! -f results/.02_done ] && [ ! -f results/.03_done ]; then
    echo "[04] ERROR: No survey results found. Run 02/03 first."
    exit 1
fi

mkdir -p results/metrics

echo "[04] ── Step A: GT-based metrics (WD, JSD, KS, MeanDiff) ──"
python -m metrics.step_a_gt_based

echo ""
echo "[04] ── Step C: GT-free metrics (SCS, VCR, ICE) ──"
python -m metrics.step_c_compute_all

touch results/.04_done
echo ""
echo "[04] Metrics computation complete."
