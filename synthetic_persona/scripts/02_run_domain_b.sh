#!/usr/bin/env bash
# Run Domain B (WVS Wave 7) survey simulation — vLLM offline mode
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "[02] Running Domain B (WVS) survey simulation..."
CUDA_VISIBLE_DEVICES=0,1 python -m engine.run_survey --domain wvs

# Verify outputs exist
WVS_RESULTS=$(find results/wvs -name "*.csv" 2>/dev/null | wc -l)
if [ "$WVS_RESULTS" -eq 0 ]; then
    echo "[02] ERROR: No WVS results generated."
    exit 1
fi

echo "[02] Domain B complete: $WVS_RESULTS result files."
touch results/.02_done
