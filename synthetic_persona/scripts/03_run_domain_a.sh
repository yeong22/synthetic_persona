#!/usr/bin/env bash
# Run Domain A (Privacy Calculus) survey simulation — vLLM offline mode
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "[03] Running Domain A (Privacy Calculus) survey simulation..."
CUDA_VISIBLE_DEVICES=0,1 python -m engine.run_survey --domain privacy

# Verify outputs exist
PRIV_RESULTS=$(find results/privacy -name "*.csv" 2>/dev/null | wc -l)
if [ "$PRIV_RESULTS" -eq 0 ]; then
    echo "[03] ERROR: No Privacy results generated."
    exit 1
fi

echo "[03] Domain A complete: $PRIV_RESULTS result files."
touch results/.03_done
