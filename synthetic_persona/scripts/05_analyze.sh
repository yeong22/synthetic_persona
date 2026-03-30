#!/usr/bin/env bash
# Analyze: DI ↔ GT-based rank concordance
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Pre-check
if [ ! -f results/.04_done ]; then
    echo "[05] ERROR: Metrics not computed. Run 04 first."
    exit 1
fi

echo "[05] Running concordance analysis..."
python -m metrics.analysis --metrics-dir results/metrics

touch results/.05_done
echo ""
echo "[05] Analysis complete. Results in results/metrics/"
