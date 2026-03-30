#!/usr/bin/env bash
# ============================================================
# Synthetic Persona Survey — Full Pipeline
# vLLM offline mode: GPU freed automatically after each step.
# Run: bash run_all.sh
# ============================================================
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

TOTAL_START=$(date +%s)

log() { echo "[run_all] $(date '+%H:%M:%S') $*"; }

# ----- Step 1: Environment check -----
log "Step 1/5: Environment check..."
bash scripts/01_start_vllm.sh
log "Step 1/5 done."

# ----- Step 2: Domain B (WVS) -----
log "Step 2/5: Running Domain B (WVS)..."
bash scripts/02_run_domain_b.sh
log "Step 2/5 done."

# ----- Step 3: Domain A (Privacy Calculus) -----
log "Step 3/5: Running Domain A (Privacy)..."
bash scripts/03_run_domain_a.sh
log "Step 3/5 done."

# ----- Step 4: Compute metrics -----
log "Step 4/5: Computing metrics..."
bash scripts/04_compute_metrics.sh
log "Step 4/5 done."

# ----- Step 5: Analysis -----
log "Step 5/5: Running analysis..."
bash scripts/05_analyze.sh
log "Step 5/5 done."

# ----- Git commit + push -----
log "Committing results..."
git add results/
git commit -m "experiment: synthetic persona survey results $(date '+%Y-%m-%d %H:%M')" || true
git push || log "WARNING: git push failed (check remote config)"

TOTAL_END=$(date +%s)
ELAPSED=$(( TOTAL_END - TOTAL_START ))
log "Pipeline complete in ${ELAPSED}s. Results in results/"
