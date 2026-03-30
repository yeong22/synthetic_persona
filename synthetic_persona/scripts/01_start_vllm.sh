#!/usr/bin/env bash
# Environment check: vLLM offline mode + model availability
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "[01] Checking vLLM offline mode..."
python3 -c "from vllm import LLM, SamplingParams; print('[01] vLLM import OK')"

echo "[01] Checking model cache..."
python3 -c "
from config.experiment_config import MODEL_ID
from huggingface_hub import snapshot_download
try:
    path = snapshot_download(MODEL_ID, local_files_only=True)
    print(f'[01] Model cached: {MODEL_ID} → {path}')
except Exception:
    print(f'[01] Model not cached. Will download on first run: {MODEL_ID}')
"

echo "[01] Environment ready."
