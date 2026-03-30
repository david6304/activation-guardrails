#!/bin/bash
# Run SAE feature encoding for the main experiment on a GPU node.
set -euo pipefail

PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/main.yaml"
INPUT_DIR="artifacts/activations/main"
OUTPUT_DIR="artifacts/features/main/sae"
METRICS_DIR="results/main/metrics"
BATCH_SIZE="256"
DEVICE="cuda"
DTYPE="float32"
SCRATCH_DIR=""
MODEL_CACHE="${HOME}/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --toolchain-rc) TOOLCHAIN_RC="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --metrics-dir) METRICS_DIR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --scratch-dir) SCRATCH_DIR="$2"; shift 2 ;;
    --model-cache) MODEL_CACHE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

source "$TOOLCHAIN_RC"
source "${VENV_PATH}/bin/activate"

cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1

log "=== SAE encoding starting ==="
log "Node: $(hostname)  Job: ${SLURM_JOB_ID:-local}"
log "Config: ${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  | while read -r line; do log "GPU: $line"; done || true

export HF_HOME="$MODEL_CACHE"
export HUGGINGFACE_HUB_CACHE="${MODEL_CACHE}/hub"
export TRANSFORMERS_CACHE="${MODEL_CACHE}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ -n "$SCRATCH_DIR" ]]; then
  mkdir -p "$SCRATCH_DIR"
fi

python scripts/main/encode_sae_features.py \
  --config "$CONFIG" \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --metrics-dir "$METRICS_DIR" \
  --batch-size "$BATCH_SIZE" \
  --device "$DEVICE" \
  --dtype "$DTYPE"

log "=== SAE encoding done ==="
