#!/bin/bash
# Run the SAE follow-on pipeline stages for the main experiment.
#
# Stages:
#   encode  — GPU-intensive SAE feature encoding from cached dense activations
#   probes  — CPU-only SAE probe training on encoded SAE features
#   report  — CPU-only results-table refresh including SAE probe metrics
#   all     — encode → probes → report
#
# Usage:
#   bash scripts/cluster/run_sae_pipeline.sh --stage encode
#   bash scripts/cluster/run_sae_pipeline.sh --stage probes
#   bash scripts/cluster/run_sae_pipeline.sh --stage report
set -euo pipefail

STAGE="encode"
PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/main.yaml"
INPUT_DIR="artifacts/activations/main"
FEATURE_DIR="artifacts/features/main/sae"
PROBE_DIR="artifacts/models/main/sae_probes"
RESULTS_PATH="results/main/results_table.csv"
METRICS_DIR="results/main/metrics"
BATCH_SIZE="256"
DEVICE="cuda"
DTYPE="float32"
SCRATCH_DIR=""
MODEL_CACHE="${HOME}/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --toolchain-rc) TOOLCHAIN_RC="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --feature-dir|--output-dir) FEATURE_DIR="$2"; shift 2 ;;
    --probe-dir) PROBE_DIR="$2"; shift 2 ;;
    --results-path) RESULTS_PATH="$2"; shift 2 ;;
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

log "=== SAE pipeline starting — stage: ${STAGE} ==="
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

run_encode() {
  log "--- stage: encode ---"
  python scripts/main/encode_sae_features.py \
    --config "$CONFIG" \
    --input-dir "$INPUT_DIR" \
    --output-dir "$FEATURE_DIR" \
    --metrics-dir "$METRICS_DIR" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --dtype "$DTYPE"
  log "--- stage: encode done ---"
}

run_probes() {
  log "--- stage: probes ---"
  python scripts/main/train_sae_probes.py \
    --config "$CONFIG" \
    --input-dir "$FEATURE_DIR" \
    --output-dir "$PROBE_DIR" \
    --metrics-dir "$METRICS_DIR"
  log "--- stage: probes done ---"
}

run_report() {
  log "--- stage: report ---"
  python scripts/main/make_results_table.py \
    --config "$CONFIG" \
    --text-metrics "${METRICS_DIR}/text_baseline_metrics.json" \
    --probe-metrics "${METRICS_DIR}/probe_metrics.json" \
    --sae-probe-metrics "${METRICS_DIR}/sae_probe_metrics.json" \
    --output "$RESULTS_PATH"
  log "--- stage: report done ---"
}

case "$STAGE" in
  encode) run_encode ;;
  probes) run_probes ;;
  report) run_report ;;
  all)
    run_encode
    run_probes
    run_report
    ;;
  *)
    echo "Unknown stage: $STAGE (valid: encode, probes, report, all)" >&2
    exit 1
    ;;
esac

log "=== SAE pipeline done — stage: ${STAGE} ==="
