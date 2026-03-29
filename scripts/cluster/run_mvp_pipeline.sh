#!/bin/bash
set -euo pipefail

STAGE="all"
PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/mvp/mvp.yaml"
DATASET="data/processed/mvp_prompts.jsonl"
ACTIVATION_DIR="artifacts/activations/mvp"
TEXT_DIR="artifacts/models/text_baseline"
PROBE_DIR="artifacts/models/activation_probes"
RESULTS_PATH="results/mvp_results.csv"
SCRATCH_DIR=""
MODEL_CACHE="${HOME}/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --project-dir)
      PROJECT_DIR="$2"
      shift 2
      ;;
    --venv-path)
      VENV_PATH="$2"
      shift 2
      ;;
    --toolchain-rc)
      TOOLCHAIN_RC="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --activation-dir)
      ACTIVATION_DIR="$2"
      shift 2
      ;;
    --text-dir)
      TEXT_DIR="$2"
      shift 2
      ;;
    --probe-dir)
      PROBE_DIR="$2"
      shift 2
      ;;
    --results-path)
      RESULTS_PATH="$2"
      shift 2
      ;;
    --scratch-dir)
      SCRATCH_DIR="$2"
      shift 2
      ;;
    --model-cache)
      MODEL_CACHE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

source "$TOOLCHAIN_RC"
source "${VENV_PATH}/bin/activate"

cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1

log "=== MVP pipeline starting — stage: ${STAGE} ==="
log "Node: $(hostname)  Job: ${SLURM_JOB_ID:-local}"
log "Config: ${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read line; do log "GPU: $line"; done

# Model weights: always point to persistent NFS cache (compute nodes have no internet)
export HF_HOME="$MODEL_CACHE"
export HUGGINGFACE_HUB_CACHE="${MODEL_CACHE}/hub"
export TRANSFORMERS_CACHE="${MODEL_CACHE}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Scratch: fast local disk for large intermediate files (activations, processed data)
if [[ -n "$SCRATCH_DIR" ]]; then
  mkdir -p "$SCRATCH_DIR"
fi

run_dataset() {
  log "--- stage: dataset ---"
  python scripts/mvp/build_dataset.py \
    --config "$CONFIG" \
    --output "$DATASET"
  log "--- stage: dataset done ---"
}

run_text() {
  log "--- stage: text baseline ---"
  python scripts/mvp/train_text_baseline.py \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --output-dir "$TEXT_DIR"
  log "--- stage: text baseline done ---"
}

run_activations() {
  log "--- stage: activations ---"
  python scripts/mvp/extract_activations.py \
    --config "$CONFIG" \
    --dataset "$DATASET" \
    --output-dir "$ACTIVATION_DIR"
  log "--- stage: activations done ---"
}

run_probes() {
  log "--- stage: probes ---"
  python scripts/mvp/train_activation_probes.py \
    --config "$CONFIG" \
    --input-dir "$ACTIVATION_DIR" \
    --output-dir "$PROBE_DIR"
  log "--- stage: probes done ---"
}

run_report() {
  log "--- stage: report ---"
  python scripts/mvp/make_results_table.py \
    --config "$CONFIG" \
    --text-metrics "${TEXT_DIR}/metrics.json" \
    --probe-metrics "${PROBE_DIR}/metrics.json" \
    --output "$RESULTS_PATH"
  log "--- stage: report done ---"
}

case "$STAGE" in
  dataset)
    run_dataset
    ;;
  text)
    run_text
    ;;
  activations)
    run_activations
    ;;
  probes)
    run_probes
    ;;
  report)
    run_report
    ;;
  all)
    run_dataset
    run_text
    run_activations
    run_probes
    run_report
    ;;
  *)
    echo "Unsupported stage: $STAGE" >&2
    exit 1
    ;;
esac
