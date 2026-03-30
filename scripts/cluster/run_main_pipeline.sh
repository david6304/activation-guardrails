#!/bin/bash
# Run the main experiment pipeline stages.
#
# Stages:
#   activations  — GPU-intensive; run via sbatch (submit_main_job.sh)
#   text         — CPU-only; run on head node after activations are done
#   probes       — CPU-only; run on head node after activations are done
#   all          — activations → text → probes (GPU job covers everything)
#
# Usage:
#   bash scripts/cluster/run_main_pipeline.sh --stage activations
#   bash scripts/cluster/run_main_pipeline.sh --stage text
#   bash scripts/cluster/run_main_pipeline.sh --stage probes
set -euo pipefail

STAGE="activations"
PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/main.yaml"
DATASET="data/processed/main_prompts.jsonl"
ADVERSARIAL_DATASET=""           # empty = skip adversarial extraction
ACTIVATION_DIR="artifacts/activations/main"
TEXT_DIR="artifacts/models/main/text_baseline"
PROBE_DIR="artifacts/models/main/activation_probes"
METRICS_DIR="results/main/metrics"
SCRATCH_DIR=""
MODEL_CACHE="${HOME}/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)           STAGE="$2";                 shift 2 ;;
    --project-dir)     PROJECT_DIR="$2";           shift 2 ;;
    --venv-path)       VENV_PATH="$2";             shift 2 ;;
    --toolchain-rc)    TOOLCHAIN_RC="$2";          shift 2 ;;
    --config)          CONFIG="$2";                shift 2 ;;
    --dataset)         DATASET="$2";               shift 2 ;;
    --adversarial-dataset) ADVERSARIAL_DATASET="$2"; shift 2 ;;
    --activation-dir)  ACTIVATION_DIR="$2";        shift 2 ;;
    --text-dir)        TEXT_DIR="$2";              shift 2 ;;
    --probe-dir)       PROBE_DIR="$2";             shift 2 ;;
    --metrics-dir)     METRICS_DIR="$2";           shift 2 ;;
    --scratch-dir)     SCRATCH_DIR="$2";           shift 2 ;;
    --model-cache)     MODEL_CACHE="$2";           shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

source "$TOOLCHAIN_RC"
source "${VENV_PATH}/bin/activate"

cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1

log "=== Main pipeline starting — stage: ${STAGE} ==="
log "Node: $(hostname)  Job: ${SLURM_JOB_ID:-local}"
log "Config: ${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  | while read -r line; do log "GPU: $line"; done || true

# Model weights: persistent NFS cache — compute nodes have no internet
export HF_HOME="$MODEL_CACHE"
export HUGGINGFACE_HUB_CACHE="${MODEL_CACHE}/hub"
export TRANSFORMERS_CACHE="${MODEL_CACHE}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ -n "$SCRATCH_DIR" ]]; then
  mkdir -p "$SCRATCH_DIR"
fi

run_activations() {
  log "--- stage: activations ---"
  ADV_ARGS=()
  if [[ -n "$ADVERSARIAL_DATASET" ]]; then
    ADV_ARGS=(--adversarial-dataset "$ADVERSARIAL_DATASET")
  fi
  python scripts/main/extract_activations.py \
    --config    "$CONFIG" \
    --dataset   "$DATASET" \
    --output-dir "$ACTIVATION_DIR" \
    "${ADV_ARGS[@]}"
  log "--- stage: activations done ---"
}

run_text() {
  log "--- stage: text baseline ---"
  python scripts/main/train_text_baseline.py \
    --config      "$CONFIG" \
    --dataset     "$DATASET" \
    --output-dir  "$TEXT_DIR" \
    --metrics-dir "$METRICS_DIR"
  log "--- stage: text baseline done ---"
}

run_probes() {
  log "--- stage: probes ---"
  python scripts/main/train_activation_probes.py \
    --config      "$CONFIG" \
    --input-dir   "$ACTIVATION_DIR" \
    --output-dir  "$PROBE_DIR" \
    --metrics-dir "$METRICS_DIR"
  log "--- stage: probes done ---"
}

case "$STAGE" in
  activations) run_activations ;;
  text)        run_text ;;
  probes)      run_probes ;;
  all)
    run_activations
    run_text
    run_probes
    ;;
  *)
    echo "Unknown stage: $STAGE (valid: activations, text, probes, all)" >&2
    exit 1
    ;;
esac

log "=== Main pipeline done — stage: ${STAGE} ==="
