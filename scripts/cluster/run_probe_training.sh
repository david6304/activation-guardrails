#!/bin/bash
# Run CPU-only probe training pipeline.
#
# Trains activation probes, SAE probes, and the TF-IDF text baseline using
# cached activation/feature artifacts, then writes the results table.
#
# Usage:
#   bash scripts/cluster/run_probe_training.sh \
#     --config configs/main/refusal.yaml \
#     --activation-dir artifacts/activations/refusal \
#     --sae-dir artifacts/features/refusal/sae \
#     --dataset data/processed/refusal_labelled.jsonl \
#     --metrics-dir results/refusal/metrics_cv
set -euo pipefail

PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/refusal.yaml"
ACTIVATION_DIR="artifacts/activations/refusal"
SAE_DIR="artifacts/features/refusal/sae"
DATASET="data/processed/refusal_labelled.jsonl"
PROBE_MODEL_DIR="artifacts/models/refusal/activation_probes_cv"
SAE_MODEL_DIR="artifacts/models/refusal/sae_probes_cv"
METRICS_DIR="results/refusal/metrics_cv"
RESULTS_OUTPUT="results/refusal/results_table.csv"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --toolchain-rc) TOOLCHAIN_RC="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --activation-dir) ACTIVATION_DIR="$2"; shift 2 ;;
    --sae-dir) SAE_DIR="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --probe-model-dir) PROBE_MODEL_DIR="$2"; shift 2 ;;
    --sae-model-dir) SAE_MODEL_DIR="$2"; shift 2 ;;
    --metrics-dir) METRICS_DIR="$2"; shift 2 ;;
    --results-output) RESULTS_OUTPUT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

source "$TOOLCHAIN_RC"
source "${VENV_PATH}/bin/activate"

cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1

log "=== Probe training pipeline starting ==="
log "Node: $(hostname)  Job: ${SLURM_JOB_ID:-local}"
log "Config: ${CONFIG}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

log "--- stage: activation probes ---"
python scripts/main/train_activation_probes.py \
  --config "$CONFIG" \
  --input-dir "$ACTIVATION_DIR" \
  --output-dir "$PROBE_MODEL_DIR" \
  --metrics-dir "$METRICS_DIR"
log "--- stage: activation probes done ---"

log "--- stage: SAE probes ---"
python scripts/main/train_sae_probes.py \
  --config "$CONFIG" \
  --input-dir "$SAE_DIR" \
  --output-dir "$SAE_MODEL_DIR" \
  --metrics-dir "$METRICS_DIR"
log "--- stage: SAE probes done ---"

log "--- stage: text baseline ---"
python scripts/main/train_text_baseline.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --metrics-dir "$METRICS_DIR"
log "--- stage: text baseline done ---"

log "--- stage: results table ---"
python scripts/main/make_results_table.py \
  --config "$CONFIG" \
  --text-metrics "${METRICS_DIR}/text_baseline_metrics.json" \
  --probe-metrics "${METRICS_DIR}/probe_metrics.json" \
  --sae-probe-metrics "${METRICS_DIR}/sae_probe_metrics.json" \
  --output "$RESULTS_OUTPUT"
log "--- stage: results table done ---"

log "=== Probe training pipeline done ==="
