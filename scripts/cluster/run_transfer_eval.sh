#!/bin/bash
# Run CPU-only transfer evaluation pipeline.
#
# Applies frozen plain-text probes to cached cipher activations and writes
# per-cipher transfer metrics + a combined results table.
#
# Usage:
#   bash scripts/cluster/run_transfer_eval.sh \
#     --config configs/main/refusal_large_train.yaml \
#     --probe-dir artifacts/models/refusal/activation_probes_cv \
#     --sae-probe-dir artifacts/models/refusal/sae_probes_cv \
#     --plain-metrics-dir results/refusal/metrics_cv_large_train \
#     --cipher-base-dir artifacts/cipher_transfer \
#     --output-dir results/refusal/cipher_transfer_large_train
set -euo pipefail

PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/refusal_large_train.yaml"
PROBE_DIR="artifacts/models/refusal/activation_probes_cv"
SAE_PROBE_DIR="artifacts/models/refusal/sae_probes_cv"
TEXT_PIPELINE="artifacts/models/main/text_baseline/tfidf_lr_pipeline.joblib"
PLAIN_METRICS_DIR="results/refusal/metrics_cv_large_train"
CIPHER_BASE_DIR="artifacts/cipher_transfer"
OUTPUT_DIR="results/refusal/cipher_transfer_large_train"
CIPHERS="reverse rot13 rot9"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --toolchain-rc) TOOLCHAIN_RC="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --probe-dir) PROBE_DIR="$2"; shift 2 ;;
    --sae-probe-dir) SAE_PROBE_DIR="$2"; shift 2 ;;
    --text-pipeline) TEXT_PIPELINE="$2"; shift 2 ;;
    --plain-metrics-dir) PLAIN_METRICS_DIR="$2"; shift 2 ;;
    --cipher-base-dir) CIPHER_BASE_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --ciphers) CIPHERS="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

source "$TOOLCHAIN_RC"
source "${VENV_PATH}/bin/activate"

cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

log "=== Transfer eval pipeline starting ==="
log "Node: $(hostname)  Job: ${SLURM_JOB_ID:-local}"

for CIPHER in $CIPHERS; do
  log "--- cipher: ${CIPHER} ---"
  CIPHER_OUT="${OUTPUT_DIR}/${CIPHER}"
  mkdir -p "$CIPHER_OUT"

  CIPHER_ACTIVATIONS="${CIPHER_BASE_DIR}/${CIPHER}/activations"
  CIPHER_SAE="${CIPHER_BASE_DIR}/${CIPHER}/sae"
  CIPHER_DATASET="${CIPHER_BASE_DIR}/${CIPHER}/data/${CIPHER}_labelled.jsonl"

  python scripts/main/eval_activation_transfer.py \
    --input-dir "$CIPHER_ACTIVATIONS" \
    --probe-dir "$PROBE_DIR" \
    --plain-metrics "${PLAIN_METRICS_DIR}/probe_metrics.json" \
    --output "${CIPHER_OUT}/activation_transfer_metrics.json" \
    --eval-name "$CIPHER"

  python scripts/main/eval_sae_transfer.py \
    --input-dir "$CIPHER_SAE" \
    --probe-dir "$SAE_PROBE_DIR" \
    --plain-metrics "${PLAIN_METRICS_DIR}/sae_probe_metrics.json" \
    --output "${CIPHER_OUT}/sae_transfer_metrics.json" \
    --eval-name "$CIPHER"

  python scripts/main/eval_text_transfer.py \
    --dataset "$CIPHER_DATASET" \
    --pipeline-path "$TEXT_PIPELINE" \
    --plain-metrics "${PLAIN_METRICS_DIR}/text_baseline_metrics.json" \
    --output "${CIPHER_OUT}/text_transfer_metrics.json" \
    --eval-name "$CIPHER"

  log "--- cipher: ${CIPHER} done ---"
done

log "--- stage: results table ---"
python scripts/main/make_cipher_transfer_results_table.py \
  --text-metrics "${PLAIN_METRICS_DIR}/text_baseline_metrics.json" \
  --probe-metrics "${PLAIN_METRICS_DIR}/probe_metrics.json" \
  --sae-probe-metrics "${PLAIN_METRICS_DIR}/sae_probe_metrics.json" \
  --transfer-metrics-dir "$OUTPUT_DIR" \
  --output "${OUTPUT_DIR}/results_table.csv"
log "--- stage: results table done ---"

log "=== Transfer eval pipeline done ==="
