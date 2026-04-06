#!/bin/bash
# Run CPU-only probe training pipeline.
#
# Trains activation probes, SAE probes, the TF-IDF text baseline, and
# optionally Latent Guard using cached activation/feature artifacts, then
# writes a results table.
#
# Usage:
#   bash scripts/cluster/run_probe_training.sh \
#     --config configs/main/refusal.yaml \
#     --activation-dir artifacts/activations/refusal \
#     --feature-dir artifacts/features/refusal/sae \
#     --dataset data/processed/refusal_labelled.jsonl \
#     --adversarial-dataset data/processed/refusal_labelled_adversarial.jsonl \
#     --metrics-dir results/refusal/metrics_cv \
#     --label-key label \
#     --with-latent-guard
set -euo pipefail

PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/refusal.yaml"
ACTIVATION_DIR="artifacts/activations/refusal"
FEATURE_DIR="artifacts/features/refusal/sae"
DATASET="data/processed/refusal_labelled.jsonl"
ADVERSARIAL_DATASET=""
TEXT_MODEL_DIR="artifacts/models/refusal/text_baseline_cv"
PROBE_MODEL_DIR="artifacts/models/refusal/activation_probes_cv"
SAE_MODEL_DIR="artifacts/models/refusal/sae_probes_cv"
LATENT_GUARD_DIR="artifacts/models/refusal/latent_guard_cv"
METRICS_DIR="results/refusal/metrics_cv"
RESULTS_OUTPUT="results/refusal/results_table.csv"
LABEL_KEY="label"
WITH_LATENT_GUARD="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --toolchain-rc) TOOLCHAIN_RC="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --activation-dir) ACTIVATION_DIR="$2"; shift 2 ;;
    --feature-dir|--sae-dir) FEATURE_DIR="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --adversarial-dataset) ADVERSARIAL_DATASET="$2"; shift 2 ;;
    --text-model-dir) TEXT_MODEL_DIR="$2"; shift 2 ;;
    --probe-model-dir) PROBE_MODEL_DIR="$2"; shift 2 ;;
    --sae-model-dir) SAE_MODEL_DIR="$2"; shift 2 ;;
    --latent-guard-dir) LATENT_GUARD_DIR="$2"; shift 2 ;;
    --metrics-dir) METRICS_DIR="$2"; shift 2 ;;
    --results-output) RESULTS_OUTPUT="$2"; shift 2 ;;
    --label-key) LABEL_KEY="$2"; shift 2 ;;
    --with-latent-guard) WITH_LATENT_GUARD="1"; shift ;;
    --without-latent-guard) WITH_LATENT_GUARD="0"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

case "$LABEL_KEY" in
  label|source_label)
    ;;
  *)
    echo "Unknown --label-key: $LABEL_KEY (valid: label, source_label)" >&2
    exit 1
    ;;
esac

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

TEXT_ARGS=()
if [[ -n "$ADVERSARIAL_DATASET" ]]; then
  TEXT_ARGS+=(--adversarial-dataset "$ADVERSARIAL_DATASET")
fi

log "--- stage: activation probes ---"
python scripts/main/train_activation_probes.py \
  --config "$CONFIG" \
  --input-dir "$ACTIVATION_DIR" \
  --output-dir "$PROBE_MODEL_DIR" \
  --metrics-dir "$METRICS_DIR" \
  --label-key "$LABEL_KEY"
log "--- stage: activation probes done ---"

log "--- stage: SAE probes ---"
python scripts/main/train_sae_probes.py \
  --config "$CONFIG" \
  --input-dir "$FEATURE_DIR" \
  --output-dir "$SAE_MODEL_DIR" \
  --metrics-dir "$METRICS_DIR" \
  --label-key "$LABEL_KEY"
log "--- stage: SAE probes done ---"

log "--- stage: text baseline ---"
python scripts/main/train_text_baseline.py \
  --config "$CONFIG" \
  --dataset "$DATASET" \
  --output-dir "$TEXT_MODEL_DIR" \
  --metrics-dir "$METRICS_DIR" \
  --label-key "$LABEL_KEY" \
  "${TEXT_ARGS[@]}"
log "--- stage: text baseline done ---"

if [[ "$WITH_LATENT_GUARD" == "1" ]]; then
  log "--- stage: latent guard ---"
  python scripts/main/train_latent_guard.py \
    --config "$CONFIG" \
    --input-dir "$ACTIVATION_DIR" \
    --output-dir "$LATENT_GUARD_DIR" \
    --metrics-dir "$METRICS_DIR" \
    --label-key "$LABEL_KEY"
  log "--- stage: latent guard done ---"
fi

log "--- stage: results table ---"
RESULTS_ARGS=(
  --config "$CONFIG"
  --text-metrics "${METRICS_DIR}/text_baseline_metrics.json"
  --probe-metrics "${METRICS_DIR}/probe_metrics.json"
  --sae-probe-metrics "${METRICS_DIR}/sae_probe_metrics.json"
  --output "$RESULTS_OUTPUT"
)
if [[ "$WITH_LATENT_GUARD" == "1" ]]; then
  RESULTS_ARGS+=(--latent-guard-metrics "${METRICS_DIR}/latent_guard_metrics.json")
fi
python scripts/main/make_results_table.py "${RESULTS_ARGS[@]}"
log "--- stage: results table done ---"

log "=== Probe training pipeline done ==="
