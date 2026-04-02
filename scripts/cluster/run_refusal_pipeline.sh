#!/bin/bash
# Run the refusal-probing pipeline stages.
#
# Stages:
#   responses   - GPU-intensive response generation from refusal prompts
#   judge       - GPU-intensive refusal/compliance labelling
#   relabel     - CPU-only dataset rebuild with judge-assigned refusal labels
#   activations - GPU-intensive hidden-state extraction from the relabelled dataset
#   all         - responses -> judge -> relabel -> activations
#
# Usage:
#   bash scripts/cluster/run_refusal_pipeline.sh --stage responses
#   bash scripts/cluster/run_refusal_pipeline.sh --stage judge
#   bash scripts/cluster/run_refusal_pipeline.sh --stage relabel
#   bash scripts/cluster/run_refusal_pipeline.sh --stage activations
set -euo pipefail

STAGE="responses"
PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/refusal.yaml"
SOURCE_DATASET="data/processed/refusal_prompts.jsonl"
RESPONSES_PATH="artifacts/responses/refusal/responses.jsonl"
LABELLED_RESPONSES_PATH="artifacts/responses/refusal/labelled_responses.jsonl"
RELABELLED_DATASET="data/processed/refusal_labelled.jsonl"
ACTIVATION_DIR="artifacts/activations/refusal"
SCRATCH_DIR=""
MODEL_CACHE="${HOME}/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --toolchain-rc) TOOLCHAIN_RC="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --source-dataset|--dataset) SOURCE_DATASET="$2"; shift 2 ;;
    --responses-path|--responses) RESPONSES_PATH="$2"; shift 2 ;;
    --labelled-responses) LABELLED_RESPONSES_PATH="$2"; shift 2 ;;
    --relabelled-dataset) RELABELLED_DATASET="$2"; shift 2 ;;
    --activation-dir|--output-dir) ACTIVATION_DIR="$2"; shift 2 ;;
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

log "=== Refusal pipeline starting - stage: ${STAGE} ==="
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

run_responses() {
  log "--- stage: responses ---"
  mkdir -p "$(dirname "$RESPONSES_PATH")"
  python scripts/main/generate_responses.py \
    --config "$CONFIG" \
    --dataset "$SOURCE_DATASET" \
    --output "$RESPONSES_PATH"
  log "--- stage: responses done ---"
}

run_judge() {
  log "--- stage: judge ---"
  mkdir -p "$(dirname "$LABELLED_RESPONSES_PATH")"
  python scripts/main/label_refusals.py \
    --config "$CONFIG" \
    --responses "$RESPONSES_PATH" \
    --output "$LABELLED_RESPONSES_PATH"
  log "--- stage: judge done ---"
}

run_relabel() {
  log "--- stage: relabel ---"
  mkdir -p "$(dirname "$RELABELLED_DATASET")"
  python scripts/main/build_refusal_dataset.py \
    --config "$CONFIG" \
    --mode relabel \
    --source-dataset "$SOURCE_DATASET" \
    --labelled-responses "$LABELLED_RESPONSES_PATH" \
    --output "$RELABELLED_DATASET"
  log "--- stage: relabel done ---"
}

run_activations() {
  log "--- stage: activations ---"
  mkdir -p "$ACTIVATION_DIR"
  python scripts/main/extract_activations.py \
    --config "$CONFIG" \
    --dataset "$RELABELLED_DATASET" \
    --output-dir "$ACTIVATION_DIR"
  log "--- stage: activations done ---"
}

case "$STAGE" in
  responses) run_responses ;;
  judge) run_judge ;;
  relabel) run_relabel ;;
  activations) run_activations ;;
  all)
    run_responses
    run_judge
    run_relabel
    run_activations
    ;;
  *)
    echo "Unknown stage: $STAGE (valid: responses, judge, relabel, activations, all)" >&2
    exit 1
    ;;
esac

log "=== Refusal pipeline done - stage: ${STAGE} ==="
