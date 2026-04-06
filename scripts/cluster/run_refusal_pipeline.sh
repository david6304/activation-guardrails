#!/bin/bash
# Run the refusal-probing pipeline stages.
#
# Stages:
#   source      - CPU-only build of vanilla and optional adversarial source datasets
#   responses   - GPU-intensive response generation from refusal prompts
#   judge       - GPU-intensive refusal/compliance labelling
#   relabel     - CPU-only dataset rebuild with judge-assigned refusal labels
#   activations - GPU-intensive hidden-state extraction from relabelled datasets
#   all         - source -> responses -> judge -> relabel -> activations
#
# Usage:
#   bash scripts/cluster/run_refusal_pipeline.sh --stage source
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
VANILLA_SOURCE_DATASET="data/processed/refusal_prompts.jsonl"
ADVERSARIAL_SOURCE_DATASET="data/processed/refusal_prompts_adversarial.jsonl"
VANILLA_RESPONSES_PATH="artifacts/responses/refusal/responses.jsonl"
ADVERSARIAL_RESPONSES_PATH="artifacts/responses/refusal/responses_adversarial.jsonl"
VANILLA_LABELLED_RESPONSES_PATH="artifacts/responses/refusal/labelled_responses.jsonl"
ADVERSARIAL_LABELLED_RESPONSES_PATH="artifacts/responses/refusal/labelled_responses_adversarial.jsonl"
VANILLA_RELABELLED_DATASET="data/processed/refusal_labelled.jsonl"
ADVERSARIAL_RELABELLED_DATASET="data/processed/refusal_labelled_adversarial.jsonl"
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
    --vanilla-source-dataset|--source-dataset|--dataset) VANILLA_SOURCE_DATASET="$2"; shift 2 ;;
    --adversarial-source-dataset) ADVERSARIAL_SOURCE_DATASET="$2"; shift 2 ;;
    --vanilla-responses-path|--responses-path|--responses) VANILLA_RESPONSES_PATH="$2"; shift 2 ;;
    --adversarial-responses-path) ADVERSARIAL_RESPONSES_PATH="$2"; shift 2 ;;
    --vanilla-labelled-responses|--labelled-responses) VANILLA_LABELLED_RESPONSES_PATH="$2"; shift 2 ;;
    --adversarial-labelled-responses) ADVERSARIAL_LABELLED_RESPONSES_PATH="$2"; shift 2 ;;
    --vanilla-relabelled-dataset|--relabelled-dataset) VANILLA_RELABELLED_DATASET="$2"; shift 2 ;;
    --adversarial-relabelled-dataset) ADVERSARIAL_RELABELLED_DATASET="$2"; shift 2 ;;
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

has_adversarial_source() {
  [[ -f "$ADVERSARIAL_SOURCE_DATASET" ]]
}

has_adversarial_relabelled() {
  [[ -f "$ADVERSARIAL_RELABELLED_DATASET" ]]
}

run_source() {
  log "--- stage: source ---"
  mkdir -p "$(dirname "$VANILLA_SOURCE_DATASET")"
  python scripts/main/build_refusal_dataset.py \
    --config "$CONFIG" \
    --mode source \
    --output "$VANILLA_SOURCE_DATASET" \
    --adversarial-output "$ADVERSARIAL_SOURCE_DATASET"
  log "--- stage: source done ---"
}

run_responses_for_dataset() {
  local dataset_label="$1"
  local dataset_path="$2"
  local responses_path="$3"
  log "--- stage: responses (${dataset_label}) ---"
  mkdir -p "$(dirname "$responses_path")"
  python scripts/main/generate_responses.py \
    --config "$CONFIG" \
    --dataset "$dataset_path" \
    --output "$responses_path"
  log "--- stage: responses (${dataset_label}) done ---"
}

run_responses() {
  run_responses_for_dataset "vanilla" "$VANILLA_SOURCE_DATASET" "$VANILLA_RESPONSES_PATH"
  if has_adversarial_source; then
    run_responses_for_dataset \
      "adversarial" \
      "$ADVERSARIAL_SOURCE_DATASET" \
      "$ADVERSARIAL_RESPONSES_PATH"
  fi
}

run_judge_for_dataset() {
  local dataset_label="$1"
  local responses_path="$2"
  local labelled_path="$3"
  log "--- stage: judge (${dataset_label}) ---"
  mkdir -p "$(dirname "$labelled_path")"
  python scripts/main/label_refusals.py \
    --config "$CONFIG" \
    --responses "$responses_path" \
    --output "$labelled_path"
  log "--- stage: judge (${dataset_label}) done ---"
}

run_judge() {
  run_judge_for_dataset \
    "vanilla" \
    "$VANILLA_RESPONSES_PATH" \
    "$VANILLA_LABELLED_RESPONSES_PATH"
  if has_adversarial_source; then
    run_judge_for_dataset \
      "adversarial" \
      "$ADVERSARIAL_RESPONSES_PATH" \
      "$ADVERSARIAL_LABELLED_RESPONSES_PATH"
  fi
}

run_relabel_for_dataset() {
  local dataset_label="$1"
  local source_dataset="$2"
  local labelled_path="$3"
  local output_dataset="$4"
  log "--- stage: relabel (${dataset_label}) ---"
  mkdir -p "$(dirname "$output_dataset")"
  python scripts/main/build_refusal_dataset.py \
    --config "$CONFIG" \
    --mode relabel \
    --source-dataset "$source_dataset" \
    --labelled-responses "$labelled_path" \
    --output "$output_dataset"
  log "--- stage: relabel (${dataset_label}) done ---"
}

run_relabel() {
  run_relabel_for_dataset \
    "vanilla" \
    "$VANILLA_SOURCE_DATASET" \
    "$VANILLA_LABELLED_RESPONSES_PATH" \
    "$VANILLA_RELABELLED_DATASET"
  if has_adversarial_source; then
    run_relabel_for_dataset \
      "adversarial" \
      "$ADVERSARIAL_SOURCE_DATASET" \
      "$ADVERSARIAL_LABELLED_RESPONSES_PATH" \
      "$ADVERSARIAL_RELABELLED_DATASET"
  fi
}

run_activations() {
  log "--- stage: activations ---"
  mkdir -p "$ACTIVATION_DIR"
  ADV_ARGS=()
  if has_adversarial_relabelled; then
    ADV_ARGS=(--adversarial-dataset "$ADVERSARIAL_RELABELLED_DATASET")
  fi
  python scripts/main/extract_activations.py \
    --config "$CONFIG" \
    --dataset "$VANILLA_RELABELLED_DATASET" \
    --output-dir "$ACTIVATION_DIR" \
    "${ADV_ARGS[@]}"
  log "--- stage: activations done ---"
}

case "$STAGE" in
  source) run_source ;;
  responses) run_responses ;;
  judge) run_judge ;;
  relabel) run_relabel ;;
  activations) run_activations ;;
  all)
    run_source
    run_responses
    run_judge
    run_relabel
    run_activations
    ;;
  *)
    echo "Unknown stage: $STAGE (valid: source, responses, judge, relabel, activations, all)" >&2
    exit 1
    ;;
esac

log "=== Refusal pipeline done - stage: ${STAGE} ==="
