#!/bin/bash
# Run one cipher-transfer pipeline for refusal probing.
#
# Stages:
#   build       - build ciphered test prompts from the plain-text prompt set
#   responses   - generate model responses on cipher prompts
#   judge       - label cipher responses as refusal/compliance
#   relabel     - rebuild the cipher dataset with judge-assigned refusal labels
#   activations - extract dense activations from the relabelled cipher dataset
#   encode      - encode cached activations with SAEs
#   all         - build -> responses -> judge -> relabel -> activations -> encode
set -euo pipefail

STAGE="all"
CIPHER=""
PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/refusal.yaml"
PLAIN_DATASET="data/processed/refusal_prompts.jsonl"
BASE_DIR="artifacts/cipher_transfer"
MODEL_CACHE="${HOME}/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --cipher) CIPHER="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --toolchain-rc) TOOLCHAIN_RC="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --plain-dataset) PLAIN_DATASET="$2"; shift 2 ;;
    --base-dir) BASE_DIR="$2"; shift 2 ;;
    --model-cache) MODEL_CACHE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$CIPHER" ]]; then
  echo "--cipher is required" >&2
  exit 1
fi

DATA_DIR="${BASE_DIR}/${CIPHER}/data"
RESPONSES_DIR="${BASE_DIR}/${CIPHER}/responses"
ACTIVATIONS_DIR="${BASE_DIR}/${CIPHER}/activations"
FEATURES_DIR="${BASE_DIR}/${CIPHER}/sae"
METRICS_DIR="results/refusal/cipher_transfer/${CIPHER}"

CIPHER_DATASET="${DATA_DIR}/${CIPHER}_prompts.jsonl"
LABELLED_RESPONSES="${RESPONSES_DIR}/labelled_responses.jsonl"
RESPONSES_PATH="${RESPONSES_DIR}/responses.jsonl"
RELABELLED_DATASET="${DATA_DIR}/${CIPHER}_labelled.jsonl"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

source "$TOOLCHAIN_RC"
source "${VENV_PATH}/bin/activate"

cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1
export HF_HOME="$MODEL_CACHE"
export HUGGINGFACE_HUB_CACHE="${MODEL_CACHE}/hub"
export TRANSFORMERS_CACHE="${MODEL_CACHE}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_build() {
  mkdir -p "$DATA_DIR"
  python scripts/main/build_cipher_dataset.py \
    --dataset "$PLAIN_DATASET" \
    --cipher "$CIPHER" \
    --split test \
    --output "$CIPHER_DATASET"
}

run_responses() {
  mkdir -p "$RESPONSES_DIR"
  python scripts/main/generate_responses.py \
    --config "$CONFIG" \
    --dataset "$CIPHER_DATASET" \
    --output "$RESPONSES_PATH"
}

run_judge() {
  mkdir -p "$RESPONSES_DIR"
  python scripts/main/label_refusals.py \
    --config "$CONFIG" \
    --responses "$RESPONSES_PATH" \
    --output "$LABELLED_RESPONSES"
}

run_relabel() {
  mkdir -p "$DATA_DIR"
  python scripts/main/build_refusal_dataset.py \
    --config "$CONFIG" \
    --mode relabel \
    --source-dataset "$CIPHER_DATASET" \
    --labelled-responses "$LABELLED_RESPONSES" \
    --output "$RELABELLED_DATASET"
}

run_activations() {
  mkdir -p "$ACTIVATIONS_DIR"
  python scripts/main/extract_activations.py \
    --config "$CONFIG" \
    --dataset "$RELABELLED_DATASET" \
    --output-dir "$ACTIVATIONS_DIR" \
    --splits test
}

run_encode() {
  mkdir -p "$FEATURES_DIR" "$METRICS_DIR"
  python scripts/main/encode_sae_features.py \
    --config "$CONFIG" \
    --input-dir "$ACTIVATIONS_DIR" \
    --output-dir "$FEATURES_DIR" \
    --metrics-dir "$METRICS_DIR" \
    --splits test \
    --device cuda
}

case "$STAGE" in
  build) run_build ;;
  responses) run_responses ;;
  judge) run_judge ;;
  relabel) run_relabel ;;
  activations) run_activations ;;
  encode) run_encode ;;
  all)
    run_build
    run_responses
    run_judge
    run_relabel
    run_activations
    run_encode
    ;;
  *)
    echo "Unknown stage: $STAGE (valid: build, responses, judge, relabel, activations, encode, all)" >&2
    exit 1
    ;;
esac

log "Cipher-transfer pipeline complete for cipher=${CIPHER} stage=${STAGE}"
