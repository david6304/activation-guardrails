#!/bin/bash
# Agent-oriented convenience wrapper for the WildJailbreak C1 pipeline.
#
# This wrapper derives all standard artifact paths from a single run tag so an
# agent can map "run WildJailbreak C1 on <gpu> with tag <x>" to one command.
#
# Stages:
#   refusal-source       -> build vanilla + adversarial source datasets (CPU)
#   refusal-responses    -> generate responses for vanilla + adversarial (GPU)
#   refusal-judge        -> judge vanilla + adversarial responses (GPU)
#   refusal-relabel      -> relabel both datasets with refusal labels (CPU)
#   refusal-activations  -> extract vanilla + adversarial activations (GPU)
#   refusal-pipeline     -> source -> responses -> judge -> relabel -> activations
#   sae-cache            -> cache SAE weights on the head node (CPU, direct run)
#   sae-encode           -> encode SAE features from cached activations (GPU)
#   refusal-train        -> train refusal text/dense/SAE/Latent Guard models
#   harmfulness-train    -> train harmfulness text/dense/SAE models on same caches
#   paths                -> print the derived paths only
#
# Examples:
#   bash scripts/cluster/submit_wildjailbreak_c1.sh --stage refusal-pipeline
#   bash scripts/cluster/submit_wildjailbreak_c1.sh --stage sae-encode --gpu-type a6000
#   bash scripts/cluster/submit_wildjailbreak_c1.sh --stage refusal-train --run-tag wjb_c1_a40
#   bash scripts/cluster/submit_wildjailbreak_c1.sh --stage harmfulness-train --print-only
set -euo pipefail

STAGE="refusal-pipeline"
RUN_TAG="wildjailbreak_c1"
CONFIG="configs/main/refusal_wildjailbreak.yaml"
GPU_TYPE="a40"
PARTITION="Teaching"
TIME_LIMIT=""
JOB_NAME=""
PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
MODEL_CACHE="${HOME}/models"
NO_LATENT_GUARD="0"
PRINT_ONLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --run-tag|--tag) RUN_TAG="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --toolchain-rc) TOOLCHAIN_RC="$2"; shift 2 ;;
    --model-cache) MODEL_CACHE="$2"; shift 2 ;;
    --no-latent-guard) NO_LATENT_GUARD="1"; shift ;;
    --print-only) PRINT_ONLY="1"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

case "$STAGE" in
  refusal-source|refusal-responses|refusal-judge|refusal-relabel|refusal-activations|refusal-pipeline|sae-cache|sae-encode|refusal-train|harmfulness-train|paths)
    ;;
  *)
    echo "Unknown --stage: $STAGE" >&2
    exit 1
    ;;
esac

DATA_ROOT="data/processed/${RUN_TAG}"
RESPONSES_ROOT="artifacts/responses/${RUN_TAG}"
ACTIVATION_DIR="artifacts/activations/${RUN_TAG}"
SAE_DIR="artifacts/features/${RUN_TAG}/sae"
SHARED_RESULTS_ROOT="results/refusal/${RUN_TAG}/shared"
REFUSAL_MODEL_ROOT="artifacts/models/${RUN_TAG}/refusal"
REFUSAL_RESULTS_ROOT="results/refusal/${RUN_TAG}/refusal"
HARMFULNESS_MODEL_ROOT="artifacts/models/${RUN_TAG}/harmfulness"
HARMFULNESS_RESULTS_ROOT="results/refusal/${RUN_TAG}/harmfulness"

VANILLA_SOURCE_DATASET="${DATA_ROOT}/refusal_prompts.jsonl"
ADVERSARIAL_SOURCE_DATASET="${DATA_ROOT}/refusal_prompts_adversarial.jsonl"
VANILLA_RESPONSES_PATH="${RESPONSES_ROOT}/responses.jsonl"
ADVERSARIAL_RESPONSES_PATH="${RESPONSES_ROOT}/responses_adversarial.jsonl"
VANILLA_LABELLED_RESPONSES_PATH="${RESPONSES_ROOT}/labelled_responses.jsonl"
ADVERSARIAL_LABELLED_RESPONSES_PATH="${RESPONSES_ROOT}/labelled_responses_adversarial.jsonl"
VANILLA_RELABELLED_DATASET="${DATA_ROOT}/refusal_labelled.jsonl"
ADVERSARIAL_RELABELLED_DATASET="${DATA_ROOT}/refusal_labelled_adversarial.jsonl"

default_job_name() {
  case "$STAGE" in
    refusal-source) echo "${RUN_TAG}-refusal-source" ;;
    refusal-responses) echo "${RUN_TAG}-refusal-responses" ;;
    refusal-judge) echo "${RUN_TAG}-refusal-judge" ;;
    refusal-relabel) echo "${RUN_TAG}-refusal-relabel" ;;
    refusal-activations) echo "${RUN_TAG}-refusal-activations" ;;
    refusal-pipeline) echo "${RUN_TAG}-refusal-pipeline" ;;
    sae-encode) echo "${RUN_TAG}-sae-encode" ;;
    refusal-train) echo "${RUN_TAG}-refusal-train" ;;
    harmfulness-train) echo "${RUN_TAG}-harmfulness-train" ;;
    *)
      echo ""
      ;;
  esac
}

append_common_submit_args() {
  CMD+=(--partition "$PARTITION")
  if [[ -n "$TIME_LIMIT" ]]; then
    CMD+=(--time "$TIME_LIMIT")
  fi
  if [[ -n "$JOB_NAME" ]]; then
    CMD+=(--job-name "$JOB_NAME")
  else
    local auto_job_name
    auto_job_name="$(default_job_name)"
    if [[ -n "$auto_job_name" ]]; then
      CMD+=(--job-name "$auto_job_name")
    fi
  fi
}

print_paths() {
  cat <<EOF
run_tag=${RUN_TAG}
config=${CONFIG}
project_dir=${PROJECT_DIR}
data_root=${DATA_ROOT}
responses_root=${RESPONSES_ROOT}
activation_dir=${ACTIVATION_DIR}
sae_dir=${SAE_DIR}
shared_results_root=${SHARED_RESULTS_ROOT}
refusal_model_root=${REFUSAL_MODEL_ROOT}
refusal_results_root=${REFUSAL_RESULTS_ROOT}
harmfulness_model_root=${HARMFULNESS_MODEL_ROOT}
harmfulness_results_root=${HARMFULNESS_RESULTS_ROOT}
vanilla_source_dataset=${VANILLA_SOURCE_DATASET}
adversarial_source_dataset=${ADVERSARIAL_SOURCE_DATASET}
vanilla_responses_path=${VANILLA_RESPONSES_PATH}
adversarial_responses_path=${ADVERSARIAL_RESPONSES_PATH}
vanilla_labelled_responses_path=${VANILLA_LABELLED_RESPONSES_PATH}
adversarial_labelled_responses_path=${ADVERSARIAL_LABELLED_RESPONSES_PATH}
vanilla_relabelled_dataset=${VANILLA_RELABELLED_DATASET}
adversarial_relabelled_dataset=${ADVERSARIAL_RELABELLED_DATASET}
EOF
}

print_command() {
  printf 'Command:\n'
  printf '%q ' "$@"
  printf '\n'
}

run_or_print() {
  print_command "$@"
  if [[ "$PRINT_ONLY" == "1" ]]; then
    return 0
  fi
  "$@"
}

build_refusal_submit_command() {
  local refusal_stage="$1"
  CMD=(
    bash "${PROJECT_DIR}/scripts/cluster/submit_refusal_job.sh"
    --stage "$refusal_stage"
    --project-dir "$PROJECT_DIR"
    --venv-path "$VENV_PATH"
    --toolchain-rc "$TOOLCHAIN_RC"
    --config "$CONFIG"
    --gpu-type "$GPU_TYPE"
    --source-dataset "$VANILLA_SOURCE_DATASET"
    --adversarial-source-dataset "$ADVERSARIAL_SOURCE_DATASET"
    --responses-path "$VANILLA_RESPONSES_PATH"
    --adversarial-responses-path "$ADVERSARIAL_RESPONSES_PATH"
    --labelled-responses "$VANILLA_LABELLED_RESPONSES_PATH"
    --adversarial-labelled-responses "$ADVERSARIAL_LABELLED_RESPONSES_PATH"
    --relabelled-dataset "$VANILLA_RELABELLED_DATASET"
    --adversarial-relabelled-dataset "$ADVERSARIAL_RELABELLED_DATASET"
    --activation-dir "$ACTIVATION_DIR"
    --model-cache "$MODEL_CACHE"
  )
  append_common_submit_args
  if [[ "$PRINT_ONLY" == "1" ]]; then
    CMD+=(--print-only)
  fi
}

build_probe_training_submit_command() {
  local label_key="$1"
  local model_root="$2"
  local results_root="$3"
  local include_latent_guard="$4"
  CMD=(
    bash "${PROJECT_DIR}/scripts/cluster/submit_probe_training_job.sh"
    --project-dir "$PROJECT_DIR"
    --venv-path "$VENV_PATH"
    --toolchain-rc "$TOOLCHAIN_RC"
    --config "$CONFIG"
    --activation-dir "$ACTIVATION_DIR"
    --feature-dir "$SAE_DIR"
    --dataset "$VANILLA_RELABELLED_DATASET"
    --adversarial-dataset "$ADVERSARIAL_RELABELLED_DATASET"
    --text-model-dir "${model_root}/text_baseline"
    --probe-model-dir "${model_root}/activation_probes"
    --sae-model-dir "${model_root}/sae_probes"
    --latent-guard-dir "${model_root}/latent_guard"
    --metrics-dir "${results_root}/metrics"
    --results-output "${results_root}/results_table.csv"
    --label-key "$label_key"
  )
  if [[ "$include_latent_guard" == "1" ]]; then
    CMD+=(--with-latent-guard)
  fi
  append_common_submit_args
  if [[ "$PRINT_ONLY" == "1" ]]; then
    CMD+=(--print-only)
  fi
}

print_paths

case "$STAGE" in
  paths)
    ;;
  refusal-source)
    build_refusal_submit_command "source"
    run_or_print "${CMD[@]}"
    ;;
  refusal-responses)
    build_refusal_submit_command "responses"
    run_or_print "${CMD[@]}"
    ;;
  refusal-judge)
    build_refusal_submit_command "judge"
    run_or_print "${CMD[@]}"
    ;;
  refusal-relabel)
    build_refusal_submit_command "relabel"
    run_or_print "${CMD[@]}"
    ;;
  refusal-activations)
    build_refusal_submit_command "activations"
    run_or_print "${CMD[@]}"
    ;;
  refusal-pipeline)
    build_refusal_submit_command "all"
    run_or_print "${CMD[@]}"
    ;;
  sae-cache)
    CMD=(
      bash "${PROJECT_DIR}/scripts/cluster/run_sae_pipeline.sh"
      --stage cache
      --project-dir "$PROJECT_DIR"
      --venv-path "$VENV_PATH"
      --toolchain-rc "$TOOLCHAIN_RC"
      --config "$CONFIG"
      --model-cache "$MODEL_CACHE"
    )
    run_or_print "${CMD[@]}"
    ;;
  sae-encode)
    CMD=(
      bash "${PROJECT_DIR}/scripts/cluster/submit_sae_job.sh"
      --stage encode
      --project-dir "$PROJECT_DIR"
      --venv-path "$VENV_PATH"
      --toolchain-rc "$TOOLCHAIN_RC"
      --config "$CONFIG"
      --gpu-type "$GPU_TYPE"
      --input-dir "$ACTIVATION_DIR"
      --feature-dir "$SAE_DIR"
      --metrics-dir "$SHARED_RESULTS_ROOT"
      --model-cache "$MODEL_CACHE"
    )
    append_common_submit_args
    if [[ "$PRINT_ONLY" == "1" ]]; then
      CMD+=(--print-only)
    fi
    run_or_print "${CMD[@]}"
    ;;
  refusal-train)
    build_probe_training_submit_command \
      "label" \
      "$REFUSAL_MODEL_ROOT" \
      "$REFUSAL_RESULTS_ROOT" \
      "$([[ "$NO_LATENT_GUARD" == "1" ]] && echo 0 || echo 1)"
    run_or_print "${CMD[@]}"
    ;;
  harmfulness-train)
    build_probe_training_submit_command \
      "source_label" \
      "$HARMFULNESS_MODEL_ROOT" \
      "$HARMFULNESS_RESULTS_ROOT" \
      "0"
    run_or_print "${CMD[@]}"
    ;;
esac
