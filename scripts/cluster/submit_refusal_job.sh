#!/bin/bash
# Submit refusal-probing GPU stages to SLURM.
#
# Default: response generation on an A40 (48 GB VRAM).
# Source building and relabelling are CPU-only; this wrapper submits those
# stages without a GPU request, while GPU-backed stages stay on A40/A6000.
#
# GPU type options:
#   a40          -> gpu:a40:1              (crannog nodes, 48 GB - default)
#   a6000        -> gpu:nvidia_rtx_a6000:1 (landonia11, 48 GB)
#   any          -> gpu:1 restricted to A40 or A6000 nodes
#   unrestricted -> gpu:1 no node filter (may land on a 2080 Ti and OOM)
#
# Examples:
#   bash scripts/cluster/submit_refusal_job.sh --stage source
#   bash scripts/cluster/submit_refusal_job.sh --stage responses
#   bash scripts/cluster/submit_refusal_job.sh --stage judge
#   bash scripts/cluster/submit_refusal_job.sh --stage activations
#   bash scripts/cluster/submit_refusal_job.sh --stage all --print-only
set -euo pipefail

PARTITION="Teaching"
GPU_TYPE="a40"
TIME_LIMIT=""
JOB_NAME=""
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
SCRATCH_DIR="/disk/scratch/${USER}/activation-guardrails"
MODEL_CACHE="${HOME}/models"
TOKEN_POSITION=""
PRINT_ONLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
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
    --token-position) TOKEN_POSITION="$2"; shift 2 ;;
    --print-only) PRINT_ONLY="1"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

case "$STAGE" in
  source|responses|judge|relabel|activations|all)
    ;;
  *)
    echo "Unknown --stage: $STAGE (valid: source, responses, judge, relabel, activations, all)" >&2
    exit 1
    ;;
esac

GPU_REQUIRED="1"
case "$STAGE" in
  source|relabel)
    GPU_REQUIRED="0"
    ;;
esac

if [[ -z "$TIME_LIMIT" ]]; then
  case "$STAGE" in
    source) TIME_LIMIT="00:15:00" ;;
    responses) TIME_LIMIT="08:00:00" ;;
    judge) TIME_LIMIT="04:00:00" ;;
    relabel) TIME_LIMIT="00:30:00" ;;
    activations) TIME_LIMIT="12:00:00" ;;
    all) TIME_LIMIT="24:00:00" ;;
  esac
fi

if [[ -z "$JOB_NAME" ]]; then
  JOB_NAME="refusal-${STAGE}"
fi

SBATCH_CMD=(
  sbatch
  -p "$PARTITION"
  --nodes=1
  --time="$TIME_LIMIT"
  --job-name="$JOB_NAME"
)

if [[ "$GPU_REQUIRED" == "1" ]]; then
  case "$GPU_TYPE" in
    a40)
      SBATCH_CMD+=(--gres=gpu:a40:1)
      ;;
    a6000)
      SBATCH_CMD+=(--gres=gpu:nvidia_rtx_a6000:1)
      ;;
    any)
      SBATCH_CMD+=(--gres=gpu:1 --nodelist=crannog[01-02],landonia11)
      ;;
    unrestricted)
      SBATCH_CMD+=(--gres=gpu:1)
      ;;
    *)
      echo "Unknown --gpu-type: $GPU_TYPE (valid: a40, a6000, any, unrestricted)" >&2
      exit 1
      ;;
  esac
fi

PIPELINE_ARGS=(
  "--stage" "$STAGE"
  "--project-dir" "$PROJECT_DIR"
  "--venv-path" "$VENV_PATH"
  "--toolchain-rc" "$TOOLCHAIN_RC"
  "--config" "$CONFIG"
  "--source-dataset" "$VANILLA_SOURCE_DATASET"
  "--adversarial-source-dataset" "$ADVERSARIAL_SOURCE_DATASET"
  "--responses-path" "$VANILLA_RESPONSES_PATH"
  "--adversarial-responses-path" "$ADVERSARIAL_RESPONSES_PATH"
  "--labelled-responses" "$VANILLA_LABELLED_RESPONSES_PATH"
  "--adversarial-labelled-responses" "$ADVERSARIAL_LABELLED_RESPONSES_PATH"
  "--relabelled-dataset" "$VANILLA_RELABELLED_DATASET"
  "--adversarial-relabelled-dataset" "$ADVERSARIAL_RELABELLED_DATASET"
  "--activation-dir" "$ACTIVATION_DIR"
  "--scratch-dir" "$SCRATCH_DIR"
  "--model-cache" "$MODEL_CACHE"
)
if [[ -n "$TOKEN_POSITION" ]]; then
  PIPELINE_ARGS+=("--token-position" "$TOKEN_POSITION")
fi

SBATCH_CMD+=(
  --wrap
  "bash ${PROJECT_DIR}/scripts/cluster/run_refusal_pipeline.sh ${PIPELINE_ARGS[*]}"
)

printf 'Command:\n%s\n' "${SBATCH_CMD[*]}"

if [[ "$PRINT_ONLY" == "1" ]]; then
  exit 0
fi

"${SBATCH_CMD[@]}"
