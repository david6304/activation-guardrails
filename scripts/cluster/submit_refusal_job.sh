#!/bin/bash
# Submit refusal-probing GPU stages to SLURM.
#
# Default: response generation on an A40 (48 GB VRAM).
# The relabel stage is CPU-only; run it directly on the head node after the
# judge stage, or use --stage all to bundle it into the same batch job.
#
# GPU type options:
#   a40          -> gpu:a40:1              (crannog nodes, 48 GB - default)
#   a6000        -> gpu:nvidia_rtx_a6000:1 (landonia11, 48 GB)
#   any          -> gpu:1 restricted to A40 or A6000 nodes
#   unrestricted -> gpu:1 no node filter (may land on a 2080 Ti and OOM)
#
# Examples:
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
SOURCE_DATASET="data/processed/refusal_prompts.jsonl"
RESPONSES_PATH="artifacts/responses/refusal/responses.jsonl"
LABELLED_RESPONSES_PATH="artifacts/responses/refusal/labelled_responses.jsonl"
RELABELLED_DATASET="data/processed/refusal_labelled.jsonl"
ACTIVATION_DIR="artifacts/activations/refusal"
SCRATCH_DIR="/disk/scratch/${USER}/activation-guardrails"
MODEL_CACHE="${HOME}/models"
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
    --source-dataset|--dataset) SOURCE_DATASET="$2"; shift 2 ;;
    --responses-path|--responses) RESPONSES_PATH="$2"; shift 2 ;;
    --labelled-responses) LABELLED_RESPONSES_PATH="$2"; shift 2 ;;
    --relabelled-dataset) RELABELLED_DATASET="$2"; shift 2 ;;
    --activation-dir|--output-dir) ACTIVATION_DIR="$2"; shift 2 ;;
    --scratch-dir) SCRATCH_DIR="$2"; shift 2 ;;
    --model-cache) MODEL_CACHE="$2"; shift 2 ;;
    --print-only) PRINT_ONLY="1"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

case "$STAGE" in
  responses|judge|activations|all)
    ;;
  *)
    echo "Unknown --stage: $STAGE (valid: responses, judge, activations, all)" >&2
    exit 1
    ;;
esac

if [[ -z "$TIME_LIMIT" ]]; then
  case "$STAGE" in
    responses) TIME_LIMIT="06:00:00" ;;
    judge) TIME_LIMIT="02:00:00" ;;
    activations) TIME_LIMIT="12:00:00" ;;
    all) TIME_LIMIT="18:00:00" ;;
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

PIPELINE_ARGS=(
  "--stage" "$STAGE"
  "--project-dir" "$PROJECT_DIR"
  "--venv-path" "$VENV_PATH"
  "--toolchain-rc" "$TOOLCHAIN_RC"
  "--config" "$CONFIG"
  "--source-dataset" "$SOURCE_DATASET"
  "--responses-path" "$RESPONSES_PATH"
  "--labelled-responses" "$LABELLED_RESPONSES_PATH"
  "--relabelled-dataset" "$RELABELLED_DATASET"
  "--activation-dir" "$ACTIVATION_DIR"
  "--scratch-dir" "$SCRATCH_DIR"
  "--model-cache" "$MODEL_CACHE"
)

SBATCH_CMD+=(
  --wrap
  "bash ${PROJECT_DIR}/scripts/cluster/run_refusal_pipeline.sh ${PIPELINE_ARGS[*]}"
)

printf 'Command:\n%s\n' "${SBATCH_CMD[*]}"

if [[ "$PRINT_ONLY" == "1" ]]; then
  exit 0
fi

"${SBATCH_CMD[@]}"
