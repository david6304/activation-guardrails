#!/bin/bash
# Submit one cipher-transfer refusal-probing job to SLURM.
set -euo pipefail

PARTITION="Teaching"
GPU_TYPE="a40"
TIME_LIMIT=""
JOB_NAME=""
STAGE="all"
CIPHER=""
PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/refusal.yaml"
PLAIN_DATASET="data/processed/refusal_prompts.jsonl"
BASE_DIR="artifacts/cipher_transfer"
MODEL_CACHE="${HOME}/models"
PRINT_ONLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --stage) STAGE="$2"; shift 2 ;;
    --cipher) CIPHER="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --toolchain-rc) TOOLCHAIN_RC="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --plain-dataset) PLAIN_DATASET="$2"; shift 2 ;;
    --base-dir) BASE_DIR="$2"; shift 2 ;;
    --model-cache) MODEL_CACHE="$2"; shift 2 ;;
    --print-only) PRINT_ONLY="1"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$CIPHER" ]]; then
  echo "--cipher is required" >&2
  exit 1
fi

case "$STAGE" in
  build|responses|judge|relabel|activations|encode|all)
    ;;
  *)
    echo "Unknown --stage: $STAGE (valid: build, responses, judge, relabel, activations, encode, all)" >&2
    exit 1
    ;;
esac

if [[ -z "$TIME_LIMIT" ]]; then
  case "$STAGE" in
    build|relabel) TIME_LIMIT="00:30:00" ;;
    responses) TIME_LIMIT="06:00:00" ;;
    judge) TIME_LIMIT="02:00:00" ;;
    activations) TIME_LIMIT="06:00:00" ;;
    encode) TIME_LIMIT="06:00:00" ;;
    all) TIME_LIMIT="18:00:00" ;;
  esac
fi

if [[ -z "$JOB_NAME" ]]; then
  JOB_NAME="cipher-${CIPHER}-${STAGE}"
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
  "--cipher" "$CIPHER"
  "--project-dir" "$PROJECT_DIR"
  "--venv-path" "$VENV_PATH"
  "--toolchain-rc" "$TOOLCHAIN_RC"
  "--config" "$CONFIG"
  "--plain-dataset" "$PLAIN_DATASET"
  "--base-dir" "$BASE_DIR"
  "--model-cache" "$MODEL_CACHE"
)

SBATCH_CMD+=(
  --wrap
  "bash ${PROJECT_DIR}/scripts/cluster/run_cipher_transfer_pipeline.sh ${PIPELINE_ARGS[*]}"
)

printf 'Command:\n%s\n' "${SBATCH_CMD[*]}"

if [[ "$PRINT_ONLY" == "1" ]]; then
  exit 0
fi

"${SBATCH_CMD[@]}"
