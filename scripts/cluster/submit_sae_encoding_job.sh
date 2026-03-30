#!/bin/bash
# Submit SAE encoding for the main experiment to SLURM.
set -euo pipefail

PARTITION="Teaching"
GPU_TYPE="a40"
TIME_LIMIT="04:00:00"
JOB_NAME="main-sae-encode"
PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/main.yaml"
INPUT_DIR="artifacts/activations/main"
OUTPUT_DIR="artifacts/features/main/sae"
METRICS_DIR="results/main/metrics"
BATCH_SIZE="256"
DEVICE="cuda"
DTYPE="float32"
SCRATCH_DIR="/disk/scratch/${USER}/activation-guardrails"
MODEL_CACHE="${HOME}/models"
PRINT_ONLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --gpu-type) GPU_TYPE="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --toolchain-rc) TOOLCHAIN_RC="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --metrics-dir) METRICS_DIR="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --dtype) DTYPE="$2"; shift 2 ;;
    --scratch-dir) SCRATCH_DIR="$2"; shift 2 ;;
    --model-cache) MODEL_CACHE="$2"; shift 2 ;;
    --print-only) PRINT_ONLY="1"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

SBATCH_CMD=(
  sbatch
  -p "$PARTITION"
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

RUN_ARGS=(
  "--project-dir" "$PROJECT_DIR"
  "--venv-path" "$VENV_PATH"
  "--toolchain-rc" "$TOOLCHAIN_RC"
  "--config" "$CONFIG"
  "--input-dir" "$INPUT_DIR"
  "--output-dir" "$OUTPUT_DIR"
  "--metrics-dir" "$METRICS_DIR"
  "--batch-size" "$BATCH_SIZE"
  "--device" "$DEVICE"
  "--dtype" "$DTYPE"
  "--scratch-dir" "$SCRATCH_DIR"
  "--model-cache" "$MODEL_CACHE"
)

SBATCH_CMD+=(
  --wrap
  "bash ${PROJECT_DIR}/scripts/cluster/run_sae_encoding.sh ${RUN_ARGS[*]}"
)

printf 'Command:\n%s\n' "${SBATCH_CMD[*]}"

if [[ "$PRINT_ONLY" == "1" ]]; then
  exit 0
fi

"${SBATCH_CMD[@]}"
