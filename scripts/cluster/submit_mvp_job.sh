#!/bin/bash
set -euo pipefail

# GPU type controls which GRES is requested:
#   a40          → gpu:a40:1            (default; crannog nodes, 48 GB VRAM)
#   a6000        → gpu:nvidia_rtx_a6000:1 (landonia11, 48 GB VRAM)
#   any          → gpu:1 restricted to capable nodes (A40 or A6000)
#   unrestricted → gpu:1 with no node filter (may land on 2080 Ti — not suitable for 7B models)

PARTITION="Teaching"
GPU_TYPE="a40"
TIME_LIMIT="08:00:00"
JOB_NAME="mvp-guardrails"
STAGE="all"
PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/mvp/mvp.yaml"
SCRATCH_DIR="/disk/scratch/${USER}/activation-guardrails"
PRINT_ONLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --gpu-type)
      GPU_TYPE="$2"
      shift 2
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --stage)
      STAGE="$2"
      shift 2
      ;;
    --project-dir)
      PROJECT_DIR="$2"
      shift 2
      ;;
    --venv-path)
      VENV_PATH="$2"
      shift 2
      ;;
    --toolchain-rc)
      TOOLCHAIN_RC="$2"
      shift 2
      ;;
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --scratch-dir)
      SCRATCH_DIR="$2"
      shift 2
      ;;
    --print-only)
      PRINT_ONLY="1"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
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
    # Any GPU with enough VRAM for 7B models (A40 or A6000 nodes only)
    SBATCH_CMD+=(--gres=gpu:1 --nodelist=crannog[01-02],landonia11)
    ;;
  unrestricted)
    # No node filter — may land on a 2080 Ti (11 GB), which will OOM on 7B models
    SBATCH_CMD+=(--gres=gpu:1)
    ;;
  *)
    echo "Unknown --gpu-type: $GPU_TYPE (valid: a40, a6000, any, unrestricted)" >&2
    exit 1
    ;;
esac

SBATCH_CMD+=(
  --wrap
  "bash ${PROJECT_DIR}/scripts/cluster/run_mvp_pipeline.sh --stage ${STAGE} --project-dir ${PROJECT_DIR} --venv-path ${VENV_PATH} --toolchain-rc ${TOOLCHAIN_RC} --config ${CONFIG} --scratch-dir ${SCRATCH_DIR}"
)

printf 'Command:\n%s\n' "${SBATCH_CMD[*]}"

if [[ "$PRINT_ONLY" == "1" ]]; then
  exit 0
fi

"${SBATCH_CMD[@]}"
