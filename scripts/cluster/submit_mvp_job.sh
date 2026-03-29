#!/bin/bash
set -euo pipefail

PARTITION="Teaching"
GPU_GRES="gpu:1"
GPU_CONSTRAINT="a40"
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
    --gpu-gres)
      GPU_GRES="$2"
      shift 2
      ;;
    --gpu-type)
      GPU_CONSTRAINT="$2"
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

if [[ -n "$GPU_GRES" && "$GPU_GRES" != "none" ]]; then
  SBATCH_CMD+=(--gres="$GPU_GRES")
fi

if [[ -n "$GPU_GRES" && "$GPU_GRES" != "none" && -n "$GPU_CONSTRAINT" && "$GPU_CONSTRAINT" != "any" ]]; then
  SBATCH_CMD+=(--constraint="$GPU_CONSTRAINT")
fi

SBATCH_CMD+=(
  --wrap
  "bash ${PROJECT_DIR}/scripts/cluster/run_mvp_pipeline.sh --stage ${STAGE} --project-dir ${PROJECT_DIR} --venv-path ${VENV_PATH} --toolchain-rc ${TOOLCHAIN_RC} --config ${CONFIG} --scratch-dir ${SCRATCH_DIR}"
)

printf 'Command:\n%s\n' "${SBATCH_CMD[*]}"

if [[ "$PRINT_ONLY" == "1" ]]; then
  exit 0
fi

"${SBATCH_CMD[@]}"
