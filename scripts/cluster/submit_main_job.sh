#!/bin/bash
# Submit the main experiment pipeline to SLURM.
#
# Default: activations-only stage on an A40 (48 GB VRAM).
# Text baseline and probes are CPU-only — run them on the head node
# after the job completes.
#
# GPU type options:
#   a40          → gpu:a40:1            (crannog nodes, 48 GB — default)
#   a6000        → gpu:nvidia_rtx_a6000:1 (landonia11, 48 GB)
#   any          → gpu:1 restricted to A40 or A6000 nodes
#   unrestricted → gpu:1 no node filter (may land on 2080 Ti — will OOM)
#
# Examples:
#   # Activations only (default)
#   bash scripts/cluster/submit_main_job.sh
#
#   # Activations + adversarial in the same GPU job
#   bash scripts/cluster/submit_main_job.sh \
#       --adversarial-dataset data/processed/main_adversarial.jsonl
#
#   # Full pipeline in one job (GPU required for all stages)
#   bash scripts/cluster/submit_main_job.sh --stage all
#
#   # Dry run — print sbatch command without submitting
#   bash scripts/cluster/submit_main_job.sh --print-only
set -euo pipefail

PARTITION="Teaching"
GPU_TYPE="a40"
TIME_LIMIT="12:00:00"
JOB_NAME="main-guardrails"
STAGE="activations"
PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/main.yaml"
DATASET="data/processed/main_prompts.jsonl"
ADVERSARIAL_DATASET=""
ACTIVATION_DIR="artifacts/activations/main"
TEXT_DIR="artifacts/models/main/text_baseline"
PROBE_DIR="artifacts/models/main/activation_probes"
METRICS_DIR="results/main/metrics"
SCRATCH_DIR="/disk/scratch/${USER}/activation-guardrails"
MODEL_CACHE="${HOME}/models"
PRINT_ONLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition)           PARTITION="$2";            shift 2 ;;
    --gpu-type)            GPU_TYPE="$2";             shift 2 ;;
    --time)                TIME_LIMIT="$2";           shift 2 ;;
    --job-name)            JOB_NAME="$2";             shift 2 ;;
    --stage)               STAGE="$2";                shift 2 ;;
    --project-dir)         PROJECT_DIR="$2";          shift 2 ;;
    --venv-path)           VENV_PATH="$2";            shift 2 ;;
    --toolchain-rc)        TOOLCHAIN_RC="$2";         shift 2 ;;
    --config)              CONFIG="$2";               shift 2 ;;
    --dataset)             DATASET="$2";              shift 2 ;;
    --adversarial-dataset) ADVERSARIAL_DATASET="$2";  shift 2 ;;
    --activation-dir)      ACTIVATION_DIR="$2";       shift 2 ;;
    --text-dir)            TEXT_DIR="$2";             shift 2 ;;
    --probe-dir)           PROBE_DIR="$2";            shift 2 ;;
    --metrics-dir)         METRICS_DIR="$2";          shift 2 ;;
    --scratch-dir)         SCRATCH_DIR="$2";          shift 2 ;;
    --model-cache)         MODEL_CACHE="$2";          shift 2 ;;
    --print-only)          PRINT_ONLY="1";            shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

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
  "--stage"           "$STAGE"
  "--project-dir"     "$PROJECT_DIR"
  "--venv-path"       "$VENV_PATH"
  "--toolchain-rc"    "$TOOLCHAIN_RC"
  "--config"          "$CONFIG"
  "--dataset"         "$DATASET"
  "--activation-dir"  "$ACTIVATION_DIR"
  "--text-dir"        "$TEXT_DIR"
  "--probe-dir"       "$PROBE_DIR"
  "--metrics-dir"     "$METRICS_DIR"
  "--scratch-dir"     "$SCRATCH_DIR"
  "--model-cache"     "$MODEL_CACHE"
)
if [[ -n "$ADVERSARIAL_DATASET" ]]; then
  PIPELINE_ARGS+=("--adversarial-dataset" "$ADVERSARIAL_DATASET")
fi

SBATCH_CMD+=(
  --wrap
  "bash ${PROJECT_DIR}/scripts/cluster/run_main_pipeline.sh ${PIPELINE_ARGS[*]}"
)

printf 'Command:\n%s\n' "${SBATCH_CMD[*]}"

if [[ "$PRINT_ONLY" == "1" ]]; then
  exit 0
fi

"${SBATCH_CMD[@]}"
