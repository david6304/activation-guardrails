#!/bin/bash
# Submit CPU-only transfer evaluation to SLURM.
#
# Applies frozen plain-text probes to cached cipher activations for all
# ciphers and writes the combined results table.
#
# Examples:
#   bash scripts/cluster/submit_transfer_eval_job.sh
#   bash scripts/cluster/submit_transfer_eval_job.sh \
#     --config configs/main/refusal_large_train.yaml \
#     --probe-dir artifacts/models/refusal_large_train/activation_probes_cv \
#     --sae-probe-dir artifacts/models/refusal_large_train/sae_probes_cv \
#     --plain-metrics-dir results/refusal/metrics_cv_large_train \
#     --output-dir results/refusal/cipher_transfer_large_train
#   bash scripts/cluster/submit_transfer_eval_job.sh --print-only
set -euo pipefail

PARTITION="Teaching"
TIME_LIMIT="00:30:00"
JOB_NAME="transfer-eval"
PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/refusal_large_train.yaml"
PROBE_DIR="artifacts/models/refusal_large_train/activation_probes_cv"
SAE_PROBE_DIR="artifacts/models/refusal_large_train/sae_probes_cv"
TEXT_PIPELINE="artifacts/models/main/text_baseline/tfidf_lr_pipeline.joblib"
PLAIN_METRICS_DIR="results/refusal/metrics_cv_large_train"
CIPHER_BASE_DIR="artifacts/cipher_transfer"
OUTPUT_DIR="results/refusal/cipher_transfer_large_train"
CIPHERS="reverse rot13 rot9"
PRINT_ONLY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2 ;;
    --time) TIME_LIMIT="$2"; shift 2 ;;
    --job-name) JOB_NAME="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --toolchain-rc) TOOLCHAIN_RC="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --probe-dir) PROBE_DIR="$2"; shift 2 ;;
    --sae-probe-dir) SAE_PROBE_DIR="$2"; shift 2 ;;
    --text-pipeline) TEXT_PIPELINE="$2"; shift 2 ;;
    --plain-metrics-dir) PLAIN_METRICS_DIR="$2"; shift 2 ;;
    --cipher-base-dir) CIPHER_BASE_DIR="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --ciphers) CIPHERS="$2"; shift 2 ;;
    --print-only) PRINT_ONLY="1"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

PIPELINE_ARGS=(
  "--project-dir" "$PROJECT_DIR"
  "--venv-path" "$VENV_PATH"
  "--toolchain-rc" "$TOOLCHAIN_RC"
  "--config" "$CONFIG"
  "--probe-dir" "$PROBE_DIR"
  "--sae-probe-dir" "$SAE_PROBE_DIR"
  "--text-pipeline" "$TEXT_PIPELINE"
  "--plain-metrics-dir" "$PLAIN_METRICS_DIR"
  "--cipher-base-dir" "$CIPHER_BASE_DIR"
  "--output-dir" "$OUTPUT_DIR"
  "--ciphers" "$CIPHERS"
)

SBATCH_CMD=(
  sbatch
  -p "$PARTITION"
  --nodes=1
  --time="$TIME_LIMIT"
  --job-name="$JOB_NAME"
  --wrap "bash ${PROJECT_DIR}/scripts/cluster/run_transfer_eval.sh ${PIPELINE_ARGS[*]}"
)

printf 'Command:\n%s\n' "${SBATCH_CMD[*]}"

if [[ "$PRINT_ONLY" == "1" ]]; then
  exit 0
fi

"${SBATCH_CMD[@]}"
