#!/bin/bash
# Submit CPU-only probe training to SLURM.
#
# Trains activation probes, SAE probes, the TF-IDF text baseline, and
# optionally Latent Guard, then writes the results table. No GPU required.
#
# Examples:
#   bash scripts/cluster/submit_probe_training_job.sh
#   bash scripts/cluster/submit_probe_training_job.sh --config configs/main/refusal_large_train.yaml
#   bash scripts/cluster/submit_probe_training_job.sh --label-key source_label
#   bash scripts/cluster/submit_probe_training_job.sh --with-latent-guard
#   bash scripts/cluster/submit_probe_training_job.sh --print-only
set -euo pipefail

PARTITION="Teaching"
TIME_LIMIT="02:00:00"
JOB_NAME="probe-training"
PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/refusal.yaml"
ACTIVATION_DIR="artifacts/activations/refusal"
FEATURE_DIR="artifacts/features/refusal/sae"
DATASET="data/processed/refusal_labelled.jsonl"
ADVERSARIAL_DATASET=""
TEXT_MODEL_DIR="artifacts/models/refusal/text_baseline_cv"
PROBE_MODEL_DIR="artifacts/models/refusal/activation_probes_cv"
SAE_MODEL_DIR="artifacts/models/refusal/sae_probes_cv"
LATENT_GUARD_DIR="artifacts/models/refusal/latent_guard_cv"
METRICS_DIR="results/refusal/metrics_cv"
RESULTS_OUTPUT="results/refusal/results_table.csv"
LABEL_KEY="label"
WITH_LATENT_GUARD="0"
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
    --activation-dir) ACTIVATION_DIR="$2"; shift 2 ;;
    --feature-dir|--sae-dir) FEATURE_DIR="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --adversarial-dataset) ADVERSARIAL_DATASET="$2"; shift 2 ;;
    --text-model-dir) TEXT_MODEL_DIR="$2"; shift 2 ;;
    --probe-model-dir) PROBE_MODEL_DIR="$2"; shift 2 ;;
    --sae-model-dir) SAE_MODEL_DIR="$2"; shift 2 ;;
    --latent-guard-dir) LATENT_GUARD_DIR="$2"; shift 2 ;;
    --metrics-dir) METRICS_DIR="$2"; shift 2 ;;
    --results-output) RESULTS_OUTPUT="$2"; shift 2 ;;
    --label-key) LABEL_KEY="$2"; shift 2 ;;
    --with-latent-guard) WITH_LATENT_GUARD="1"; shift ;;
    --without-latent-guard) WITH_LATENT_GUARD="0"; shift ;;
    --print-only) PRINT_ONLY="1"; shift ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

case "$LABEL_KEY" in
  label|source_label)
    ;;
  *)
    echo "Unknown --label-key: $LABEL_KEY (valid: label, source_label)" >&2
    exit 1
    ;;
esac

PIPELINE_ARGS=(
  "--project-dir" "$PROJECT_DIR"
  "--venv-path" "$VENV_PATH"
  "--toolchain-rc" "$TOOLCHAIN_RC"
  "--config" "$CONFIG"
  "--activation-dir" "$ACTIVATION_DIR"
  "--feature-dir" "$FEATURE_DIR"
  "--dataset" "$DATASET"
  "--text-model-dir" "$TEXT_MODEL_DIR"
  "--probe-model-dir" "$PROBE_MODEL_DIR"
  "--sae-model-dir" "$SAE_MODEL_DIR"
  "--latent-guard-dir" "$LATENT_GUARD_DIR"
  "--metrics-dir" "$METRICS_DIR"
  "--results-output" "$RESULTS_OUTPUT"
  "--label-key" "$LABEL_KEY"
)
if [[ -n "$ADVERSARIAL_DATASET" ]]; then
  PIPELINE_ARGS+=("--adversarial-dataset" "$ADVERSARIAL_DATASET")
fi
if [[ "$WITH_LATENT_GUARD" == "1" ]]; then
  PIPELINE_ARGS+=("--with-latent-guard")
fi

SBATCH_CMD=(
  sbatch
  -p "$PARTITION"
  --nodes=1
  --time="$TIME_LIMIT"
  --job-name="$JOB_NAME"
  --wrap "bash ${PROJECT_DIR}/scripts/cluster/run_probe_training.sh ${PIPELINE_ARGS[*]}"
)

printf 'Command:\n%s\n' "${SBATCH_CMD[*]}"

if [[ "$PRINT_ONLY" == "1" ]]; then
  exit 0
fi

"${SBATCH_CMD[@]}"
