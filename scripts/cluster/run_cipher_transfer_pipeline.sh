#!/bin/bash
# Run one cipher-transfer pipeline for refusal probing.
#
# Stages:
#   build       - build ciphered test prompts from the plain-text prompt set
#   filter      - filter existing labelled responses to the current cipher dataset
#                 (use instead of responses+judge when reusing a previous run)
#   responses   - generate model responses on cipher prompts
#   judge       - label cipher responses as refusal/compliance
#   relabel     - rebuild the cipher dataset with judge-assigned refusal labels
#   activations - extract dense activations from the relabelled cipher dataset
#   encode      - encode cached activations with SAEs
#   all         - build -> responses -> judge -> relabel -> activations -> encode
#   subset_fill - build -> filter -> generate/judge missing -> merge -> relabel
#                 -> activations -> encode
#                 (reuses overlap, tops up only missing labels)
set -euo pipefail

STAGE="all"
CIPHER=""
PROJECT_DIR="${HOME}/activation-guardrails"
VENV_PATH="${HOME}/venvs/ml"
TOOLCHAIN_RC="/home/htang2/toolchain-20251006/toolchain.rc"
CONFIG="configs/main/refusal.yaml"
PLAIN_DATASET="data/processed/refusal_prompts.jsonl"
BASE_DIR="artifacts/cipher_transfer"
EXISTING_BASE_DIR=""  # source dir for filter stage; defaults to BASE_DIR if empty
MODEL_CACHE="${HOME}/models"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --cipher) CIPHER="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --venv-path) VENV_PATH="$2"; shift 2 ;;
    --toolchain-rc) TOOLCHAIN_RC="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --plain-dataset) PLAIN_DATASET="$2"; shift 2 ;;
    --base-dir) BASE_DIR="$2"; shift 2 ;;
    --existing-base-dir) EXISTING_BASE_DIR="$2"; shift 2 ;;
    --model-cache) MODEL_CACHE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$CIPHER" ]]; then
  echo "--cipher is required" >&2
  exit 1
fi

# Default existing base dir to BASE_DIR (used by filter stage).
if [[ -z "$EXISTING_BASE_DIR" ]]; then
  EXISTING_BASE_DIR="$BASE_DIR"
fi

DATA_DIR="${BASE_DIR}/${CIPHER}/data"
RESPONSES_DIR="${BASE_DIR}/${CIPHER}/responses"
ACTIVATIONS_DIR="${BASE_DIR}/${CIPHER}/activations"
FEATURES_DIR="${BASE_DIR}/${CIPHER}/sae"
METRICS_DIR="results/refusal/cipher_transfer/${CIPHER}"

CIPHER_DATASET="${DATA_DIR}/${CIPHER}_prompts.jsonl"
LABELLED_RESPONSES="${RESPONSES_DIR}/labelled_responses.jsonl"
RESPONSES_PATH="${RESPONSES_DIR}/responses.jsonl"
RELABELLED_DATASET="${DATA_DIR}/${CIPHER}_labelled.jsonl"
REUSED_LABELLED_RESPONSES="${RESPONSES_DIR}/reused_labelled_responses.jsonl"
MISSING_DATASET="${DATA_DIR}/${CIPHER}_missing_prompts.jsonl"
MISSING_RESPONSES_PATH="${RESPONSES_DIR}/missing_responses.jsonl"
MISSING_LABELLED_RESPONSES="${RESPONSES_DIR}/missing_labelled_responses.jsonl"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

source "$TOOLCHAIN_RC"
source "${VENV_PATH}/bin/activate"

cd "$PROJECT_DIR"
export PYTHONPATH=src:.
export PYTHONUNBUFFERED=1

log "=== Cipher-transfer pipeline starting - stage: ${STAGE} cipher: ${CIPHER} ==="
log "Node: $(hostname)  Job: ${SLURM_JOB_ID:-local}"
log "Config: ${CONFIG}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
  | while read -r line; do log "GPU: $line"; done || true

export HF_HOME="$MODEL_CACHE"
export HUGGINGFACE_HUB_CACHE="${MODEL_CACHE}/hub"
export TRANSFORMERS_CACHE="${MODEL_CACHE}/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

run_build() {
  log "--- stage: build ---"
  mkdir -p "$DATA_DIR"
  python scripts/main/build_cipher_dataset.py \
    --dataset "$PLAIN_DATASET" \
    --cipher "$CIPHER" \
    --split test \
    --output "$CIPHER_DATASET"
  log "--- stage: build done ---"
}

run_responses() {
  local dataset_path="${1:-$CIPHER_DATASET}"
  local output_path="${2:-$RESPONSES_PATH}"
  log "--- stage: responses ---"
  mkdir -p "$RESPONSES_DIR"
  python scripts/main/generate_responses.py \
    --config "$CONFIG" \
    --dataset "$dataset_path" \
    --output "$output_path"
  log "--- stage: responses done ---"
}

run_judge() {
  local responses_path="${1:-$RESPONSES_PATH}"
  local output_path="${2:-$LABELLED_RESPONSES}"
  log "--- stage: judge ---"
  mkdir -p "$RESPONSES_DIR"
  python scripts/main/label_refusals.py \
    --config "$CONFIG" \
    --responses "$responses_path" \
    --output "$output_path"
  log "--- stage: judge done ---"
}

run_filter() {
  local output_path="${1:-$LABELLED_RESPONSES}"
  log "--- stage: filter ---"
  # Filter existing labelled responses down to the example IDs in the current
  # cipher dataset. Use this instead of responses+judge when the current dataset
  # is a subset of one that has already been generated and judged.
  local existing_labelled="${EXISTING_BASE_DIR}/${CIPHER}/responses/labelled_responses.jsonl"
  if [[ ! -f "$existing_labelled" ]]; then
    echo "filter stage: existing labelled responses not found at ${existing_labelled}" >&2
    exit 1
  fi
  mkdir -p "$RESPONSES_DIR"
  python - "$CIPHER_DATASET" "$existing_labelled" "$output_path" <<'EOF'
import json, sys
cipher_path, existing_path, output_path = sys.argv[1], sys.argv[2], sys.argv[3]

with open(cipher_path) as f:
    cipher_ids = {json.loads(line)["example_id"] for line in f}

kept = []
with open(existing_path) as f:
    for line in f:
        rec = json.loads(line)
        if rec["example_id"] in cipher_ids:
            kept.append(rec)

with open(output_path, "w") as f:
    for rec in kept:
        f.write(json.dumps(rec) + "\n")

print(f"Filtered {len(kept)} / {len(cipher_ids)} records "
      f"(expected {len(cipher_ids)})")
if len(kept) != len(cipher_ids):
    missing = len(cipher_ids) - len(kept)
    print(f"WARNING: {missing} cipher examples have no existing labelled response",
          file=__import__("sys").stderr)
EOF
  log "--- stage: filter done ---"
}

run_prepare_missing() {
  log "--- stage: prepare-missing ---"
  mkdir -p "$DATA_DIR"
  python - "$CIPHER_DATASET" "$REUSED_LABELLED_RESPONSES" "$MISSING_DATASET" <<'EOF'
import json, sys

cipher_path, reused_path, missing_path = sys.argv[1], sys.argv[2], sys.argv[3]

with open(cipher_path) as f:
    cipher_records = [json.loads(line) for line in f]

reused_ids = set()
try:
    with open(reused_path) as f:
        for line in f:
            line = line.strip()
            if line:
                reused_ids.add(json.loads(line)["example_id"])
except FileNotFoundError:
    pass

missing_records = [
    rec for rec in cipher_records if rec["example_id"] not in reused_ids
]

with open(missing_path, "w") as f:
    for rec in missing_records:
        f.write(json.dumps(rec) + "\n")

print(
    f"Prepared missing-only dataset: {len(missing_records)} / {len(cipher_records)} "
    "examples require new responses+labels"
)
EOF
  log "--- stage: prepare-missing done ---"
}

run_merge_labelled() {
  log "--- stage: merge-labels ---"
  mkdir -p "$RESPONSES_DIR"
  python - "$CIPHER_DATASET" "$LABELLED_RESPONSES" "$REUSED_LABELLED_RESPONSES" "$MISSING_LABELLED_RESPONSES" <<'EOF'
import json, sys
from pathlib import Path

cipher_path, final_path, reused_path, missing_path = sys.argv[1:5]

with open(cipher_path) as f:
    cipher_records = [json.loads(line) for line in f]

records_by_id = {}
for input_path in [reused_path, missing_path]:
    path = Path(input_path)
    if not path.exists():
        continue
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records_by_id[record["example_id"]] = record

missing_ids = [
    rec["example_id"] for rec in cipher_records
    if rec["example_id"] not in records_by_id
]
if missing_ids:
    raise SystemExit(
        f"Cannot merge labelled responses: still missing {len(missing_ids)} "
        "cipher examples"
    )

with open(final_path, "w") as f:
    for rec in cipher_records:
        f.write(json.dumps(records_by_id[rec["example_id"]]) + "\n")

print(f"Merged labelled responses: {len(cipher_records)} examples")
EOF
  log "--- stage: merge-labels done ---"
}

run_relabel() {
  log "--- stage: relabel ---"
  mkdir -p "$DATA_DIR"
  python scripts/main/build_refusal_dataset.py \
    --config "$CONFIG" \
    --mode relabel \
    --source-dataset "$CIPHER_DATASET" \
    --labelled-responses "$LABELLED_RESPONSES" \
    --output "$RELABELLED_DATASET"
  log "--- stage: relabel done ---"
}

run_activations() {
  log "--- stage: activations ---"
  mkdir -p "$ACTIVATIONS_DIR"
  python scripts/main/extract_activations.py \
    --config "$CONFIG" \
    --dataset "$RELABELLED_DATASET" \
    --output-dir "$ACTIVATIONS_DIR" \
    --splits test
  log "--- stage: activations done ---"
}

run_encode() {
  log "--- stage: encode ---"
  mkdir -p "$FEATURES_DIR" "$METRICS_DIR"
  python scripts/main/encode_sae_features.py \
    --config "$CONFIG" \
    --input-dir "$ACTIVATIONS_DIR" \
    --output-dir "$FEATURES_DIR" \
    --metrics-dir "$METRICS_DIR" \
    --splits test \
    --device cuda
  log "--- stage: encode done ---"
}

case "$STAGE" in
  build) run_build ;;
  filter) run_filter ;;
  responses) run_responses ;;
  judge) run_judge ;;
  relabel) run_relabel ;;
  activations) run_activations ;;
  encode) run_encode ;;
  all)
    run_build
    run_responses
    run_judge
    run_relabel
    run_activations
    run_encode
    ;;
  subset_fill)
    # Reuse overlap from an existing cipher run, then generate/judge only the
    # missing examples before continuing with feature extraction.
    run_build
    run_filter "$REUSED_LABELLED_RESPONSES"
    run_prepare_missing
    if [[ -s "$MISSING_DATASET" ]]; then
      run_responses "$MISSING_DATASET" "$MISSING_RESPONSES_PATH"
      run_judge "$MISSING_RESPONSES_PATH" "$MISSING_LABELLED_RESPONSES"
    else
      log "No missing cipher examples; skipping responses/judge top-up."
      rm -f "$MISSING_RESPONSES_PATH" "$MISSING_LABELLED_RESPONSES"
    fi
    run_merge_labelled
    run_relabel
    run_activations
    run_encode
    ;;
  *)
    echo "Unknown stage: $STAGE (valid: build, filter, responses, judge, relabel, activations, encode, all, subset_fill)" >&2
    exit 1
    ;;
esac

log "=== Cipher-transfer pipeline done - stage: ${STAGE} cipher: ${CIPHER} ==="
