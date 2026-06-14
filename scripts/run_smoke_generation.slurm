#!/usr/bin/env bash
# Submit from the repository root with:
# sbatch scripts/run_smoke_generation.slurm
# Reusing the same output path safely resumes completed examples.
#SBATCH --job-name=ag-smoke-generation
#SBATCH --partition=Wintermute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --signal=TERM@120
#SBATCH --output=slurm-smoke-generation-%j.out
#SBATCH --error=slurm-smoke-generation-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/activation-guardrails}"
TOOLCHAIN_RC="${TOOLCHAIN_RC:-/home/htang2/toolchain-20251006/toolchain.rc}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$HOME/venvs/ml/bin/activate}"
MANIFEST="${MANIFEST:-$REPO_DIR/data/processed/wildjailbreak_smoke_manifest.jsonl}"
OUTPUT="${OUTPUT:-$REPO_DIR/artifacts/responses/wildjailbreak_smoke_responses.jsonl}"
MODEL_PATH="${MODEL_PATH:-/home/s2296274/models/gemma-3-4b-it-heretic}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
BASE_SEED="${BASE_SEED:-20260612}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"

summarize() {
    if [[ ! -f "$OUTPUT" ]]; then
        printf '%s\n' '{"aggregate_status":"no_output_file"}'
        return
    fi
    python - "$OUTPUT" <<'PY'
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

records = [
    json.loads(line)
    for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
]
statuses = Counter(record.get("status") for record in records)
terminations = Counter(
    record.get("termination_reason")
    for record in records
    if record.get("status") == "success"
)
response_lengths = [
    record["response_token_count"]
    for record in records
    if record.get("status") == "success"
]
summary = {
    "aggregate_status": "response_artifact_summary",
    "records": len(records),
    "successes": statuses["success"],
    "failures": statuses["failure"],
    "termination_eos": terminations["eos"],
    "termination_length_cap": terminations["length_cap"],
    "termination_other": terminations["other"],
    "mean_response_tokens": (
        round(statistics.fmean(response_lengths), 2) if response_lengths else None
    ),
    "max_response_tokens": max(response_lengths, default=None),
}
print(json.dumps(summary, sort_keys=True))
PY
}

for required_path in "$TOOLCHAIN_RC" "$VENV_ACTIVATE" "$MANIFEST"; do
    if [[ ! -e "$required_path" ]]; then
        printf 'Required path is missing: %s\n' "$required_path" >&2
        exit 2
    fi
done
for required_dir in "$REPO_DIR" "$MODEL_PATH" "$TOKENIZER_PATH"; do
    if [[ ! -d "$required_dir" ]]; then
        printf 'Required directory is missing: %s\n' "$required_dir" >&2
        exit 2
    fi
done

source "$TOOLCHAIN_RC"
source "$VENV_ACTIVATE"
cd "$REPO_DIR"

export HF_HOME="${HF_HOME:-$HOME/models}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

mkdir -p "$(dirname "$OUTPUT")"

date --iso-8601=seconds
hostname
git rev-parse HEAD
git status --short
python --version
python -c 'import torch, transformers; print("torch", torch.__version__); print("transformers", transformers.__version__)'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - "$MODEL_PATH" "$TOKENIZER_PATH" <<'PY'
import hashlib
import sys

from transformers import AutoConfig, AutoTokenizer

model_path, tokenizer_path = sys.argv[1:]
config = AutoConfig.from_pretrained(model_path, local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
template = tokenizer.chat_template
if not template:
    raise RuntimeError("cached tokenizer has no chat template")
print(
    "cache_preflight",
    {
        "model_type": config.model_type,
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
    },
)
PY

GENERATION_COMMAND=(
    python scripts/generate_smoke_responses.py
    --manifest "$MANIFEST"
    --output "$OUTPUT"
    --model-path "$MODEL_PATH"
    --tokenizer-path "$TOKENIZER_PATH"
    --base-seed "$BASE_SEED"
    --temperature "$TEMPERATURE"
    --top-p "$TOP_P"
    --max-new-tokens "$MAX_NEW_TOKENS"
)

printf 'generation_command'
printf ' %q' "${GENERATION_COMMAND[@]}"
printf '\n'

"${GENERATION_COMMAND[@]}" --dry-run

set +e
"${GENERATION_COMMAND[@]}"
generation_status=$?
set -e

summarize
exit "$generation_status"
