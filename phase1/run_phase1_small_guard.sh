#!/bin/bash
# Fine-tune the frozen DeBERTa baseline on one 11GB-capable Interactive GPU.
# Set SMOKE_PER_CLASS (for example 2) to run smoke then full in one allocation.
#SBATCH --job-name=phase1_deberta
#SBATCH --partition=Interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=phase1_deberta_%j.out
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the submitted checkout commit}"
CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    echo "Checkout mismatch: current=$CURRENT_COMMIT expected=$EXPECTED_COMMIT" >&2
    exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked checkout is dirty" >&2
    git status --short
    exit 1
fi
if [[ ! -d ~/.cache/huggingface/hub/models--microsoft--deberta-v3-small ]]; then
    echo "Missing frozen DeBERTa cache" >&2
    exit 1
fi
if [[ ! -f data/phase1_translations/metadata.json ]]; then
    echo "Missing frozen translation metadata" >&2
    exit 1
fi

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "commit=$CURRENT_COMMIT"
python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'

SMOKE_PER_CLASS="${SMOKE_PER_CLASS:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-data/phase1_small_guard_${SLURM_JOB_ID}.npz}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-data/phase1_small_guard_${SLURM_JOB_ID}_checkpoint.pt}"
if (( SMOKE_PER_CLASS > 0 )); then
    SMOKE_OUTPUT="data/phase1_small_guard_smoke_${SLURM_JOB_ID}.npz"
    SMOKE_CHECKPOINT="data/phase1_small_guard_smoke_${SLURM_JOB_ID}_checkpoint.pt"
    echo "smoke=python -m phase1.phase1_text_encoders --baseline small_guard --smoke-per-class $SMOKE_PER_CLASS --out $SMOKE_OUTPUT"
    python -m phase1.phase1_text_encoders "$@" \
        --baseline small_guard \
        --smoke-per-class "$SMOKE_PER_CLASS" \
        --out "$SMOKE_OUTPUT" \
        --checkpoint-out "$SMOKE_CHECKPOINT"
fi
echo "full=python -m phase1.phase1_text_encoders --baseline small_guard --out $OUTPUT_PATH"
python -m phase1.phase1_text_encoders "$@" \
    --baseline small_guard \
    --out "$OUTPUT_PATH" \
    --checkpoint-out "$CHECKPOINT_PATH"
