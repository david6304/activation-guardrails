#!/bin/bash
# Fit the frozen multilingual-E5 baseline on one 11GB-capable Interactive GPU.
# Set SMOKE_PER_CLASS (for example 2) to run smoke then full in one allocation.
#SBATCH --job-name=phase1_e5
#SBATCH --partition=Interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=phase1_e5_%j.out
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
if [[ ! -d ~/.cache/huggingface/hub/models--intfloat--multilingual-e5-base ]]; then
    echo "Missing frozen multilingual-E5 cache" >&2
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
OUTPUT_PATH="${OUTPUT_PATH:-data/phase1_multilingual_e5_${SLURM_JOB_ID}.npz}"
if (( SMOKE_PER_CLASS > 0 )); then
    SMOKE_OUTPUT="data/phase1_multilingual_e5_smoke_${SLURM_JOB_ID}.npz"
    echo "smoke=python -m phase1.phase1_text_encoders --baseline multilingual_e5 --smoke-per-class $SMOKE_PER_CLASS --out $SMOKE_OUTPUT"
    python -m phase1.phase1_text_encoders "$@" \
        --baseline multilingual_e5 \
        --smoke-per-class "$SMOKE_PER_CLASS" \
        --out "$SMOKE_OUTPUT"
fi
echo "full=python -m phase1.phase1_text_encoders --baseline multilingual_e5 --out $OUTPUT_PATH"
python -m phase1.phase1_text_encoders "$@" \
    --baseline multilingual_e5 \
    --out "$OUTPUT_PATH"
