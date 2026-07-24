#!/bin/bash
# Audit frozen scores and score missing French/Hindi/Zulu guards in one allocation.
#SBATCH --job-name=phase1_multiguard
#SBATCH --partition=Wintermute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=phase1_multiguard_%j.out
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

BASELINES_SOURCE="${BASELINES_SOURCE:-data/phase1_baselines_multilingual_tfidf.npz}"
for path in \
    ~/.cache/huggingface/hub/models--google--shieldgemma-9b \
    ~/.cache/huggingface/hub/models--allenai--wildguard \
    "$BASELINES_SOURCE" \
    data/phase1_baselines.npz \
    data/phase1_baselines.json \
    data/phase1_translations/french.jsonl \
    data/phase1_translations/hindi.jsonl \
    data/phase1_translations/swahili.jsonl \
    data/phase1_translations/zulu.jsonl; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required input: $path" >&2
        exit 1
    fi
done

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "commit=$CURRENT_COMMIT"
python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'

OUTPUT_PATH="${OUTPUT_PATH:-data/phase1_baselines_multilingual_${SLURM_JOB_ID}.npz}"
echo "command=python -m phase1.extend_multilingual_guards --source $BASELINES_SOURCE --out $OUTPUT_PATH $*"
python -m phase1.extend_multilingual_guards \
    --source "$BASELINES_SOURCE" \
    --out "$OUTPUT_PATH" \
    "$@"
