#!/bin/bash
# Score only missing French/Hindi/Zulu conditions with the frozen 27B detectors.
# The exact model/scoring path and A100 batch size were already validated by job
# 3565792, so this continuation deliberately avoids a separate smoke and reload.
#SBATCH --job-name=phase1_multiact
#SBATCH --partition=Wintermute
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=phase1_multiact_%j.out
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
for path in \
    ~/.cache/huggingface/hub/models--google--gemma-3-27b-it \
    data/phase1_activation_27b.npz \
    data/phase1_translations/french.jsonl \
    data/phase1_translations/hindi.jsonl \
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

OUTPUT_PATH="${OUTPUT_PATH:-data/phase1_activation_multilingual_27b_${SLURM_JOB_ID}.npz}"
echo "command=python -m phase1.extend_multilingual_activation --batch-size 4 --out $OUTPUT_PATH $*"
python -m phase1.extend_multilingual_activation \
    --batch-size 4 \
    --out "$OUTPUT_PATH" \
    "$@"
