#!/bin/bash
# P1 scoring on MLP. gemma-3-27b bf16 (~55 GB) fits on ONE A100 80GB or one
# h200_3g.71gb slice, so this asks for a single GPU: it schedules far more easily
# than the 2x L40S the Eddie job needs, and it sidesteps sharding entirely.
# Usage:
#   sbatch -p Wintermute --gres=gpu:nvidia_a100_80gb_pcie:1 phase1/run_p1_position_mlp.sh
#   sbatch -p ICF-Free   --gres=gpu:h200_3g.71gb:1          phase1/run_p1_position_mlp.sh
# Smoke first with LIMIT=8 and --time=01:00:00.
#SBATCH --job-name=p1_position
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=p1_position_%j.out
set -euo pipefail

# toolchain.rc reads LD_LIBRARY_PATH unguarded, which is fatal under `set -u`.
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

BATCH_SIZE="${BATCH_SIZE:-4}"
LIMIT="${LIMIT:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-data/p1_position_scores_${SLURM_JOB_ID}.npz}"
for path in \
    ~/.cache/huggingface/hub/models--google--gemma-3-27b-it \
    data/judged_main_prompts.jsonl \
    data/p1_conditions_manifest.json \
    data/phase1_activation_multilingual_27b.npz \
    data/phase1_layerwise_27b.npz; do
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
echo "command=python -m phase1.extend_position_activation --out $OUTPUT_PATH --batch-size $BATCH_SIZE --limit $LIMIT"

python -m phase1.extend_position_activation \
    --out "$OUTPUT_PATH" \
    --batch-size "$BATCH_SIZE" \
    --limit "$LIMIT"
