#!/bin/bash
# P1: score the frozen 27B detectors on the cipher conditions at both read
# positions, on two Eddie L40S (gemma-3-27b bf16 is ~55 GB, one card is 48 GB).
# Smoke first: qsub -l h_rt=01:00:00 -v EXPECTED_COMMIT=<commit>,LIMIT=8 ...
#$ -N p1_position
#$ -cwd
#$ -q gpu
#$ -l gpu=2
#$ -l l40s=true
#$ -pe sharedmem 12
#$ -l h_rss=32G
#$ -l h_rt=06:00:00
#$ -o p1_position_$JOB_ID.out
#$ -j y
set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda/12.1.1
source /exports/eddie/scratch/s2296274/venv/bin/activate

export HF_HOME=/exports/eddie/scratch/s2296274/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${SGE_O_WORKDIR:?qsub must run from the intended checkout}"

: "${EXPECTED_COMMIT:?submit with qsub -v EXPECTED_COMMIT=<commit>}"
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
OUTPUT_PATH="${OUTPUT_PATH:-data/p1_position_scores_${JOB_ID}.npz}"
for path in \
    "$HF_HOME/hub/models--google--gemma-3-27b-it" \
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
