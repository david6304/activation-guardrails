#!/bin/bash
# Score all six Phase 1 conditions with Qwen3Guard-Gen-8B on one Eddie L40S.
#$ -N c4_modern_guards
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l l40s=true
#$ -pe sharedmem 4
#$ -l h_rss=32G
#$ -l h_rt=03:00:00
#$ -o c4_modern_guards_$JOB_ID.out
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

BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT="${LIMIT:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-data/c4_modern_guards_${JOB_ID}.npz}"
for path in \
    "$HF_HOME/hub/models--Qwen--Qwen3Guard-Gen-8B" \
    data/judged_main_prompts.jsonl \
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
echo "command=python -m phase1.score_modern_guards --out $OUTPUT_PATH --batch-size $BATCH_SIZE --limit $LIMIT"

python -m phase1.score_modern_guards \
    --out "$OUTPUT_PATH" \
    --batch-size "$BATCH_SIZE" \
    --limit "$LIMIT"
