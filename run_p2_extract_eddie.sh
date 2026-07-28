#!/bin/bash
# P2 stage 3: teacher-force prompt+response through the PROTECTED 27B on Eddie and
# score every response position with the frozen probe. Two L40S (27B bf16 is ~55 GB).
# Only the response JSONL crosses from MLP; the abliterated weights stay there.
# Smoke first: qsub -l h_rt=01:00:00 -v EXPECTED_COMMIT=<commit>,LIMIT=8 ...
#$ -N p2_extract
#$ -cwd
#$ -q gpu
#$ -l gpu=2
#$ -l l40s=true
#$ -pe sharedmem 12
#$ -l h_rss=32G
#$ -l h_rt=08:00:00
#$ -o p2_extract_$JOB_ID.out
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

JUDGED="${JUDGED:-data/p2_judged.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LIMIT="${LIMIT:-0}"
OUTPUT_PATH="${OUTPUT_PATH:-data/p2_latency_scores_${JOB_ID}.npz}"
for path in \
    "$HF_HOME/hub/models--google--gemma-3-27b-it" \
    "$JUDGED" \
    data/phase1_activation_multilingual_27b.npz; do
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
echo "command=python p2_extract_latency.py --in $JUDGED --out $OUTPUT_PATH --batch-size $BATCH_SIZE --limit $LIMIT"

python p2_extract_latency.py \
    --in "$JUDGED" \
    --out "$OUTPUT_PATH" \
    --batch-size "$BATCH_SIZE" \
    --limit "$LIMIT"
