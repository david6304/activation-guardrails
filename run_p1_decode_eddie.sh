#!/bin/bash
# P1 decode gate: does 27B actually decode the base64 prompts P1 tests?
# Two L40S because gemma-3-27b bf16 is ~55 GB and one card is 48 GB; generate()
# shards fine across both. Runs from the submitting checkout, like the P1 scoring
# job, so a concurrent branch move in the shared tree cannot change what ran.
# Usage: qsub -v EXPECTED_COMMIT=<commit> run_p1_decode_eddie.sh
#$ -N p1_decode
#$ -cwd
#$ -q gpu
#$ -l gpu=2
#$ -l l40s=true
#$ -pe sharedmem 8
#$ -l h_rss=16G
#$ -l h_rt=02:00:00
#$ -o p1_decode_$JOB_ID.out
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

N="${N:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
OUTPUT_PATH="${OUTPUT_PATH:-data/p1_decode_fidelity_${JOB_ID}.jsonl}"
for path in \
    "$HF_HOME/hub/models--google--gemma-3-27b-it" \
    data/judged_main_prompts.jsonl; do
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
echo "command=python p1_decode_fidelity.py --out $OUTPUT_PATH --n $N --batch-size $BATCH_SIZE --max-new-tokens $MAX_NEW_TOKENS"

python p1_decode_fidelity.py \
    --out "$OUTPUT_PATH" \
    --n "$N" \
    --batch-size "$BATCH_SIZE" \
    --max-new-tokens "$MAX_NEW_TOKENS"
