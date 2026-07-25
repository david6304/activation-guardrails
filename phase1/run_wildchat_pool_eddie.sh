#!/bin/bash
# Score the frozen C3 WildChat background pool with one detector on Eddie.
# Submit with -v DETECTOR=probe|shieldgemma|qwen3guard, CONDITION=plain|swahili.
# The 27B probe needs two L40S: qsub -l gpu=2 -l h_rt=... .
#$ -N c3_pool
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l l40s=true
#$ -pe sharedmem 4
#$ -l h_rss=32G
#$ -l h_rt=04:00:00
#$ -o c3_pool_$JOB_ID.out
#$ -j y
set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda/12.1.1
source /exports/eddie/scratch/s2296274/venv/bin/activate

export HF_HOME=/exports/eddie/scratch/s2296274/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
    exit 1
fi

DETECTOR="${DETECTOR:?set -v DETECTOR=probe|shieldgemma|qwen3guard}"
CONDITION="${CONDITION:-plain}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LIMIT="${LIMIT:-0}"
SUFFIX="${SUFFIX:-$JOB_ID}"
OUTPUT_PATH="${OUTPUT_PATH:-data/c3_pool_${DETECTOR}_${CONDITION}_${SUFFIX}.npz}"

for path in data/c3_wildchat_prompts.jsonl data/phase1_activation_27b.npz; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required input: $path" >&2
        exit 1
    fi
done
if [[ "$CONDITION" == "swahili" && ! -e data/c3_wildchat_swahili.jsonl ]]; then
    echo "Missing required input: data/c3_wildchat_swahili.jsonl" >&2
    exit 1
fi

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "commit=$CURRENT_COMMIT"
python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'
echo "command=python -m phase1.score_wildchat_pool --detector $DETECTOR --condition $CONDITION --out $OUTPUT_PATH --batch-size $BATCH_SIZE --limit $LIMIT"

python -m phase1.score_wildchat_pool \
    --detector "$DETECTOR" \
    --condition "$CONDITION" \
    --out "$OUTPUT_PATH" \
    --batch-size "$BATCH_SIZE" \
    --limit "$LIMIT"
