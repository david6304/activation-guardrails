#!/bin/bash
#$ -N judge_eddie
#$ -cwd
#$ -q gpu
#$ -l gpu=2
#$ -l l40s=true
#$ -pe sharedmem 8
#$ -l h_rss=16G
#$ -l h_rt=06:00:00
#$ -o judge_eddie_$JOB_ID.out
#$ -j y
set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda/12.1.1
source /exports/eddie/scratch/s2296274/venv/bin/activate

export HF_HOME=/exports/eddie/scratch/s2296274/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /exports/eddie/scratch/s2296274/activation-guardrails

IN=${1:-data/judge_main_prompts.jsonl}
OUT=${2:-data/judged_main_prompts.jsonl}
BATCH=${3:-32}
MAXTOK=${4:-64}
MODE=${5:-prompt}

date --iso-8601=seconds
hostname
nvidia-smi -L
git rev-parse HEAD || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "IN=$IN OUT=$OUT BATCH=$BATCH MAXTOK=$MAXTOK MODE=$MODE"

if [[ ! -f "$IN" ]]; then
    echo "Missing input file: $IN" >&2
    exit 1
fi
if [[ ! -d "$HF_HOME/hub/models--Qwen--Qwen3.6-27B" ]]; then
    echo "Missing model cache: $HF_HOME/hub/models--Qwen--Qwen3.6-27B" >&2
    exit 1
fi

python judge_responses.py --in "$IN" --out "$OUT" --batch-size "$BATCH" --max-new-tokens "$MAXTOK" --mode "$MODE"
