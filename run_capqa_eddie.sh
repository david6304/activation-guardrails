#!/bin/bash
# Benign-QA cipher/language decode capability on protected Gemma-3 (Eddie / Grid Engine).
# Slurm twin: run_capqa.sh. capability_qa.py uses model.generate(), so a sharded 27B
# across 2xL40S is fine here (no cross-device hidden-state stack, unlike the probe).
# Usage: qsub [-l gpu=N] run_capqa_eddie.sh [N] [CIPHERS] [OUT] [BATCH] [LIMIT] [MODEL]
#   12B (1 L40S):  qsub run_capqa_eddie.sh 150 plain,french,hindi,swahili data/cap_qa_langs_12b.jsonl 16 0 google/gemma-3-12b-it
#   27B (2 L40S):  qsub -l gpu=2 run_capqa_eddie.sh 150 plain,french,hindi,swahili data/cap_qa_langs_27b.jsonl 16 0 google/gemma-3-27b-it
# With prebuilt rows (ITEMS, 7th arg) the first five args are ignored by capability_qa.py:
#   qsub -l gpu=2 run_capqa_eddie.sh 0 plain data/squad_cipher_27b.jsonl 8 0 google/gemma-3-27b-it data/squad_cipher_items.jsonl 128
#$ -N capqa
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l l40s=true
#$ -pe sharedmem 8
#$ -l h_rss=16G
#$ -l h_rt=02:00:00
#$ -o capqa_$JOB_ID.out
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

N=${1:-150}
CIPHERS=${2:-plain,reverse,nato,morse,zulu}
OUT=${3:-data/cap_qa_27b.jsonl}
BATCH=${4:-16}
LIMIT=${5:-0}
MODEL=${6:-google/gemma-3-27b-it}
ITEMS=${7:-}
MAXNEW=${8:-64}

date --iso-8601=seconds
hostname
nvidia-smi -L
git rev-parse HEAD || true
cat LOCAL_COMMIT.txt 2>/dev/null || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "N=$N CIPHERS=$CIPHERS OUT=$OUT BATCH=$BATCH LIMIT=$LIMIT MODEL=$MODEL ITEMS=$ITEMS MAXNEW=$MAXNEW"

CACHE_DIR="$HF_HOME/hub/models--${MODEL//\//--}"
if [[ ! -d "$CACHE_DIR" ]]; then
    echo "Missing model cache: $CACHE_DIR" >&2
    exit 1
fi

python capability_qa.py --n "$N" --ciphers "$CIPHERS" --out "$OUT" \
  --batch-size "$BATCH" --limit "$LIMIT" --model "$MODEL" \
  --max-new-tokens "$MAXNEW" ${ITEMS:+--items "$ITEMS"}
