#!/bin/bash
# Benign-QA cipher-decode capability on the PROTECTED gemma-3-27b-it (WebQuestions).
# Usage: sbatch -p Teaching --gres=gpu:h200_3g.71gb:1 --time=00:40:00 run_capqa.sh [N] [CIPHERS] [OUT] [BATCH] [LIMIT]
#SBATCH --job-name=capqa
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=capqa_%j.out
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

N=${1:-150}
CIPHERS=${2:-plain,reverse,nato,morse,zulu}
OUT=${3:-data/cap_qa_27b.jsonl}
BATCH=${4:-16}
LIMIT=${5:-0}
MODEL=${6:-google/gemma-3-27b-it}

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short
echo "N=$N CIPHERS=$CIPHERS OUT=$OUT BATCH=$BATCH LIMIT=$LIMIT MODEL=$MODEL"

python capability_qa.py --n "$N" --ciphers "$CIPHERS" --out "$OUT" \
  --batch-size "$BATCH" --limit "$LIMIT" --model "$MODEL"
