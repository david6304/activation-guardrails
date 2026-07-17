#!/bin/bash
# Reverse-mechanism pilot: last-token all-layer probe on protected gemma-3-27b, plain->reverse
# transfer. Forward-pass only (no generation).
# Usage: sbatch -p Teaching --gres=gpu:h200_3g.71gb:1 --time=01:00:00 run_reverse_pilot.sh [N] [BATCH] [OUT] [LIMIT]
#SBATCH --job-name=rev_pilot
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=rev_pilot_%j.out
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

N=${1:-300}
BATCH=${2:-8}
OUT=${3:-data/reverse_pilot.json}
LIMIT=${4:-0}

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short | grep -v '\.out$' || true
echo "N=$N BATCH=$BATCH OUT=$OUT LIMIT=$LIMIT"

python reverse_pilot.py --n "$N" --batch-size "$BATCH" --out "$OUT" --limit "$LIMIT"
