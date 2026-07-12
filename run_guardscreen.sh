#!/bin/bash
# Guard-blindness screen: WildGuard/ShieldGemma/LlamaGuard recall+FPR on plain vs ciphered
# WJ-harmful / XSTest-safe prompts. Guards load one at a time (single GPU suffices).
# Usage: sbatch -p Teaching --gres=gpu:a6000:1 --time=01:00:00 run_guardscreen.sh [N] [GUARDS] [CIPHERS] [OUT] [BATCH] [LIMIT]
#SBATCH --job-name=guardscreen
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=guardscreen_%j.out
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

N=${1:-50}
GUARDS=${2:-wildguard,shieldgemma}
CIPHERS=${3:-plain,reverse,nato,morse,zulu}
OUT=${4:-data/guard_screen.jsonl}
BATCH=${5:-8}
LIMIT=${6:-0}

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short
echo "N=$N GUARDS=$GUARDS CIPHERS=$CIPHERS OUT=$OUT BATCH=$BATCH LIMIT=$LIMIT"

python guard_screen.py --n "$N" --guards "$GUARDS" --ciphers "$CIPHERS" \
  --out "$OUT" --batch-size "$BATCH" --limit "$LIMIT"
