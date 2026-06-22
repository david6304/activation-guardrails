#!/bin/bash
# Train the streaming linear probe on the cached activations. I/O-bound (reads the
# full ~1.8 TB activation cache per epoch from Lustre); modest VRAM (linear probe),
# so request whatever GPU is free rather than queueing for an A100.
# Usage: sbatch -p Teaching --gres=gpu:<TYPE>:1 --time=06:00:00 run_train.sh [EPOCHS] [BATCH] [OUT]
#SBATCH --job-name=train_probe
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=train_probe_%j.out
set -euo pipefail

source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

EPOCHS=${1:-3}
BATCH=${2:-8}
OUT=${3:-data/probe.pt}

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short
echo "EPOCHS=$EPOCHS BATCH=$BATCH OUT=$OUT"

python train_probe.py --manifest data/acts_manifest.jsonl \
  --epochs "$EPOCHS" --batch-size "$BATCH" --num-workers 8 --out "$OUT"
