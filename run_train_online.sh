#!/bin/bash
# Train the streaming probe online: recompute Gemma 3 12B activations each step
# instead of reading the 1.8 TB cache (Lustre read was ~5.5 h/epoch). Needs a GPU
# that fits the 12B (same H200 MIG slice as extraction), not the 2080 Ti.
# Usage: sbatch run_train_online.sh [IN] [EPOCHS] [BATCH] [OUT]
#SBATCH --job-name=train_probe_online
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=train_probe_online_%j.out
set -euo pipefail

source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

IN=${1:-data/judged_train.jsonl}
EPOCHS=${2:-10}    # ceiling; train_probe_online early-stops on val
BATCH=${3:-8}
OUT=${4:-data/probe.pt}

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short
echo "IN=$IN EPOCHS=$EPOCHS BATCH=$BATCH OUT=$OUT EXTRA=${*:5}"

# Extra args from position 5 on pass through (e.g. --lr 1e-4 --weight-decay 0.1).
python train_probe_online.py --in "$IN" --epochs "$EPOCHS" --batch-size "$BATCH" --out "$OUT" "${@:5}"
