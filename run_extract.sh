#!/bin/bash
# Cache all-layer response activations (protected Gemma 3 12B) for the judged set.
# Resumable: skip-if-exists per id, so a timeout/preempt loses no work.
# Usage: sbatch run_extract.sh   (override target with -p/--gres/--time on the CLI)
#SBATCH --job-name=extract_acts
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=extract_acts_%j.out
set -euo pipefail

source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

IN=${1:-data/judged_train.jsonl}
BATCH=${2:-8}

date --iso-8601=seconds
hostname
git rev-parse HEAD
git status --short
echo "IN=$IN BATCH=$BATCH"

python extract_activations.py --in "$IN" --batch-size "$BATCH"
