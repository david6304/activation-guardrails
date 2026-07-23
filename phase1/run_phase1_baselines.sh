#!/bin/bash
# Matched Phase 1 TF-IDF, ShieldGemma and WildGuard scoring (MLP/ICF).
# Usage from the repository root: sbatch ... phase1/run_phase1_baselines.sh [EXTRA...]
#SBATCH --job-name=phase1_text
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=phase1_text_%j.out
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short
echo "EXTRA=$*"

python -m phase1.phase1_baselines "$@"
