#!/bin/bash
# ShieldGemma over the C7 external pool, plain and Swahili.
# Usage: sbatch -p Teaching --gres=gpu:h200_3g.71gb:1 phase1/run_guard_c7.sh [EXTRA...]
#SBATCH --job-name=c7_guard
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=c7_guard_%j.out
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
nvidia-smi -L
git rev-parse HEAD

python -m phase1.score_c7_guard "$@"
