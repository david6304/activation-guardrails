#!/bin/bash
# Swahili translation of the frozen C7 external pool (NLLB-600M, pinned revision).
# Usage: sbatch -p Teaching --gres=gpu:h200_1g.18gb:1 phase1/run_translate_c7.sh
#SBATCH --job-name=c7_translate
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=c7_translate_%j.out
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

python -m phase1.translate_c7_external "$@"
