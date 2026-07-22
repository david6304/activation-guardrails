#!/bin/bash
# Per-layer operational-harm diagnostic on protected Gemma-3 27B.
# Usage: sbatch -p Wintermute --gres=gpu:nvidia_a100_80gb_pcie:1 \
#        --time=06:00:00 run_probe_layers.sh [EXTRA...]
#SBATCH --job-name=probe_layers
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=probe_layers_%j.out
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short
echo "EXTRA=$*"

python probe_prompt_layers.py "$@"
