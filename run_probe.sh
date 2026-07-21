#!/bin/bash
# Plain-trained input-only prompt probe on protected Gemma-3.
# Usage: sbatch -p Teaching --gres=gpu:h200_3g.71gb:1 --time=1-00:00:00 run_probe.sh [MODEL] [EXTRA...]
# Example smoke: ... run_probe.sh google/gemma-3-27b-it --limit 60 --batch-size 4
#SBATCH --job-name=probe_prompt
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=probe_prompt_%j.out
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL=${1:-google/gemma-3-27b-it}

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short
echo "MODEL=$MODEL EXTRA=${*:2}"

python probe_prompt.py --model "$MODEL" "${@:2}"
