#!/bin/bash
set -euo pipefail

REPO=/exports/eddie/scratch/s2296274/activation-guardrails
VENV=/exports/eddie/scratch/s2296274/venv

if [[ ! -d "$REPO" || ! -f "$REPO/judge_responses.py" ]]; then
    echo "Missing Eddie repository or judge_responses.py at $REPO" >&2
    exit 1
fi

. /etc/profile.d/modules.sh
module load anaconda/2024.02 cuda/12.1.1

if [[ ! -f "$VENV/bin/activate" ]]; then
    python -m venv "$VENV"
fi
source "$VENV/bin/activate"

pip install --upgrade pip
pip install torch==2.12.0 transformers==5.12.0 accelerate==1.14.0 huggingface_hub

export HF_HOME=/exports/eddie/scratch/s2296274/hf
echo "Downloading Qwen/Qwen3.6-27B (~55 GB); this may take a while."
hf download Qwen/Qwen3.6-27B
echo "Qwen/Qwen3.6-27B download complete."
echo "SETUP DONE"
