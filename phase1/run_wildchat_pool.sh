#!/bin/bash
# Score the frozen C3 WildChat background pool with one detector (MLP/ICF).
# Eddie twin: phase1/run_wildchat_pool_eddie.sh.
# Usage from the repository root:
#   sbatch -p Teaching --gres=gpu:nvidia_rtx_a6000:1 phase1/run_wildchat_pool.sh \
#       --detector qwen3guard --condition plain --out data/c3_pool_qwen_plain.npz
#SBATCH --job-name=c3_pool
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=c3_pool_%j.out
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
git status --short --untracked-files=no
echo "EXTRA=$*"

python -m phase1.score_wildchat_pool "$@"
