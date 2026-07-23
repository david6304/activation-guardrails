#!/bin/bash
# Phase 2 t_post_inst scoring on Gemma 3 27B (Eddie, 2x L40S).
# Usage from the repository root: qsub phase2/run_phase2_eddie.sh [EXTRA...]
#$ -N phase2_pos
#$ -cwd
#$ -q gpu
#$ -l gpu=2
#$ -l l40s=true
#$ -pe sharedmem 12
#$ -l h_rss=32G
#$ -l h_rt=04:00:00
#$ -o phase2_pos_$JOB_ID.out
#$ -j y
set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda/12.1.1
source /exports/eddie/scratch/s2296274/venv/bin/activate

export HF_HOME=/exports/eddie/scratch/s2296274/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${SGE_O_WORKDIR:?qsub must be run from the intended checkout}"

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short
echo "EXTRA=$*"

python -m phase1.phase1_activation \
  --position t_post_inst \
  --out data/phase2_activation_t_post_inst_27b.npz \
  --batch-size 8 \
  "$@"
