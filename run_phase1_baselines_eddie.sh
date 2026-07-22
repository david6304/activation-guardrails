#!/bin/bash
# Matched Phase 1 text and guard scoring (Eddie, 1x L40S).
# Usage: qsub run_phase1_baselines_eddie.sh [EXTRA...]
#$ -N phase1_text
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l l40s=true
#$ -pe sharedmem 4
#$ -l h_rss=32G
#$ -l h_rt=02:00:00
#$ -o phase1_text_$JOB_ID.out
#$ -j y
set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda/12.1.1
source /exports/eddie/scratch/s2296274/venv/bin/activate

export HF_HOME=/exports/eddie/scratch/s2296274/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${SGE_O_WORKDIR:?qsub must be run from the intended checkout}"

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short
echo "EXTRA=$*"

python phase1_baselines.py "$@"
