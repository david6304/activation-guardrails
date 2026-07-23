#!/bin/bash
# Definitive operational-label layerwise diagnostic at t_inst (2x L40S).
# Usage: qsub phase1/run_phase1_layerwise_eddie.sh [EXTRA...]
# Smoke: qsub -l h_rt=01:00:00 phase1/run_phase1_layerwise_eddie.sh --smoke
#$ -N phase1_layers
#$ -cwd
#$ -q gpu
#$ -l gpu=2
#$ -l l40s=true
#$ -pe sharedmem 12
#$ -l h_rss=32G
#$ -l h_rt=06:00:00
#$ -o phase1_layers_$JOB_ID.out
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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "EXTRA=$*"

CACHE_DIR="$HF_HOME/hub/models--google--gemma-3-27b-it"
if [[ ! -d "$CACHE_DIR" ]]; then
    echo "Missing model cache: $CACHE_DIR" >&2
    exit 1
fi
if [[ ! -f data/phase1_translations/metadata.json ]]; then
    echo "Missing frozen translation metadata" >&2
    exit 1
fi

python -m phase1.phase1_layerwise --batch-size 8 "$@"
