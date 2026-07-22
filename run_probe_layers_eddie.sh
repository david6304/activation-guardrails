#!/bin/bash
# Per-layer operational-harm diagnostic on protected Gemma-3 27B (Eddie).
# Usage: qsub run_probe_layers_eddie.sh [EXTRA...]
# Smoke: qsub -l h_rt=01:00:00 run_probe_layers_eddie.sh --limit 60 \
#          --translations-dir data/probe_prompt_translations_limit60 \
#          --scores-out data/probe_prompt_layers_gemma-3-27b-it_smoke_scores.npz \
#          --report-out data/probe_prompt_layers_gemma-3-27b-it_smoke_report.json
#$ -N probe_layers
#$ -cwd
#$ -q gpu
#$ -l gpu=2
#$ -l l40s=true
#$ -pe sharedmem 12
#$ -l h_rss=16G
#$ -l h_rt=08:00:00
#$ -o probe_layers_$JOB_ID.out
#$ -j y
set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda/12.1.1
source /exports/eddie/scratch/s2296274/venv/bin/activate

export HF_HOME=/exports/eddie/scratch/s2296274/hf
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${SGE_O_WORKDIR:?qsub must be run from the intended repository checkout}"

date --iso-8601=seconds
hostname
nvidia-smi -L
git rev-parse HEAD
git status --short
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "EXTRA=$*"

CACHE_DIR="$HF_HOME/hub/models--google--gemma-3-27b-it"
if [[ ! -d "$CACHE_DIR" ]]; then
    echo "Missing model cache: $CACHE_DIR" >&2
    exit 1
fi

python probe_prompt_layers.py "$@"

echo "peak_rss_kb=$(grep VmHWM /proc/self/status 2>/dev/null || echo n/a)"
