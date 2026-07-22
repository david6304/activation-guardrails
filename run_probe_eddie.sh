#!/bin/bash
# Plain-trained input-only prompt probe on protected Gemma-3 (Eddie / Grid Engine).
# Slurm twin: run_probe.sh. See docs/EDDIE.md.
# Usage: qsub run_probe_eddie.sh [MODEL] [EXTRA...]
# Prepare the shared frozen translations once before either definitive model run:
#   qsub run_probe_eddie.sh google/gemma-3-12b-it --prepare-translations
# Smoke:  qsub run_probe_eddie.sh google/gemma-3-12b-it --limit 60 --batch-size 4
#$ -N probe_prompt
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l l40s=true
#$ -pe sharedmem 8
#$ -l h_rss=16G
#$ -l h_rt=08:00:00
#$ -o probe_prompt_$JOB_ID.out
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

cd /exports/eddie/scratch/s2296274/activation-guardrails

MODEL=${1:-google/gemma-3-27b-it}

date --iso-8601=seconds
hostname
nvidia-smi -L
git rev-parse HEAD || true
cat LOCAL_COMMIT.txt 2>/dev/null || true
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "MODEL=$MODEL EXTRA=${*:2}"

# Fail early rather than 30 min into a cold load.
CACHE_DIR="$HF_HOME/hub/models--${MODEL//\//--}"
if [[ ! -d "$CACHE_DIR" ]]; then
    echo "Missing model cache: $CACHE_DIR" >&2
    exit 1
fi

python probe_prompt.py --model "$MODEL" "${@:2}"

# Peak RSS: x_train is ~8 GB at 27B, so watch this before scaling the run up.
echo "peak_rss_kb=$(grep VmHWM /proc/self/status 2>/dev/null || echo n/a)"
