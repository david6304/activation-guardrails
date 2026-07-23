#!/bin/bash
# Frozen two-layer dense-versus-SAE Phase 3 run on Eddie.
# Smoke:
#   qsub -l h_rt=01:00:00 run_phase3_eddie.sh --smoke --batch-size 4 \
#     --out data/phase3_smoke.npz
# Reportable:
#   qsub run_phase3_eddie.sh --batch-size 8
#$ -N phase3_sae
#$ -cwd
#$ -q gpu
#$ -l gpu=2
#$ -l l40s=true
#$ -pe sharedmem 12
#$ -l h_rss=16G
#$ -l h_rt=04:00:00
#$ -o phase3_sae_$JOB_ID.out
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
if git rev-parse HEAD >/dev/null 2>&1; then
    git rev-parse HEAD
    git status --short
else
    cat LOCAL_COMMIT.txt
fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "EXTRA=$*"

for cache in \
    "$HF_HOME/hub/models--google--gemma-3-27b-it" \
    "$HF_HOME/hub/models--google--gemma-scope-2-27b-it"
do
    if [[ ! -d "$cache" ]]; then
        echo "Missing model cache: $cache" >&2
        exit 1
    fi
done

python -m phase3.phase3_sae "$@"

echo "peak_rss_kb=$(grep VmHWM /proc/self/status 2>/dev/null || echo n/a)"
