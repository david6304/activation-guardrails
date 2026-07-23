#!/bin/bash
# Frozen two-layer dense-versus-SAE Phase 3 run on MLP.
# Smoke:
#   sbatch --partition=Teaching --gres=gpu:nvidia_rtx_a6000:2 \
#     --nodelist=landonia11 run_phase3_mlp.sh --smoke --batch-size 4 \
#     --out data/phase3_smoke.npz
#SBATCH --job-name=phase3_sae
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=phase3_sae_%j.out
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${SLURM_SUBMIT_DIR:?sbatch must be run from the intended repository checkout}"

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
    "$HOME/.cache/huggingface/hub/models--google--gemma-3-27b-it" \
    "$HOME/.cache/huggingface/hub/models--google--gemma-scope-2-27b-it"
do
    if [[ ! -d "$cache" ]]; then
        echo "Missing model cache: $cache" >&2
        exit 1
    fi
done

python -m phase3.phase3_sae "$@"
