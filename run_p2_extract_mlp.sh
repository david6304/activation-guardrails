#!/bin/bash
# P2 stage 3 on MLP: teacher-force prompt+response through the PROTECTED 27B and
# score every response position with the frozen probe. Mirrors
# run_p2_extract_eddie.sh; MLP is used when Eddie's L40S queue is backed up.
# The protected model is the stock google/gemma-3-27b-it, so nothing abliterated
# is involved and no weights move between clusters.
#SBATCH --job-name=p2_extract
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=p2_extract_%j.out
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd "${SLURM_SUBMIT_DIR:?sbatch must run from the intended checkout}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

JUDGED=${1:-data/p2_judged_v2.jsonl}
OUT=${2:-data/p2_latency_scores.npz}
BATCH=${3:-2}
LIMIT=${4:-0}

for path in "$JUDGED" data/phase1_activation_multilingual_27b.npz; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required input: $path" >&2
        exit 1
    fi
done

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short
echo "JUDGED=$JUDGED OUT=$OUT BATCH=$BATCH LIMIT=$LIMIT"

python p2_extract_latency.py --in "$JUDGED" --out "$OUT" --batch-size "$BATCH" --limit "$LIMIT"
