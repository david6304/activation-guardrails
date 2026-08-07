#!/bin/bash
# P2 guard comparator on MLP: Qwen3Guard-Gen-8B over the reader's response
# prefixes on the K_GRID. Reads only data/p2_latency_scores.npz and the judged
# rows, so no 27B weights are loaded and nothing abliterated is involved.
# Usage: sbatch -p Teaching --gres=gpu:h200_3g.71gb:1 run_p2_guard_mlp.sh
#SBATCH --job-name=p2_guard
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=p2_guard_%j.out
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd "${SLURM_SUBMIT_DIR:?sbatch must run from the intended checkout}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

SCORES=${1:-data/p2_latency_scores.npz}
JUDGED=${2:-data/p2_judged_analysis.jsonl}
OUT=${3:-data/p2_guard_monitor.npz}
BATCH=${4:-8}
LIMIT=${5:-0}
REQFIELD=${6:-prompt}

for path in "$SCORES" "$JUDGED"; do
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
echo "SCORES=$SCORES JUDGED=$JUDGED OUT=$OUT BATCH=$BATCH LIMIT=$LIMIT REQFIELD=$REQFIELD"

python p2_guard_monitor.py --scores "$SCORES" --judged "$JUDGED" \
  --out "$OUT" --batch-size "$BATCH" --limit "$LIMIT" --request-field "$REQFIELD"
