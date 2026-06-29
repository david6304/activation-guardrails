#!/bin/bash
# Forward-only probe scoring (protected Gemma 3 12B, teacher-forced, no cache). Resumable:
# re-running the same command skips ids already in $OUT, so a timeout/preempt loses no work.
# Usage: sbatch run_score.sh <IN> <OUT> [PROBE] [BATCH] [LIMIT] [EXTRA...]   (override target with -p/--gres/--time)
#   extra args from position 6 pass through, e.g. --model-id /home/s2296274/models/gemma-3-12b-it-heretic
#   calibration: ... run_score.sh data/wildchat_calib.jsonl data/wildchat_scores.jsonl data/probe_v1.pt 8
#   eval:        ... run_score.sh data/judged_adv_eval.jsonl data/adv_eval_scores.jsonl  data/probe_v1.pt 8
#SBATCH --job-name=score
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=score_%j.out
set -euo pipefail

source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

IN=${1:?need IN}
OUT=${2:?need OUT}
PROBE=${3:-data/probe_v1.pt}
BATCH=${4:-8}
LIMIT=${5:-0}

date --iso-8601=seconds
hostname
git rev-parse HEAD
git status --short
echo "IN=$IN OUT=$OUT PROBE=$PROBE BATCH=$BATCH LIMIT=$LIMIT"

python score_probe.py --in "$IN" --out "$OUT" --probe "$PROBE" --batch-size "$BATCH" --limit "$LIMIT" "${@:6}"
