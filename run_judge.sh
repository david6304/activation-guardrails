#!/bin/bash
# Judge generated responses for harm (Qwen3.6-27B rubric). Resumable: re-running the
# same command skips ids already in $OUT, so a timeout/preempt loses no work.
# Usage: sbatch run_judge.sh   (override target with e.g. -p/--gres/--time on the CLI)
#SBATCH --job-name=judge_train
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_3g.71gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=judge_train_%j.out
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

IN=${1:-data/responses_train.jsonl}
OUT=${2:-data/judged_train.jsonl}
BATCH=${3:-16}
MAXTOK=${4:-64}
MODE=${5:-response}  # 'prompt' judges harmful intent of the prompt alone (input-only venue)

date --iso-8601=seconds
hostname
git rev-parse HEAD
git status --short
echo "IN=$IN OUT=$OUT BATCH=$BATCH MAXTOK=$MAXTOK MODE=$MODE"

python judge_responses.py --in "$IN" --out "$OUT" --batch-size "$BATCH" --max-new-tokens "$MAXTOK" --mode "$MODE"
