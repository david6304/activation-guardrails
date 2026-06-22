#!/bin/bash
# Scaled response generation (abliterated Gemma 3 12B, 512 tok). Resumable: re-running
# the same command skips ids already in $OUT, so a timeout/preempt loses no work.
# Usage: sbatch -p Wintermute --gres=gpu:1 --time=12:00:00 run_generate.sh [N_PER_TYPE] [OUT] [BATCH] [DATA_TYPES] [PROMPT_COL]
# Defaults to the vanilla train split; for the adversarial eval split pass e.g.
#   ... run_generate.sh 5000 data/responses_eval.jsonl 64 adversarial_harmful,adversarial_benign adversarial
#SBATCH --job-name=gen_train
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=gen_train_%j.out
set -euo pipefail

source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

N_PER_TYPE=${1:-5000}
OUT=${2:-data/responses_train.jsonl}
BATCH=${3:-64}
DATA_TYPES=${4:-vanilla_harmful,vanilla_benign}
PROMPT_COL=${5:-vanilla}

date --iso-8601=seconds
hostname
git rev-parse HEAD
git status --short
echo "N_PER_TYPE=$N_PER_TYPE OUT=$OUT BATCH=$BATCH DATA_TYPES=$DATA_TYPES PROMPT_COL=$PROMPT_COL"

python generate_responses.py \
  --data-types "$DATA_TYPES" --prompt-col "$PROMPT_COL" \
  --n-per-type "$N_PER_TYPE" --batch-size "$BATCH" --max-new-tokens 512 --out "$OUT"
