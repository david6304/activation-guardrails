#!/bin/bash
# P2 stage 1: responses to the frozen split under {plain, base64}, from the
# abliterated 27B. MLP only -- the abliterated weights never leave this cluster;
# only the response JSONL moves to Eddie for teacher-forced extraction.
# Resumable: re-running the same command skips ids already in $OUT.
# Pilot: sbatch -p Wintermute --gres=gpu:2 --time=02:00:00 run_p2_generate.sh \
#          data/p2_pilot_prompts.jsonl data/p2_pilot_responses.jsonl 16
#SBATCH --job-name=p2_generate
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=p2_generate_%j.out
set -euo pipefail

# toolchain.rc reads LD_LIBRARY_PATH unguarded, which is fatal under `set -u`.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

PROMPTS=${1:-data/p2_pilot_prompts.jsonl}
OUT=${2:-data/p2_pilot_responses.jsonl}
BATCH=${3:-16}
MODEL=${4:-$HOME/models/gemma-3-27b-it-heretic}

for path in "$PROMPTS" "$MODEL"; do
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
echo "PROMPTS=$PROMPTS OUT=$OUT BATCH=$BATCH MODEL=$MODEL"

python generate_responses.py \
  --prompts "$PROMPTS" \
  --model "$MODEL" \
  --batch-size "$BATCH" \
  --max-new-tokens 512 \
  --out "$OUT"
