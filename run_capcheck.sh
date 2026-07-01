#!/bin/bash
# Cipher gist capability check (default input-only: cipher in, plain out; reverse+rot13).
# One model per submit; parametrised by model + tag. Overwrites $OUT each run (small job).
# Usage: sbatch -p Wintermute --gres=gpu:1 --time=00:30:00 run_capcheck.sh MODEL TAG [N] [OUT] [BATCH] [MAXTOK] [CIPHERS] [CONDS]
#   e.g. sbatch ... run_capcheck.sh ~/models/gemma-3-4b-it-heretic  ablit4b  30 data/cap_4b.jsonl
#        sbatch ... run_capcheck.sh ~/models/gemma-3-12b-it-heretic ablit12b 30 data/cap_12b.jsonl
#        sbatch ... run_capcheck.sh ~/models/gemma-3-27b-it-heretic ablit27b 30 data/cap_27b.jsonl
#SBATCH --job-name=capcheck
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=capcheck_%j.out
set -euo pipefail

source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

MODEL=${1:?pass MODEL path}
TAG=${2:?pass TAG}
N=${3:-30}
OUT=${4:-data/cap_${TAG}.jsonl}
BATCH=${5:-16}
MAXTOK=${6:-512}
CIPHERS=${7:-reverse,rot13}
CONDS=${8:-in}

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git rev-parse HEAD
git status --short
echo "MODEL=$MODEL TAG=$TAG N=$N OUT=$OUT BATCH=$BATCH MAXTOK=$MAXTOK CIPHERS=$CIPHERS CONDS=$CONDS"

python capability_check.py --model "$MODEL" --model-tag "$TAG" --n "$N" \
  --ciphers "$CIPHERS" --conds "$CONDS" \
  --batch-size "$BATCH" --max-new-tokens "$MAXTOK" --out "$OUT"
