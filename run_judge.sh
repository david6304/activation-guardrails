#!/bin/bash
#SBATCH --job-name=judge_sweep
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=judge_sweep_%j.out
set -euo pipefail

source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

IN=data/responses_train_pilot.jsonl

python judge_responses.py --in "$IN" --out data/judged_v2_512.jsonl
python judge_responses.py --in "$IN" --out data/judged_v2_384.jsonl --truncate-tokens 384
python judge_responses.py --in "$IN" --out data/judged_v2_256.jsonl --truncate-tokens 256
