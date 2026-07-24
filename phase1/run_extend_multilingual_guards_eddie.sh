#!/bin/bash
# Audit frozen guard scores and score French/Hindi/Zulu on one Eddie L40S.
#$ -N phase1_multiguard
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l l40s=true
#$ -pe sharedmem 4
#$ -l h_rss=32G
#$ -l h_rt=03:00:00
#$ -o phase1_multiguard_$JOB_ID.out
#$ -j y
set -euo pipefail

. /etc/profile.d/modules.sh
module load cuda/12.1.1
source /exports/eddie/scratch/s2296274/venv/bin/activate

export HF_HOME=/exports/eddie/scratch/s2296274/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd "${SGE_O_WORKDIR:?qsub must run from the intended checkout}"

: "${EXPECTED_COMMIT:?submit with qsub -v EXPECTED_COMMIT=<commit>}"
CURRENT_COMMIT="$(git rev-parse HEAD)"
if [[ "$CURRENT_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    echo "Checkout mismatch: current=$CURRENT_COMMIT expected=$EXPECTED_COMMIT" >&2
    exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked checkout is dirty" >&2
    git status --short
    exit 1
fi

BASELINES_SOURCE="${BASELINES_SOURCE:-data/phase1_baselines_multilingual_tfidf.npz}"
OUTPUT_PATH="${OUTPUT_PATH:-data/phase1_baselines_multilingual_${JOB_ID}.npz}"
for path in \
    "$HF_HOME/hub/models--google--shieldgemma-9b" \
    "$HF_HOME/hub/models--allenai--wildguard" \
    "$BASELINES_SOURCE" \
    data/phase1_baselines.npz \
    data/phase1_baselines.json \
    data/phase1_translations/french.jsonl \
    data/phase1_translations/hindi.jsonl \
    data/phase1_translations/swahili.jsonl \
    data/phase1_translations/zulu.jsonl; do
    if [[ ! -e "$path" ]]; then
        echo "Missing required input: $path" >&2
        exit 1
    fi
done

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "commit=$CURRENT_COMMIT"
python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'
echo "command=python -m phase1.extend_multilingual_guards --source $BASELINES_SOURCE --out $OUTPUT_PATH"

python -m phase1.extend_multilingual_guards \
    --source "$BASELINES_SOURCE" \
    --out "$OUTPUT_PATH"
