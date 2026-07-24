#!/bin/bash
# Extend the frozen Phase 1 translation bundle on one 11GB-capable Interactive GPU.
# Submit with:
# EXPECTED_COMMIT=$(git rev-parse HEAD) sbatch --export=ALL,EXPECTED_COMMIT \
#   phase1/run_prepare_multilingual_translations.sh [EXTRA...]
#SBATCH --job-name=phase1_translate
#SBATCH --partition=Interactive
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=phase1_translate_%j.out
set -euo pipefail

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source /home/htang2/toolchain-20251006/toolchain.rc
source ~/venvs/ml/bin/activate
cd ~/activation-guardrails

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

: "${EXPECTED_COMMIT:?set EXPECTED_COMMIT to the submitted checkout commit}"
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
if [[ ! -d ~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M ]]; then
    echo "Missing frozen NLLB cache" >&2
    exit 1
fi
if [[ ! -f data/phase1_translations/swahili.jsonl ]]; then
    echo "Missing frozen Swahili manifest" >&2
    exit 1
fi

date --iso-8601=seconds
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "commit=$CURRENT_COMMIT"
python -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__)'
echo "command=python -m phase1.prepare_multilingual_translations $*"

python -m phase1.prepare_multilingual_translations "$@"
