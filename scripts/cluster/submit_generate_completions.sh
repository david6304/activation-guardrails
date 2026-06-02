#!/usr/bin/env bash
# Submit the CC++ on-policy completion-generation stage to SLURM (MLP cluster).
#
# Offline GPU job: assumes the abliterated model and any HF datasets are already
# cached locally via scripts/cluster/prefetch_hf.py (compute nodes have no
# internet). See docs/CLUSTER.md. Configurable, with a --dry-run that prints the
# exact sbatch command instead of submitting.
set -euo pipefail

PARTITION=Teaching
GPU_TYPE=a6000                     # a6000 | a40 | any  (Teaching has A6000 on landonia11; no A40)
MODEL_ID="$HOME/models/gemma-3-4b-it-heretic"
REPO_DIR="$HOME/activation-guardrails"
HF_HOME_DIR="$HOME/models"
POS_MANIFEST="data/interim/ccpp/clearharm_generation_prompts.unique.jsonl"
BENIGN_MANIFEST="data/interim/ccpp/matched_benign_prompts.jsonl"
OUT_DIR="data/processed/ccpp"
SEED=0
LIMIT=""
STAGE=both                        # positives | benign | both
MAX_NEW_TOKENS=512
DRY_RUN=0

usage() {
  sed -n '2,12p' "$0"
  cat <<'EOF'

Options (all optional):
  --partition NAME        SLURM partition (default: Teaching)
  --gpu-type TYPE         a40 | a6000 | any (default: a40; never bare gpu:1)
  --model-id PATH         abliterated model dir/id (default: ~/models/gemma-3-4b-it-heretic)
  --repo-dir PATH         repo checkout (default: ~/activation-guardrails)
  --hf-home PATH          HF cache (default: ~/models)
  --positives PATH        positive prompt manifest
  --benign PATH           benign prompt manifest
  --out-dir PATH          output dir for exchange JSONL (default: data/processed/ccpp)
  --stage STAGE           positives | benign | both (default: both)
  --seed N                generation seed (default: 0)
  --limit N               cap prompts per manifest (smoke test)
  --max-new-tokens N      decode length (default: 512)
  --dry-run               print the sbatch command, do not submit
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2;;
    --gpu-type) GPU_TYPE="$2"; shift 2;;
    --model-id) MODEL_ID="$2"; shift 2;;
    --repo-dir) REPO_DIR="$2"; shift 2;;
    --hf-home) HF_HOME_DIR="$2"; shift 2;;
    --positives) POS_MANIFEST="$2"; shift 2;;
    --benign) BENIGN_MANIFEST="$2"; shift 2;;
    --out-dir) OUT_DIR="$2"; shift 2;;
    --stage) STAGE="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --limit) LIMIT="$2"; shift 2;;
    --max-new-tokens) MAX_NEW_TOKENS="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "unknown option: $1" >&2; usage; exit 1;;
  esac
done

NODELIST=""
case "$GPU_TYPE" in
  a6000) GRES="gpu:nvidia_rtx_a6000:1"; NODELIST="landonia11";;
  a40) GRES="gpu:a40:1";;
  any) GRES="gpu:1"; NODELIST="landonia11";;  # only capable GPU in Teaching
  *) echo "unsupported --gpu-type: $GPU_TYPE (a6000|a40|any)" >&2; exit 1;;
esac

LIMIT_ARG=""
[[ -n "$LIMIT" ]] && LIMIT_ARG="--limit $LIMIT"

gen_cmd() {  # $1 = manifest, $2 = output basename
  echo "python scripts/ccpp/generate_completions.py" \
       "--manifest $1 --output $OUT_DIR/$2" \
       "--model-id '$MODEL_ID' --backend transformers" \
       "--seed $SEED --max-new-tokens $MAX_NEW_TOKENS $LIMIT_ARG"
}

CMDS="source /home/htang2/toolchain-20251006/toolchain.rc"
CMDS="$CMDS && source \$HOME/venvs/ml/bin/activate"
CMDS="$CMDS && export HF_HOME='$HF_HOME_DIR' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1"
CMDS="$CMDS && cd '$REPO_DIR'"
case "$STAGE" in
  positives) CMDS="$CMDS && $(gen_cmd "$POS_MANIFEST" positive_exchanges.jsonl)";;
  benign) CMDS="$CMDS && $(gen_cmd "$BENIGN_MANIFEST" benign_exchanges.jsonl)";;
  both)
    CMDS="$CMDS && $(gen_cmd "$POS_MANIFEST" positive_exchanges.jsonl)"
    CMDS="$CMDS && $(gen_cmd "$BENIGN_MANIFEST" benign_exchanges.jsonl)";;
  *) echo "unsupported --stage: $STAGE (positives|benign|both)" >&2; exit 1;;
esac

SBATCH=(sbatch -p "$PARTITION" --gres="$GRES" -J ccpp_gen -o "slurm-ccpp-gen-%j.out")
[[ -n "$NODELIST" ]] && SBATCH+=(--nodelist="$NODELIST")
SBATCH+=(--wrap "$CMDS")

if [[ "$DRY_RUN" == "1" ]]; then
  printf '%q ' "${SBATCH[@]}"; echo
else
  "${SBATCH[@]}"
fi
