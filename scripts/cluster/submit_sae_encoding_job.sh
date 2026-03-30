#!/bin/bash
# Backward-compatible wrapper for the SAE encode-only SLURM submission.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/submit_sae_job.sh" --stage encode "$@"
