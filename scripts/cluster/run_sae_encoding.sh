#!/bin/bash
# Backward-compatible wrapper for the SAE encode stage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/run_sae_pipeline.sh" --stage encode "$@"
