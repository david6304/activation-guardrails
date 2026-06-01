#!/usr/bin/env python
"""Prefetch HF models and datasets into the cluster cache (run on the head node).

MLP compute nodes have no internet, so everything the offline GPU jobs need must
be cached on the head node (`hastings`) first. For the CC++ Gemma 3 slice that is
the base model plus Heretic's refusal/KL evaluation datasets. See docs/CLUSTER.md.

This is the online step: it deliberately clears the *_OFFLINE flags so downloads
succeed, and writes into HF_HOME (default ~/models, the persistent cache).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULT_MODELS = ["google/gemma-3-4b-it"]
# Heretic's default good/bad prompt datasets (heretic config.default.toml).
DEFAULT_DATASETS = ["mlabonne/harmless_alpaca", "mlabonne/harmful_behaviors"]


def main() -> int:
    args = parse_args()
    os.environ["HF_HOME"] = str(args.hf_home.expanduser())
    for flag in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ.pop(flag, None)

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    for model_id in args.models:
        print(f"[model]   {model_id}", flush=True)
        snapshot_download(model_id)
    for dataset_id in args.datasets:
        print(f"[dataset] {dataset_id}", flush=True)
        load_dataset(dataset_id)
    print(f"done; cache at {os.environ['HF_HOME']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", type=Path, default=Path("~/models"))
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
