"""Train/evaluate a dense activation logistic probe."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agguardrails.dense_probe import (  # noqa: E402
    load_activation_cache_arrays,
    load_and_run_dense_probe,
    split_activation_records,
)
from agguardrails.wildjailbreak import load_config  # noqa: E402


DEFAULT_CONFIG = "configs/ccpp/gemma2_9b_it_wildjailbreak.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a dense logistic probe on vanilla-train activations, select a "
            "validation threshold, and report vanilla/adversarial transfer."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--activation-cache-path",
        help="Optional activation NPZ input override.",
    )
    parser.add_argument(
        "--index-path",
        help="Optional activation index JSONL input override.",
    )
    parser.add_argument(
        "--response-cache-path",
        help="Optional response JSONL provenance override.",
    )
    parser.add_argument("--model-path", help="Optional joblib model output override.")
    parser.add_argument("--scores-path", help="Optional JSONL score output override.")
    parser.add_argument("--metrics-path", help="Optional JSON metrics output override.")
    parser.add_argument("--table-path", help="Optional CSV table output override.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print split counts without training or writing outputs.",
    )
    return parser.parse_args()


def load_config_with_overrides(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(load_config(args.config))
    outputs = config.setdefault("outputs", {})
    if args.activation_cache_path:
        outputs["activation_cache_path"] = args.activation_cache_path
    if args.index_path:
        outputs["activation_index_path"] = args.index_path
    if args.response_cache_path:
        outputs["response_cache_path"] = args.response_cache_path
    if args.model_path:
        outputs["dense_probe_model_path"] = args.model_path
    if args.scores_path:
        outputs["dense_probe_scores_path"] = args.scores_path
    if args.metrics_path:
        outputs["dense_probe_metrics_path"] = args.metrics_path
    if args.table_path:
        outputs["dense_probe_table_path"] = args.table_path
    return config


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def main() -> None:
    args = parse_args()
    config = load_config_with_overrides(args)
    outputs = config["outputs"]
    activation_cache_path = Path(outputs["activation_cache_path"])
    activation_index_path = Path(outputs["activation_index_path"])

    if args.dry_run:
        activations, records = load_activation_cache_arrays(
            activation_cache_path,
            activation_index_path,
        )
        groups = split_activation_records(records)
        print(f"Loaded activation array {list(activations.shape)}")
        print(f"Loaded {len(records)} activation records from {activation_index_path}")
        for split in ("train", "val", "test", "transfer"):
            print(f"{split}\t{len(groups[split])}")
        return

    metadata = load_and_run_dense_probe(
        config=config,
        config_path=Path(args.config),
        created_at=utc_timestamp(),
    )
    print(f"Trained {metadata['probe']} from {activation_cache_path}")
    print(f"Selected threshold {metadata['threshold_rule']['frozen_threshold']:.6g}")
    print(f"Wrote model to {metadata['artifacts']['model_path']}")
    print(f"Wrote scores to {metadata['artifacts']['scores_path']}")
    print(f"Wrote metrics to {metadata['artifacts']['metrics_path']}")
    print(f"Wrote table to {metadata['artifacts']['table_path']}")


if __name__ == "__main__":
    main()
