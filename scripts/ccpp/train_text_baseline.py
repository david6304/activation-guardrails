"""Train/evaluate a cheap TF-IDF logistic regression baseline."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agguardrails.text_baseline import (  # noqa: E402
    load_and_run_tfidf_logistic_baseline,
    split_baseline_examples,
)
from agguardrails.response_cache import load_normalized_examples  # noqa: E402
from agguardrails.wildjailbreak import load_config  # noqa: E402


DEFAULT_CONFIG = "configs/ccpp/gemma2_9b_it_wildjailbreak.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train TF-IDF logistic regression on vanilla train, select a "
            "validation threshold, and report vanilla/adversarial transfer."
        ),
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument("--data-path", help="Optional normalized JSONL input override.")
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
    if args.data_path:
        outputs["data_path"] = args.data_path
    if args.model_path:
        outputs["text_baseline_model_path"] = args.model_path
    if args.scores_path:
        outputs["text_baseline_scores_path"] = args.scores_path
    if args.metrics_path:
        outputs["text_baseline_metrics_path"] = args.metrics_path
    if args.table_path:
        outputs["text_baseline_table_path"] = args.table_path
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
    data_path = Path(outputs["data_path"])

    if args.dry_run:
        examples = load_normalized_examples(data_path)
        groups = split_baseline_examples(examples)
        print(f"Loaded {len(examples)} normalized examples from {data_path}")
        for split in ("train", "val", "test", "transfer"):
            print(f"{split}\t{len(groups[split])}")
        return

    metadata = load_and_run_tfidf_logistic_baseline(
        config=config,
        config_path=Path(args.config),
        created_at=utc_timestamp(),
    )
    print(f"Trained {metadata['baseline']} from {data_path}")
    print(f"Selected threshold {metadata['threshold_rule']['frozen_threshold']:.6g}")
    print(f"Wrote model to {metadata['artifacts']['model_path']}")
    print(f"Wrote scores to {metadata['artifacts']['scores_path']}")
    print(f"Wrote metrics to {metadata['artifacts']['metrics_path']}")
    print(f"Wrote table to {metadata['artifacts']['table_path']}")


if __name__ == "__main__":
    main()
