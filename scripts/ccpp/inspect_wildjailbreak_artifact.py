"""Inspect a normalized CC++ WildJailbreak artifact for schema/count issues."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agguardrails.response_cache import load_normalized_examples, sha256_file  # noqa: E402
from agguardrails.wildjailbreak import load_config  # noqa: E402


DEFAULT_CONFIG = "configs/ccpp/gemma2_9b_it_wildjailbreak.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect normalized WildJailbreak JSONL and metadata.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument("--data-path", help="Optional normalized JSONL path override.")
    parser.add_argument("--metadata-path", help="Optional metadata JSON path override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_path = Path(args.data_path or config["outputs"]["data_path"])
    metadata_path = Path(args.metadata_path or config["outputs"]["metadata_path"])

    examples = load_normalized_examples(data_path)
    example_ids = [example["example_id"] for example in examples]
    duplicate_count = len(example_ids) - len(set(example_ids))
    split_counts = Counter(example["split"] for example in examples)
    data_type_counts = Counter(example["data_type"] for example in examples)
    split_label_counts = Counter(
        f"{example['split']}:harmful={example['label']}" for example in examples
    )

    print(f"Artifact: {data_path}")
    print(f"SHA256: {sha256_file(data_path)}")
    print(f"Examples: {len(examples)}")
    print(f"Duplicate example_ids: {duplicate_count}")
    print(f"Split counts: {dict(sorted(split_counts.items()))}")
    print(f"Data type counts: {dict(sorted(data_type_counts.items()))}")
    print(f"Split/label counts: {dict(sorted(split_label_counts.items()))}")

    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        print(f"Metadata: {metadata_path}")
        metadata_split_counts = metadata.get("splits", {}).get("counts_by_split")
        print(f"Metadata split counts: {metadata_split_counts}")
    else:
        print(f"Metadata missing: {metadata_path}")

    if duplicate_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
