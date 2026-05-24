"""Build the CC++ WildJailbreak prompt split artifact."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agguardrails.wildjailbreak import (  # noqa: E402
    build_and_write_wildjailbreak_contract,
    load_config,
)


DEFAULT_CONFIG = "configs/ccpp/gemma2_9b_it_wildjailbreak.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normalized WildJailbreak CC++ split artifacts.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--local-path",
        help="Optional local JSONL/JSON/CSV/TSV dataset file override.",
    )
    parser.add_argument(
        "--data-path",
        help="Optional normalized JSONL output path override.",
    )
    parser.add_argument(
        "--metadata-path",
        help="Optional metadata JSON output path override.",
    )
    return parser.parse_args()


def load_config_with_overrides(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(load_config(args.config))
    if args.local_path:
        config.setdefault("dataset", {})["local_path"] = args.local_path
    if args.data_path:
        config.setdefault("outputs", {})["data_path"] = args.data_path
    if args.metadata_path:
        config.setdefault("outputs", {})["metadata_path"] = args.metadata_path
    return config


def main() -> None:
    args = parse_args()
    config = load_config_with_overrides(args)
    examples, metadata = build_and_write_wildjailbreak_contract(
        config_path=Path(args.config),
        config=config,
        rows=None,
    )

    outputs = config["outputs"]
    print(f"Wrote {len(examples)} examples to {outputs['data_path']}")
    print(f"Wrote metadata to {outputs['metadata_path']}")
    print(f"Split counts: {metadata['splits']['counts_by_split']}")


if __name__ == "__main__":
    main()
