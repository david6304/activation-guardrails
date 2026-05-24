"""Extract/cache CC++ activation features from cached prompt/response exchanges."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agguardrails.activation_cache import (  # noqa: E402
    build_activation_cache_metadata,
    build_mock_activation_cache,
    write_activation_cache,
)
from agguardrails.response_cache import load_response_cache_records  # noqa: E402
from agguardrails.wildjailbreak import load_config  # noqa: E402


DEFAULT_CONFIG = "configs/ccpp/gemma2_9b_it_wildjailbreak.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract/cache activation features from response caches.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument(
        "--response-cache-path",
        help="Optional response JSONL input override.",
    )
    parser.add_argument(
        "--activation-cache-path",
        help="Optional activation NPZ output override.",
    )
    parser.add_argument(
        "--index-path",
        help="Optional activation index JSONL output override.",
    )
    parser.add_argument(
        "--metadata-path",
        help="Optional activation metadata JSON output override.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Write deterministic mock activations instead of loading a model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print cached examples that would be featurized without writing.",
    )
    return parser.parse_args()


def load_config_with_overrides(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(load_config(args.config))
    outputs = config.setdefault("outputs", {})
    if args.response_cache_path:
        outputs["response_cache_path"] = args.response_cache_path
    if args.activation_cache_path:
        outputs["activation_cache_path"] = args.activation_cache_path
    if args.index_path:
        outputs["activation_index_path"] = args.index_path
    if args.metadata_path:
        outputs["activation_metadata_path"] = args.metadata_path
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
    response_cache_path = Path(outputs["response_cache_path"])
    activation_cache_path = Path(outputs["activation_cache_path"])
    index_path = Path(outputs["activation_index_path"])
    metadata_path = Path(outputs["activation_metadata_path"])

    response_records = load_response_cache_records(response_cache_path)
    print(f"Loaded {len(response_records)} response records from {response_cache_path}")

    if args.dry_run:
        for record in response_records:
            print(f"{record.split}\t{record.data_type}\t{record.example_id}")
        return

    if not args.mock:
        raise SystemExit(
            "Activation extraction is intentionally gated. Re-run with --mock "
            "for a schema/debug cache."
        )

    activation_records, activations = build_mock_activation_cache(
        response_records,
        config=config,
    )
    created_at = utc_timestamp()
    metadata = build_activation_cache_metadata(
        activation_records,
        activations,
        config=config,
        config_path=Path(args.config),
        dataset_artifact_path=outputs["data_path"],
        response_cache_path=response_cache_path,
        activation_cache_path=activation_cache_path,
        index_path=index_path,
        created_at=created_at,
    )
    metadata["mode"] = "mock"

    write_activation_cache(
        activation_records,
        activations,
        activation_cache_path=activation_cache_path,
        index_path=index_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )
    print(
        f"Wrote activation array {list(activations.shape)} "
        f"to {activation_cache_path}"
    )
    print(f"Wrote activation index to {index_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
