"""Generate a CC++ response cache for normalized WildJailbreak examples."""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import copy
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from agguardrails.response_cache import (  # noqa: E402
    build_response_cache_metadata,
    load_cached_example_ids,
    load_normalized_examples,
    load_response_cache_records,
    make_response_cache_record,
    select_examples_for_response_cache,
    sha256_text,
    write_response_cache,
)
from agguardrails.wildjailbreak import load_config  # noqa: E402


DEFAULT_CONFIG = "configs/ccpp/gemma2_9b_it_wildjailbreak.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate/cache Gemma responses for normalized WildJailbreak.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config. Default: {DEFAULT_CONFIG}",
    )
    parser.add_argument("--data-path", help="Optional normalized JSONL input override.")
    parser.add_argument("--cache-path", help="Optional response JSONL output override.")
    parser.add_argument(
        "--metadata-path",
        help="Optional metadata JSON output override.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional debug subset limit override.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        help="Optional debug subset split override. Pass no values for no splits.",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Write deterministic mock responses instead of loading a model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected examples without writing cache artifacts.",
    )
    return parser.parse_args()


def load_config_with_overrides(args: argparse.Namespace) -> dict:
    config = copy.deepcopy(load_config(args.config))
    outputs = config.setdefault("outputs", {})
    if args.data_path:
        outputs["data_path"] = args.data_path
    if args.cache_path:
        outputs["response_cache_path"] = args.cache_path
    if args.metadata_path:
        outputs["response_cache_metadata_path"] = args.metadata_path
    return config


def response_cache_selection(
    config: dict,
    *,
    limit_override: int | None,
    splits_override: list[str] | None,
) -> tuple[int | None, list[str] | None]:
    cache_config = config.get("response_cache", {})
    debug_subset = cache_config.get("debug_subset", {})
    if debug_subset.get("enabled", False):
        limit = debug_subset.get("limit")
        splits = debug_subset.get("splits")
    else:
        limit = None
        splits = None

    if limit_override is not None:
        limit = limit_override
    if splits_override is not None:
        splits = splits_override or None
    return limit, splits


def utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def mock_response(example: dict) -> str:
    prompt_hash = sha256_text(str(example["prompt"]))[:12]
    return (
        f"[mock response] example_id={example['example_id']} "
        f"prompt_sha256={prompt_hash}"
    )


def main() -> None:
    args = parse_args()
    config = load_config_with_overrides(args)
    outputs = config["outputs"]
    data_path = Path(outputs["data_path"])
    cache_path = Path(outputs["response_cache_path"])
    metadata_path = Path(outputs["response_cache_metadata_path"])
    limit, splits = response_cache_selection(
        config,
        limit_override=args.limit,
        splits_override=args.splits,
    )

    examples = load_normalized_examples(data_path)
    cached_ids = load_cached_example_ids(cache_path)
    selected = select_examples_for_response_cache(
        examples,
        seed=int(config["generation"].get("seed", config["sampling"]["seed"])),
        limit=limit,
        splits=splits,
        already_cached_example_ids=cached_ids,
    )

    print(f"Loaded {len(examples)} normalized examples from {data_path}")
    print(f"Existing cached examples: {len(cached_ids)}")
    print(f"Selected {len(selected)} uncached examples")

    if args.dry_run:
        for example in selected:
            print(
                f"{example['split']}\t{example['data_type']}\t{example['example_id']}"
            )
        return

    if not args.mock:
        raise SystemExit(
            "Real model generation is intentionally gated for this slice. "
            "Re-run with --mock to write a schema/debug cache."
        )

    generated_at = utc_timestamp()
    existing_records = load_response_cache_records(cache_path)
    new_records = [
        make_response_cache_record(
            example,
            response=mock_response(example),
            config=config,
            generated_at=generated_at,
        )
        for example in selected
    ]
    records = [*existing_records, *new_records]
    metadata = build_response_cache_metadata(
        records,
        config=config,
        config_path=Path(args.config),
        dataset_artifact_path=data_path,
        response_cache_path=cache_path,
        created_at=generated_at,
    )
    metadata["mode"] = "mock"
    metadata["selection"] = {
        "debug_subset_enabled": bool(
            config.get("response_cache", {})
            .get("debug_subset", {})
            .get("enabled", False)
        ),
        "limit": limit,
        "splits": splits,
    }
    metadata["resume"] = {
        "existing_records": len(existing_records),
        "new_records": len(new_records),
        "skipped_cached_example_ids": len(cached_ids),
    }

    write_response_cache(
        records,
        cache_path=cache_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )

    print(f"Wrote {len(new_records)} new responses to {cache_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
