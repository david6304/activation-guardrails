#!/usr/bin/env python
"""Build the approved non-reportable WildJailbreak smoke manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agguardrails.wildjailbreak_manifest import (
    DATASET_ID,
    DATASET_REVISION,
    SOURCE_CONFIG,
    SOURCE_SPLIT,
    build_manifest,
    validate_source_schema,
    write_manifest,
)


def main() -> int:
    args = _parse_args()
    source_rows = _load_source_rows()
    rows, metadata = build_manifest(source_rows, seed=args.seed)
    write_manifest(
        rows,
        metadata,
        output_path=args.output,
        metadata_path=args.metadata_output,
    )
    summary = {
        "manifest": str(args.output),
        "metadata": str(args.metadata_output),
        "selected_rows": metadata["counts"]["selected_rows"],
        "split_rows": metadata["counts"]["split_rows"],
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


def _load_source_rows():
    from datasets import load_dataset

    rows = load_dataset(
        DATASET_ID,
        SOURCE_CONFIG,
        split=SOURCE_SPLIT,
        revision=DATASET_REVISION,
        delimiter="\t",
        keep_default_na=False,
    )
    validate_source_schema(rows.features)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260612)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/wildjailbreak_smoke_manifest.jsonl"),
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=Path("data/processed/wildjailbreak_smoke_manifest.metadata.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
