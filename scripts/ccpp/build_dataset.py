#!/usr/bin/env python
"""Build or inspect the normalized dataset for the CC++ replication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from agguardrails.ccpp_data import (
    DatasetGateError,
    apply_grouped_split,
    dataset_metadata,
    load_normalized_jsonl,
    validate_dataset_gates,
    write_jsonl,
    write_metadata,
)


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    dataset_config = config["dataset"]
    source_inspection = (
        inspect_configured_sources(dataset_config) if args.inspect_sources else []
    )

    if args.input_jsonl is None:
        metadata = {
            "config_path": str(args.config),
            "blocked": True,
            "blocked_reason": (
                "No curated input JSONL was provided. Inspect candidate public "
                "sources first, then pass normalized/curated exchanges with "
                "--input-jsonl. The builder must not create Gemma refusal "
                "positives or synthetic harmful completions implicitly."
            ),
            "source_inspection": source_inspection,
        }
        write_metadata(args.metadata_output, metadata)
        raise SystemExit(
            "blocked: provide --input-jsonl after confirming harmful/compliant "
            "assistant completions"
        )

    examples = load_normalized_jsonl(args.input_jsonl)
    split_config = dataset_config["split"]
    examples = apply_grouped_split(
        examples,
        seed=int(split_config["seed"]),
        train_fraction=float(split_config["train"]),
        val_fraction=float(split_config["val"]),
        test_fraction=float(split_config["test"]),
    )
    validate_dataset_gates(
        examples,
        on_policy_config=dataset_config.get("on_policy_generation_gate"),
        length_balance_config=dataset_config.get("length_balance_gate"),
    )

    write_jsonl(args.output, examples)
    metadata = dataset_metadata(
        examples,
        config_path=str(args.config),
        source_inspection=source_inspection,
    )
    write_metadata(args.metadata_output, metadata)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ccpp/gemma3_4b_it_public_cbrn_probe.yaml"),
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Curated records in the normalized schema, before split assignment.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Normalized output JSONL. Defaults to dataset.normalized_path.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Metadata output JSON. Defaults to dataset.metadata_path.",
    )
    parser.add_argument(
        "--inspect-sources",
        action="store_true",
        help="Inspect configured Hugging Face source schemas before building.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_config = config["dataset"]
    args.output = args.output or Path(dataset_config["normalized_path"])
    args.metadata_output = args.metadata_output or Path(dataset_config["metadata_path"])
    return args


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return loaded


def inspect_configured_sources(dataset_config: dict[str, Any]) -> list[dict[str, Any]]:
    inspections = []
    for source_role, sources in dataset_config.get("sources", {}).items():
        for source in sources:
            inspections.append(inspect_hf_dataset(source["id"], source_role))
    return inspections


def inspect_hf_dataset(dataset_id: str, source_role: str) -> dict[str, Any]:
    """Return lightweight HF dataset metadata without loading rows."""

    try:
        from datasets import get_dataset_config_names, get_dataset_split_names
    except ImportError as exc:
        return {
            "dataset_id": dataset_id,
            "source_role": source_role,
            "available": False,
            "error": f"datasets import failed: {exc}",
        }

    inspection: dict[str, Any] = {
        "dataset_id": dataset_id,
        "source_role": source_role,
        "available": True,
        "configs": [],
    }
    try:
        configs = get_dataset_config_names(dataset_id)
    except Exception as exc:  # pragma: no cover - network/schema dependent.
        inspection["available"] = False
        inspection["error"] = str(exc)
        return inspection

    for config_name in configs[:8]:
        config_info: dict[str, Any] = {"name": config_name}
        try:
            config_info["splits"] = get_dataset_split_names(dataset_id, config_name)
        except Exception as exc:  # pragma: no cover - network/schema dependent.
            config_info["split_error"] = str(exc)
        inspection["configs"].append(config_info)
    if len(configs) > 8:
        inspection["truncated_config_count"] = len(configs) - 8
    return inspection


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DatasetGateError as exc:
        print(json.dumps({"blocked": True, "reason": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc
