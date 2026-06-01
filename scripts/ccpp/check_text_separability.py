#!/usr/bin/env python
"""Run the TF-IDF text separability gate before activation extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from agguardrails.ccpp_data import load_normalized_jsonl
from agguardrails.text_diagnostics import (
    TextSeparabilityConfig,
    text_separability_report,
)


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    examples = load_normalized_jsonl(args.dataset_jsonl)
    diagnostic_config = config["dataset"]["text_separability_gate"]
    report = text_separability_report(
        examples,
        config=TextSeparabilityConfig(
            max_allowed_roc_auc=float(diagnostic_config["max_allowed_roc_auc"]),
            text_fields=tuple(diagnostic_config["text_fields"]),
        ),
    )
    write_json(args.output, report)
    if report["gate"] != "passed":
        raise SystemExit(
            "blocked: text baseline is near ceiling; inspect generator/topic/length "
            "confounds before extracting activations"
        )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ccpp/gemma3_4b_it_public_cbrn_probe.yaml"),
    )
    parser.add_argument("--dataset-jsonl", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    args.dataset_jsonl = args.dataset_jsonl or Path(
        config["dataset"]["normalized_path"]
    )
    args.output = args.output or Path(
        config["dataset"]["text_separability_gate"]["metrics_path"]
    )
    return args


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return loaded


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
