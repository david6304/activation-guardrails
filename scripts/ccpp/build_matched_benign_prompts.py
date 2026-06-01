#!/usr/bin/env python
"""Assemble the matched dual-use benign prompt manifest for the CC++ slice.

Builds topic-matched benign CBRN/science-adjacent prompts following
``docs/matched_benign_prompt_spec.md`` and writes a prompt-only JSONL manifest
plus a metadata summary under the ignored ``data/interim/ccpp/`` tree. Raw prompt
text is never printed; stdout carries only aggregate counts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from agguardrails.ccpp_benign import (
    assert_no_harmful_prompts,
    assert_unique_prompt_ids,
    benign_prompt_metadata,
    build_benign_prompts,
    find_near_duplicate_groups,
)
from agguardrails.ccpp_data import write_generation_prompts, write_metadata


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    targets = config["dataset"]["benign_prompt_targets"]
    target_unique_groups = int(args.target or targets["target_unique_groups"])
    threshold = float(targets.get("near_duplicate_jaccard_threshold", 0.9))
    minimum_unique = targets.get("minimum_unique_groups_for_tpr_at_1pct_fpr")

    prompts = build_benign_prompts(
        target_unique_groups=target_unique_groups,
        seed=args.seed,
        source_subset=str(targets.get("source_subset", "dual_use_benign_v1")),
        source_split=str(targets.get("source_split", "synthetic")),
    )

    assert_no_harmful_prompts(prompts)
    assert_unique_prompt_ids(prompts)
    near_duplicate_pairs = find_near_duplicate_groups(prompts, threshold=threshold)

    write_generation_prompts(args.output, prompts)
    metadata = benign_prompt_metadata(
        prompts,
        target_unique_groups=target_unique_groups,
        config_path=str(args.config),
        near_duplicate_pairs=near_duplicate_pairs,
        minimum_unique_groups=(
            int(minimum_unique) if minimum_unique is not None else None
        ),
    )
    write_metadata(args.metadata_output, metadata)

    print(
        json.dumps(
            {
                "total_prompt_count": metadata["total_prompt_count"],
                "unique_group_count": metadata["unique_group_count"],
                "diversity_status": metadata["diversity_status"],
                "output": str(args.output),
            }
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ccpp/gemma2_9b_it_public_cbrn_probe.yaml"),
    )
    parser.add_argument("--target", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    targets = config["dataset"]["benign_prompt_targets"]
    args.output = args.output or Path(targets["path"])
    args.metadata_output = args.metadata_output or Path(targets["metadata_path"])
    return args


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return loaded


if __name__ == "__main__":
    raise SystemExit(main())
