#!/usr/bin/env python
"""Build prompt-only manifests for controlled positive completion generation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from agguardrails.ccpp_data import (
    GenerationPrompt,
    normalize_clearharm_prompt_row,
    normalize_harmbench_prompt_row,
    read_jsonl,
    write_generation_prompts,
    write_metadata,
)


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    allowed_domains = set(config["dataset"]["generation_prompts"]["allowed_domains"])
    prompt_config = config["dataset"]["generation_prompts"]
    if args.input_jsonl is not None:
        prompts = prompts_from_jsonl(
            args.input_jsonl,
            dataset_id=args.dataset_id,
            source_subset=args.source_subset,
            source_split=args.source_split,
        )
    else:
        prompts = prompts_from_hf(
            dataset_id=args.dataset_id,
            source_subset=args.source_subset,
            source_split=args.source_split,
            limit=args.limit,
        )
    domain_counts_before = Counter(prompt.domain for prompt in prompts)
    prompts = [prompt for prompt in prompts if prompt.domain in allowed_domains]
    if not prompts:
        raise SystemExit(
            "blocked: no prompts remain after CBRN/domain filtering; inspect "
            "allowed_domains and source categories"
        )

    write_generation_prompts(args.output, prompts)
    unique_group_count = len({prompt.group_id for prompt in prompts})
    min_total = int(prompt_config.get("min_total_prompts_for_report", 0))
    min_unique = int(prompt_config.get("min_unique_groups_for_report", 0))
    total_gate = "passed" if len(prompts) >= min_total else "smoke_only"
    unique_gate = "passed" if unique_group_count >= min_unique else "smoke_only"
    write_metadata(
        args.metadata_output,
        {
            "config_path": str(args.config),
            "dataset_id": args.dataset_id,
            "source_subset": args.source_subset,
            "source_split": args.source_split,
            "num_prompts": len(prompts),
            "unique_group_count": unique_group_count,
            "min_total_prompts_for_report": min_total,
            "min_unique_groups_for_report": min_unique,
            "total_row_size_gate": total_gate,
            "unique_group_size_gate": unique_gate,
            "reportable_size_gate": unique_gate,
            "allowed_domains": sorted(allowed_domains),
            "domain_counts_before_filter": dict(domain_counts_before),
            "domain_counts_after_filter": dict(
                Counter(prompt.domain for prompt in prompts)
            ),
            "status": "prompt_only_requires_completion_generation",
            "completion_source_required": "generated_uncensored",
            "normalized_dataset_target": config["dataset"]["normalized_path"],
        },
    )
    print(json.dumps({"num_prompts": len(prompts), "output": str(args.output)}))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ccpp/gemma3_4b_it_public_cbrn_probe.yaml"),
    )
    parser.add_argument("--dataset-id", default="walledai/HarmBench")
    parser.add_argument("--source-subset", default="standard")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metadata-output", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    prompt_config = config["dataset"]["generation_prompts"]
    args.output = args.output or Path(prompt_config["path"])
    args.metadata_output = args.metadata_output or Path(prompt_config["metadata_path"])
    return args


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return loaded


def prompts_from_jsonl(
    path: Path,
    *,
    dataset_id: str,
    source_subset: str,
    source_split: str,
) -> list[GenerationPrompt]:
    prompts = []
    for row_index, row in enumerate(read_jsonl(path)):
        prompt = normalize_prompt_row(
            row,
            dataset_id=dataset_id,
            row_index=row_index,
            source_subset=source_subset,
            source_split=source_split,
        )
        if prompt is not None:
            prompts.append(prompt)
    return prompts


def normalize_prompt_row(
    row: dict[str, Any],
    *,
    dataset_id: str,
    row_index: int,
    source_subset: str,
    source_split: str,
) -> GenerationPrompt | None:
    if dataset_id == "AlignmentResearch/ClearHarm":
        return normalize_clearharm_prompt_row(
            row,
            row_index=row_index,
            source_subset=source_subset,
            source_split=source_split,
        )
    if dataset_id == "walledai/HarmBench":
        return normalize_harmbench_prompt_row(
            row,
            row_index=row_index,
            source_subset=source_subset,
            source_split=source_split,
        )
    raise ValueError(f"unsupported prompt dataset: {dataset_id}")


def prompts_from_hf(
    *,
    dataset_id: str,
    source_subset: str,
    source_split: str,
    limit: int | None,
) -> list[GenerationPrompt]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent.
        msg = "datasets is required to load HarmBench prompt manifests"
        raise RuntimeError(msg) from exc

    split = source_split if limit is None else f"{source_split}[:{limit}]"
    dataset = load_dataset(dataset_id, source_subset, split=split)
    prompts = []
    for row_index, row in enumerate(dataset):
        prompt = normalize_prompt_row(
            row,
            dataset_id=dataset_id,
            row_index=row_index,
            source_subset=source_subset,
            source_split=source_split,
        )
        if prompt is not None:
            prompts.append(prompt)
    return prompts


if __name__ == "__main__":
    raise SystemExit(main())
