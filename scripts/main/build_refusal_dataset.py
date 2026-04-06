"""Build the refusal-probing prompt dataset.

Two modes:

1. **Source dataset** (before response generation):
   Builds the initial PromptExample JSONL from either AdvBench + Alpaca or
   WildJailbreak vanilla harmful + benign prompts, using source labels
   (harmful=1, benign=0). In WildJailbreak mode it also writes a separate
   adversarial_harmful-only evaluation set. This is the input to
   generate_responses.py.

2. **Refusal dataset** (after judge labelling):
   Overwrites prompt labels with judge-assigned refusal labels
   (refusal=1, compliance=0) and preserves the same splits.  This is
   the input to extract_activations.py and train_text_baseline.py.

Usage:
    # Step 1: build source prompt dataset
    python scripts/main/build_refusal_dataset.py \\
        --config configs/main/refusal.yaml \\
        --mode source \\
        --output data/processed/refusal_prompts.jsonl

    # Step 2: apply refusal labels from judge
    python scripts/main/build_refusal_dataset.py \\
        --config configs/main/refusal.yaml \\
        --mode relabel \\
        --labelled-responses artifacts/responses/refusal/labelled_responses.jsonl \\
        --output data/processed/refusal_labelled.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from agguardrails.data import (
    PromptExample,
    build_advbench_alpaca_dataset,
    build_wildjailbreak_refusal_adversarial_set,
    build_wildjailbreak_refusal_dataset,
    read_prompt_dataset,
    write_prompt_dataset,
)
from agguardrails.io import read_jsonl, save_metadata
from agguardrails.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or relabel the refusal-probing prompt dataset."
    )
    parser.add_argument("--config", default="configs/main/refusal.yaml")
    parser.add_argument(
        "--mode",
        choices=["source", "relabel"],
        default="source",
        help=(
            "source: build initial dataset from the configured refusal corpus. "
            "relabel: apply judge refusal labels to an existing dataset."
        ),
    )
    parser.add_argument(
        "--labelled-responses",
        default="artifacts/responses/refusal/labelled_responses.jsonl",
        help="Labelled response JSONL from label_refusals.py (used in relabel mode).",
    )
    parser.add_argument(
        "--source-dataset",
        default="data/processed/refusal_prompts.jsonl",
        help="Source prompt JSONL to relabel (used in relabel mode).",
    )
    parser.add_argument(
        "--output",
        default="data/processed/refusal_prompts.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--adversarial-output",
        default=None,
        help=(
            "Optional output JSONL for a WildJailbreak adversarial evaluation set. "
            "Ignored for AdvBench+Alpaca configs. If omitted in WildJailbreak "
            "source mode, a sibling *_adversarial.jsonl file is written."
        ),
    )
    return parser.parse_args()


def _default_adversarial_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_adversarial{output_path.suffix}")


def _build_source(
    config: dict,
    output_path: Path,
    *,
    adversarial_output_path: Path | None = None,
) -> tuple[list[PromptExample], list[PromptExample] | None]:
    seed = int(config["seed"])
    data_cfg = config["data"]
    dataset_name = str(data_cfg.get("dataset", "advbench_alpaca"))

    seed_everything(seed)
    adversarial_dataset: list[PromptExample] | None = None

    if dataset_name == "advbench_alpaca":
        dataset = build_advbench_alpaca_dataset(
            advbench_path=data_cfg["advbench_path"],
            alpaca_path=data_cfg["alpaca_path"],
            n_harmful=int(data_cfg["n_harmful"]),
            n_benign=int(data_cfg["n_benign"]),
            train_size=int(data_cfg["train_size"]),
            val_size=int(data_cfg["val_size"]),
            test_size=int(data_cfg["test_size"]),
            seed=seed,
        )
    elif dataset_name == "wildjailbreak":
        splits_cfg = data_cfg["splits"]
        dataset = build_wildjailbreak_refusal_dataset(
            wildjailbreak_path=data_cfg["wildjailbreak_path"],
            n_vanilla_harmful=int(data_cfg["n_vanilla_harmful"]),
            n_vanilla_benign=int(data_cfg["n_vanilla_benign"]),
            train_size=float(splits_cfg["train"]),
            val_size=float(splits_cfg["val"]),
            test_size=float(splits_cfg["test"]),
            seed=seed,
        )
        adversarial_dataset = build_wildjailbreak_refusal_adversarial_set(
            wildjailbreak_path=data_cfg["wildjailbreak_path"],
            n_adversarial=int(data_cfg["n_adversarial"]),
        )
        write_prompt_dataset(
            adversarial_output_path or _default_adversarial_output(output_path),
            adversarial_dataset,
        )
    else:
        msg = f"Unsupported refusal dataset: {dataset_name!r}"
        raise ValueError(msg)

    write_prompt_dataset(output_path, dataset)
    return dataset, adversarial_dataset


def _relabel(
    source_dataset_path: Path,
    labelled_responses_path: Path,
    output_path: Path,
) -> list[PromptExample]:
    """Replace source labels with judge-assigned refusal labels."""
    source_dataset = read_prompt_dataset(source_dataset_path)
    labelled_records = read_jsonl(labelled_responses_path)

    # Build lookup: example_id → refusal_label.
    refusal_lookup: dict[str, int] = {
        rec["example_id"]: int(rec["refusal_label"]) for rec in labelled_records
    }

    missing = [
        e.example_id for e in source_dataset if e.example_id not in refusal_lookup
    ]
    if missing:
        msg = (
            f"{len(missing)} examples in source dataset have no refusal label. "
            "Run label_refusals.py first."
        )
        raise ValueError(msg)

    relabelled = [
        PromptExample(
            example_id=e.example_id,
            prompt=e.prompt,
            label=refusal_lookup[e.example_id],
            split=e.split,
            source=e.source,
            source_id=e.source_id,
            source_label=e.source_label,
        )
        for e in source_dataset
    ]
    write_prompt_dataset(output_path, relabelled)
    return relabelled


def main() -> None:
    args = parse_args()

    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    output_path = Path(args.output)

    if args.mode == "source":
        dataset, adversarial_dataset = _build_source(
            config,
            output_path,
            adversarial_output_path=(
                Path(args.adversarial_output)
                if args.adversarial_output is not None
                else None
            ),
        )
        label_counts = Counter(e.label for e in dataset)
        split_counts = Counter(e.split for e in dataset)
        split_label_counts = Counter((e.split, e.label) for e in dataset)
        dataset_name = str(config["data"].get("dataset", "advbench_alpaca"))

        save_metadata(
            output_path.with_suffix(".metadata.json"),
            config_path=args.config,
            output_path=str(output_path),
            mode="source",
            dataset=dataset_name,
            seed=config["seed"],
            n_examples=len(dataset),
            label_counts=dict(label_counts),
            split_counts=dict(split_counts),
            split_label_counts={
                f"{split}:{label}": count
                for (split, label), count in sorted(split_label_counts.items())
            },
        )

        print(f"Source dataset written to: {output_path}")
        print(f"  Total: {len(dataset)}")
        print(f"  Labels: {dict(label_counts)}")
        print(f"  Splits: {dict(split_counts)}")
        if adversarial_dataset is not None:
            adversarial_output_path = (
                Path(args.adversarial_output)
                if args.adversarial_output is not None
                else _default_adversarial_output(output_path)
            )
            save_metadata(
                adversarial_output_path.with_suffix(".metadata.json"),
                config_path=args.config,
                output_path=str(adversarial_output_path),
                mode="source",
                dataset=dataset_name,
                split="adversarial",
                seed=config["seed"],
                n_examples=len(adversarial_dataset),
                label_counts={1: len(adversarial_dataset)},
                source_label_counts={"harmful": len(adversarial_dataset)},
                note=(
                    "WildJailbreak adversarial refusal evaluation set. "
                    "All prompts are harmful and assigned split=test."
                ),
            )
            print(f"Adversarial source dataset written to: {adversarial_output_path}")
            print(f"  Total: {len(adversarial_dataset)}")
            print("  All label=1, source_label=harmful, split=test")

    else:  # relabel
        dataset = _relabel(
            Path(args.source_dataset),
            Path(args.labelled_responses),
            output_path,
        )
        label_counts = Counter(e.label for e in dataset)
        split_label_counts = Counter((e.split, e.label) for e in dataset)

        save_metadata(
            output_path.with_suffix(".metadata.json"),
            config_path=args.config,
            output_path=str(output_path),
            mode="relabel",
            source_dataset_path=args.source_dataset,
            labelled_responses_path=args.labelled_responses,
            n_examples=len(dataset),
            refusal_label_counts=dict(label_counts),
            split_label_counts={
                f"{split}:{label}": count
                for (split, label), count in sorted(split_label_counts.items())
            },
        )

        print(f"Relabelled dataset written to: {output_path}")
        print(f"  Total: {len(dataset)}")
        print(f"  Refusal label counts (1=refused, 0=complied): {dict(label_counts)}")
        for (split, label), count in sorted(split_label_counts.items()):
            print(f"    {split:5s} refusal={label}: {count}")


if __name__ == "__main__":
    main()
