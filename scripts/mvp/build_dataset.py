"""Build the frozen MVP dataset and stratified splits from local raw JSONL files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from agguardrails.data import build_mvp_dataset, write_prompt_dataset
from agguardrails.io import save_metadata
from agguardrails.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/mvp/mvp.yaml",
        help="Path to the frozen MVP config.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/mvp_prompts.jsonl",
        help="Output JSONL path for the canonical dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    seed = int(config["seed"])
    data_cfg = config["data"]
    splits_cfg = data_cfg["splits"]

    seed_everything(seed)
    dataset = build_mvp_dataset(
        harmful_path=data_cfg["harmful_source"],
        benign_path=data_cfg["benign_source"],
        benign_label=data_cfg["benign_label"],
        n_harmful=int(data_cfg["n_harmful"]),
        n_benign=int(data_cfg["n_benign"]),
        train_size=float(splits_cfg["train"]),
        val_size=float(splits_cfg["val"]),
        test_size=float(splits_cfg["test"]),
        seed=seed,
    )

    output_path = write_prompt_dataset(args.output, dataset)

    label_counts = Counter(example.label for example in dataset)
    split_counts = Counter(example.split for example in dataset)
    split_label_counts = Counter((example.split, example.label) for example in dataset)

    save_metadata(
        output_path.with_suffix(".metadata.json"),
        config_path=args.config,
        output_path=str(output_path),
        seed=seed,
        n_examples=len(dataset),
        label_counts=dict(label_counts),
        split_counts=dict(split_counts),
        split_label_counts={
            f"{split}:{label}": count
            for (split, label), count in sorted(split_label_counts.items())
        },
    )

    print(f"Wrote dataset to {output_path}")
    print(f"Examples: {len(dataset)}")
    print(f"Label counts: {dict(label_counts)}")
    print(f"Split counts: {dict(split_counts)}")


if __name__ == "__main__":
    main()
