"""Build the main experiment dataset from WildJailbreak.

Produces two JSONL files:
  1. Vanilla train/val/test dataset  — used for all model training and evaluation.
  2. Adversarial held-out test set   — used only for obfuscation robustness eval.

Usage:
    python scripts/main/build_dataset.py --config configs/main/main.yaml

Prerequisites:
    Run scripts/download_wildjailbreak.py first to produce the normalised JSONL.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from agguardrails.data import build_adversarial_test_set, build_main_dataset, write_prompt_dataset
from agguardrails.io import save_metadata
from agguardrails.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build main experiment dataset from WildJailbreak.")
    parser.add_argument(
        "--config",
        default="configs/main/main.yaml",
        help="Path to main experiment config.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/main_prompts.jsonl",
        help="Output JSONL for vanilla train/val/test dataset.",
    )
    parser.add_argument(
        "--adversarial-output",
        default="data/processed/main_adversarial.jsonl",
        help="Output JSONL for adversarial held-out test set.",
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

    # --- Vanilla train/val/test ---
    dataset = build_main_dataset(
        wildjailbreak_path=data_cfg["wildjailbreak_path"],
        n_vanilla_harmful=int(data_cfg["n_vanilla_harmful"]),
        n_vanilla_benign=int(data_cfg["n_vanilla_benign"]),
        train_size=float(splits_cfg["train"]),
        val_size=float(splits_cfg["val"]),
        test_size=float(splits_cfg["test"]),
        seed=seed,
    )

    output_path = write_prompt_dataset(args.output, dataset)

    label_counts = Counter(e.label for e in dataset)
    split_counts = Counter(e.split for e in dataset)
    split_label_counts = Counter((e.split, e.label) for e in dataset)

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

    print(f"Vanilla dataset: {output_path}")
    print(f"  Total examples : {len(dataset)}")
    print(f"  Label counts   : {dict(label_counts)}")
    print(f"  Split counts   : {dict(split_counts)}")
    for (split, label), count in sorted(split_label_counts.items()):
        print(f"    {split:5s} label={label}: {count}")

    # --- Adversarial held-out test set ---
    adversarial = build_adversarial_test_set(
        wildjailbreak_path=data_cfg["wildjailbreak_path"],
        n_adversarial=int(data_cfg["n_adversarial"]),
    )

    adv_path = write_prompt_dataset(args.adversarial_output, adversarial)

    save_metadata(
        adv_path.with_suffix(".metadata.json"),
        config_path=args.config,
        output_path=str(adv_path),
        seed=seed,
        n_examples=len(adversarial),
        label_counts={1: len(adversarial)},
        note="Adversarial held-out test set — all harmful, no benign counterpart.",
    )

    print(f"\nAdversarial test set: {adv_path}")
    print(f"  Total examples : {len(adversarial)}")
    print(f"  All label=1 (harmful)")


if __name__ == "__main__":
    main()
