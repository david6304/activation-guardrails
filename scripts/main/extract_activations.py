"""Extract cached end-of-prompt hidden states for the main experiment dataset.

Extracts activations for vanilla train/val/test splits by default. Pass
--adversarial-dataset to also extract the adversarial held-out test set in the
same GPU job (avoids a second cluster submission for Phase D).

Usage:
    # Vanilla splits only
    python scripts/main/extract_activations.py --config configs/main/main.yaml

    # Vanilla + adversarial in one job
    python scripts/main/extract_activations.py \\
        --config configs/main/main.yaml \\
        --adversarial-dataset data/processed/main_adversarial.jsonl

    # Single split (e.g. re-run test only)
    python scripts/main/extract_activations.py --splits test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agguardrails.data import read_prompt_dataset, split_prompt_dataset
from agguardrails.features import (
    extract_last_token_hidden_states,
    save_activation_dataset,
    validate_layer_indices,
)
from agguardrails.models import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract hidden-state activations for the main experiment."
    )
    parser.add_argument("--config", default="configs/main/main.yaml")
    parser.add_argument(
        "--dataset",
        default="data/processed/main_prompts.jsonl",
        help="Vanilla train/val/test JSONL produced by build_dataset.py.",
    )
    parser.add_argument(
        "--adversarial-dataset",
        default=None,
        metavar="PATH",
        help=(
            "Adversarial held-out test JSONL (data/processed/main_adversarial.jsonl). "
            "If provided, activations are extracted and saved under split='adversarial'."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/activations/main",
        help="Directory for cached activation .npz files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        metavar="SPLIT",
        help="Vanilla splits to extract (default: train val test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    feature_cfg = config["features"]
    layers = [int(layer) for layer in feature_cfg["layers"]]

    print(f"Loading model: {model_cfg['name']}", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        model_cfg["name"],
        torch_dtype=model_cfg["dtype"],
    )
    validate_layer_indices(layers, model)
    print("Model loaded.", flush=True)

    # --- Vanilla splits ---
    dataset = read_prompt_dataset(args.dataset)
    splits = split_prompt_dataset(dataset)

    for split_name in args.splits:
        examples = splits[split_name]
        print(f"Extracting split: {split_name} ({len(examples)} examples)", flush=True)
        activation_dataset = extract_last_token_hidden_states(
            model=model,
            tokenizer=tokenizer,
            examples=examples,
            layers=layers,
            batch_size=int(feature_cfg["batch_size"]),
            max_length=int(feature_cfg["max_length"]),
        )
        save_activation_dataset(
            activation_dataset,
            output_dir=args.output_dir,
            split=split_name,
            config_path=args.config,
            model_name=model_cfg["name"],
        )
        print(f"Saved {split_name} activations: {len(activation_dataset.example_ids)} examples")

    # --- Adversarial held-out test set (optional) ---
    if args.adversarial_dataset is not None:
        adv_examples = read_prompt_dataset(args.adversarial_dataset)
        print(
            f"Extracting split: adversarial ({len(adv_examples)} examples)",
            flush=True,
        )
        activation_dataset = extract_last_token_hidden_states(
            model=model,
            tokenizer=tokenizer,
            examples=adv_examples,
            layers=layers,
            batch_size=int(feature_cfg["batch_size"]),
            max_length=int(feature_cfg["max_length"]),
        )
        save_activation_dataset(
            activation_dataset,
            output_dir=args.output_dir,
            split="adversarial",
            config_path=args.config,
            model_name=model_cfg["name"],
        )
        print(
            f"Saved adversarial activations: {len(activation_dataset.example_ids)} examples"
        )


if __name__ == "__main__":
    main()
