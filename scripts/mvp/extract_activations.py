"""Extract cached end-of-prompt hidden states for the frozen MVP dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agguardrails.data import read_prompt_dataset, split_prompt_dataset
from agguardrails.features import extract_last_token_hidden_states, save_activation_dataset
from agguardrails.models import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp/mvp.yaml")
    parser.add_argument("--dataset", default="data/processed/mvp_prompts.jsonl")
    parser.add_argument("--output-dir", default="artifacts/activations/mvp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    dataset = read_prompt_dataset(args.dataset)
    splits = split_prompt_dataset(dataset)
    model_cfg = config["model"]
    feature_cfg = config["features"]

    print(f"Loading model: {model_cfg['name']}", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        model_cfg["name"],
        torch_dtype=model_cfg["dtype"],
    )
    print("Model loaded.", flush=True)

    layers = [int(layer) for layer in feature_cfg["layers"]]
    for split_name in ("train", "val", "test"):
        print(f"Extracting split: {split_name}", flush=True)
        activation_dataset = extract_last_token_hidden_states(
            model=model,
            tokenizer=tokenizer,
            examples=splits[split_name],
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
        print(f"Saved {split_name} activations with {len(activation_dataset.example_ids)} examples")


if __name__ == "__main__":
    main()
