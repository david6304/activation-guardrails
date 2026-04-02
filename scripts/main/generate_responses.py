"""Generate model responses for the refusal-probing dataset.

Runs Gemma 2 9B IT on the AdvBench + Alpaca prompt dataset and writes
responses to a JSONL checkpoint file.  Resumable: pass the same --output
path to pick up where a previous run left off.

The output is consumed by label_refusals.py (LLM-as-judge) and then by
build_refusal_dataset.py to construct the final probing dataset.

Usage:
    python scripts/main/generate_responses.py \\
        --config configs/main/refusal.yaml

    # Resume a failed run:
    python scripts/main/generate_responses.py \\
        --config configs/main/refusal.yaml \\
        --output artifacts/responses/refusal/responses.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agguardrails.data import read_prompt_dataset
from agguardrails.generation import generate_responses
from agguardrails.io import save_metadata
from agguardrails.models import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate model responses for the refusal-probing dataset."
    )
    parser.add_argument(
        "--config",
        default="configs/main/refusal.yaml",
        help="Path to refusal experiment config.",
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/refusal_prompts.jsonl",
        help="Prompt JSONL produced by build_dataset.py with --config refusal.yaml.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/responses/refusal/responses.jsonl",
        help="Output JSONL for generated responses (checkpointed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    gen_cfg = config["generation"]

    print(f"Loading model: {model_cfg['name']}", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        model_cfg["name"],
        torch_dtype=model_cfg["dtype"],
    )
    print("Model loaded.", flush=True)

    examples = read_prompt_dataset(args.dataset)
    print(f"Dataset: {len(examples)} prompts from {args.dataset}", flush=True)

    output_path = Path(args.output)
    results = generate_responses(
        model,
        tokenizer,
        examples,
        max_new_tokens=int(gen_cfg["max_new_tokens"]),
        batch_size=int(gen_cfg["batch_size"]),
        temperature=float(gen_cfg["temperature"]),
        checkpoint_path=output_path,
        checkpoint_every=int(gen_cfg.get("checkpoint_every", 100)),
    )

    save_metadata(
        output_path.with_suffix(".metadata.json"),
        config_path=args.config,
        dataset_path=args.dataset,
        output_path=str(output_path),
        model_name=model_cfg["name"],
        n_responses=len(results),
        max_new_tokens=gen_cfg["max_new_tokens"],
        temperature=gen_cfg["temperature"],
    )

    print(f"\nResponses written to: {output_path}")
    print(f"Total: {len(results)}")


if __name__ == "__main__":
    main()
