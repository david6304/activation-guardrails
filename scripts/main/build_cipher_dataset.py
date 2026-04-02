"""Build a cipher-transformed evaluation dataset from an existing prompt set.

Usage:
    python scripts/main/build_cipher_dataset.py \
        --dataset data/processed/refusal_prompts.jsonl \
        --cipher rot13 \
        --output data/processed/refusal_ciphers/rot13_prompts.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from agguardrails.data import (
    build_cipher_dataset,
    read_prompt_dataset,
    write_prompt_dataset,
)
from agguardrails.io import save_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cipher-transformed prompt dataset from one split."
    )
    parser.add_argument(
        "--dataset",
        default="data/processed/refusal_prompts.jsonl",
        help="Canonical prompt dataset to transform.",
    )
    parser.add_argument(
        "--cipher",
        required=True,
        help="Cipher name from agguardrails.ciphers.CIPHER_REGISTRY.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split to transform (default: test).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSONL path for the cipher-transformed dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = read_prompt_dataset(args.dataset)
    cipher_dataset = build_cipher_dataset(
        dataset,
        cipher=args.cipher,
        split=args.split,
    )
    output_path = write_prompt_dataset(args.output, cipher_dataset)

    label_counts = Counter(example.label for example in cipher_dataset)
    source_counts = Counter(example.source_label for example in cipher_dataset)

    save_metadata(
        output_path.with_suffix(".metadata.json"),
        dataset_path=args.dataset,
        output_path=str(output_path),
        cipher=args.cipher,
        split=args.split,
        n_examples=len(cipher_dataset),
        label_counts=dict(label_counts),
        source_label_counts=dict(source_counts),
    )

    print(f"Cipher dataset written to: {output_path}")
    print(f"Cipher: {args.cipher}")
    print(f"Split: {args.split}")
    print(f"Examples: {len(cipher_dataset)}")


if __name__ == "__main__":
    main()
