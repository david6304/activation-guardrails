#!/usr/bin/env python
"""Extract teacher-forced streaming activations for the CC++ SWiM probe."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from agguardrails.activations import (
    ActivationExample,
    build_activation_example,
    format_exchange_messages,
    save_activation_cache,
)
from agguardrails.ccpp_data import load_normalized_jsonl, validate_dataset_gates


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    examples = load_normalized_jsonl(args.dataset_jsonl)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not args.skip_gates:
        validate_dataset_gates(examples)

    if args.mock:
        activations = [
            mock_activation_example(example, config=config, index=index)
            for index, example in enumerate(examples)
        ]
    else:
        activations = extract_with_transformers(
            examples,
            config=config,
            device=args.device,
        )

    save_activation_cache(
        activations,
        npz_path=args.output_npz,
        index_path=args.output_index,
        metadata=activation_metadata(args=args, config=config, count=len(activations)),
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ccpp/gemma3_4b_it_public_cbrn_probe.yaml"),
    )
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        default=None,
        help="Normalized dataset JSONL. Defaults to dataset.normalized_path.",
    )
    parser.add_argument(
        "--output-npz",
        type=Path,
        default=None,
        help="Activation NPZ. Defaults inside activation.artifact_dir.",
    )
    parser.add_argument(
        "--output-index",
        type=Path,
        default=None,
        help="Activation index JSONL. Defaults inside activation.artifact_dir.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Write deterministic synthetic activations for artifact smoke tests.",
    )
    parser.add_argument(
        "--skip-gates",
        action="store_true",
        help="Skip dataset gates for narrow debugging only.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_config = config["dataset"]
    activation_config = config["activation"]
    artifact_dir = Path(activation_config["artifact_dir"])
    args.dataset_jsonl = args.dataset_jsonl or Path(dataset_config["normalized_path"])
    args.output_npz = args.output_npz or artifact_dir / "activations.npz"
    args.output_index = args.output_index or artifact_dir / "activations.index.jsonl"
    return args


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return loaded


def mock_activation_example(
    example,
    *,
    config: dict[str, Any],
    index: int,
) -> ActivationExample:
    activation_config = config["activation"]
    token_count = max(2, min(12, len(example.user_text.split()) + 2))
    hidden_dim = 4
    hidden_state_count = 4
    seed = int(hashlib.sha256(example.example_id.encode("utf-8")).hexdigest()[:8], 16)
    generator = torch.Generator().manual_seed(seed + index)
    hidden_states = [
        torch.randn(1, token_count, hidden_dim, generator=generator) + layer_index
        for layer_index in range(hidden_state_count)
    ]
    token_ids = torch.arange(token_count, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.ones(1, token_count, dtype=torch.long)
    return build_activation_example(
        example,
        hidden_states=hidden_states,
        token_ids=token_ids,
        attention_mask=attention_mask,
        layers=activation_config["layers"],
        activation_source=f"mock_{activation_config['source']}",
    )


def extract_with_transformers(examples, *, config: dict[str, Any], device: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent.
        msg = "transformers is required for real activation extraction"
        raise RuntimeError(msg) from exc

    model_config = config["model"]
    tokenizer_config = config["tokenizer"]
    activation_config = config["activation"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["id"],
        revision=tokenizer_config.get("revision"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config.get("revision"),
        torch_dtype=_torch_dtype(activation_config.get("dtype", "float32")),
    ).to(device)
    model.eval()

    activations = []
    for example in examples:
        text = format_exchange_messages(example.exchange_messages, tokenizer=tokenizer)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=int(activation_config["max_length"]),
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        activations.append(
            build_activation_example(
                example,
                hidden_states=outputs.hidden_states,
                token_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                layers=activation_config["layers"],
                activation_source=activation_config["source"],
            )
        )
    return activations


def activation_metadata(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    count: int,
) -> dict[str, Any]:
    return {
        "config_path": str(args.config),
        "git_commit": current_git_commit(),
        "mock": bool(args.mock),
        "num_examples": count,
        "model": config["model"],
        "tokenizer": config["tokenizer"],
        "activation": config["activation"],
    }


def current_git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported activation dtype: {name!r}")


if __name__ == "__main__":
    raise SystemExit(main())
