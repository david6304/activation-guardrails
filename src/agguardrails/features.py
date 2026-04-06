"""Activation feature extraction for the MVP linear-probe experiments."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from agguardrails.data import PromptExample
from agguardrails.io import load_artifact, save_artifact, save_metadata
from agguardrails.models import format_prompt
from agguardrails.utils import batched


@dataclass(frozen=True)
class ActivationDataset:
    """Cached activation features plus labels and example metadata."""

    features_by_layer: dict[int, np.ndarray]
    labels: np.ndarray
    example_ids: list[str]


TokenPosition = Literal["last", "last_instruction"]


def validate_layer_indices(layers: list[int], model: torch.nn.Module) -> None:
    """Raise ValueError if any requested layer index exceeds the model limit."""
    n_hidden = model.config.num_hidden_layers + 1  # +1 for embedding layer at index 0
    invalid = [layer for layer in layers if layer >= n_hidden]
    if invalid:
        msg = (
            f"Requested layers {invalid} out of range; model has "
            f"{n_hidden} hidden states (indices 0-{n_hidden - 1})"
        )
        raise ValueError(msg)


def validate_token_position(token_position: str) -> TokenPosition:
    """Validate the requested token position selector."""
    valid_positions = {"last", "last_instruction"}
    if token_position not in valid_positions:
        msg = (
            f"Unsupported token_position {token_position!r}; "
            f"expected one of {sorted(valid_positions)}"
        )
        raise ValueError(msg)
    return token_position


def _find_subsequence_start(sequence: list[int], subsequence: list[int]) -> int | None:
    """Return the last start index where ``subsequence`` occurs inside ``sequence``."""
    max_start = len(sequence) - len(subsequence)
    for start in range(max_start, -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return None


def _resolve_last_instruction_position(tokenizer, prompt: str) -> int:
    """Locate the final token of the raw user instruction inside the chat template."""
    content_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not content_ids:
        msg = "Cannot resolve last_instruction for an empty prompt."
        raise ValueError(msg)

    user_turn_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=False,
    )
    start = _find_subsequence_start(user_turn_ids, content_ids)
    if start is None:
        msg = "Could not locate prompt tokens inside the formatted user turn."
        raise ValueError(msg)
    return start + len(content_ids) - 1


def extract_last_token_hidden_states(
    *,
    model: torch.nn.Module,
    tokenizer,
    examples: list[PromptExample],
    layers: list[int],
    batch_size: int,
    max_length: int,
    token_position: TokenPosition = "last",
) -> ActivationDataset:
    """Extract hidden states at a configurable token position."""
    token_position = validate_token_position(token_position)
    all_features = {layer: [] for layer in layers}
    labels: list[int] = []
    example_ids: list[str] = []

    n_batches = (len(examples) + batch_size - 1) // batch_size
    for batch_examples in tqdm(
        batched(examples, batch_size),
        total=n_batches,
        desc="extracting activations",
        file=sys.stdout,
        disable=False,
    ):
        prompts = [
            format_prompt(tokenizer, example.prompt)
            for example in batch_examples
        ]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}

        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True)

        attention_mask = encoded["attention_mask"]
        sequence_lengths = attention_mask.sum(dim=1)
        if getattr(tokenizer, "padding_side", "right") == "left":
            pad_offsets = attention_mask.size(1) - sequence_lengths
        else:
            pad_offsets = torch.zeros_like(sequence_lengths)
        if token_position == "last":
            positions = pad_offsets + sequence_lengths - 1
        else:
            resolved_positions = []
            for example, sequence_length, pad_offset in zip(
                batch_examples,
                sequence_lengths.tolist(),
                pad_offsets.tolist(),
                strict=True,
            ):
                position = _resolve_last_instruction_position(tokenizer, example.prompt)
                if position >= int(sequence_length):
                    msg = (
                        f"Prompt {example.example_id} was truncated before "
                        "token_position="
                        f"{token_position!r} could be reached "
                        f"(max_length={max_length})."
                    )
                    raise ValueError(msg)
                resolved_positions.append(int(pad_offset) + position)
            positions = torch.tensor(resolved_positions, device=model.device)

        for layer in layers:
            hidden = outputs.hidden_states[layer]
            batch_vectors = hidden[
                torch.arange(hidden.size(0), device=hidden.device),
                positions,
            ]
            # NumPy cannot reliably materialise some PyTorch dtypes such as
            # bfloat16 directly. Cast in torch first so model dtype choices
            # (for example Gemma in bfloat16) do not break extraction.
            all_features[layer].append(
                batch_vectors.detach().to(dtype=torch.float32).cpu().numpy()
            )

        labels.extend(example.label for example in batch_examples)
        example_ids.extend(example.example_id for example in batch_examples)

    features_by_layer = {
        layer: np.concatenate(chunks, axis=0) for layer, chunks in all_features.items()
    }
    return ActivationDataset(
        features_by_layer=features_by_layer,
        labels=np.array(labels, dtype=np.int64),
        example_ids=example_ids,
    )


def save_activation_dataset(
    dataset: ActivationDataset,
    *,
    output_dir: str | Path,
    split: str,
    config_path: str,
    model_name: str,
    token_position: TokenPosition = "last",
) -> dict[int, Path]:
    """Persist activation features, labels, ids, and metadata for one split."""
    token_position = validate_token_position(token_position)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[int, Path] = {}
    for layer, features in dataset.features_by_layer.items():
        saved_paths[layer] = save_artifact(
            features,
            output_dir / f"{split}_layer_{layer}_features",
        )

    labels_path = save_artifact(dataset.labels, output_dir / f"{split}_labels")
    ids_path = save_artifact(
        np.array(dataset.example_ids, dtype=str),
        output_dir / f"{split}_ids",
    )
    save_metadata(
        output_dir / f"{split}_metadata.json",
        config_path=config_path,
        model_name=model_name,
        split=split,
        token_position=token_position,
        n_examples=len(dataset.example_ids),
        layers=sorted(dataset.features_by_layer.keys()),
        labels_path=str(labels_path),
        ids_path=str(ids_path),
        feature_paths={
            str(layer): str(path) for layer, path in sorted(saved_paths.items())
        },
    )
    return saved_paths


def load_activation_split(
    *,
    input_dir: str | Path,
    split: str,
    layers: list[int],
) -> ActivationDataset:
    """Load cached dense activation features, labels, and ids for one split."""
    return load_layer_feature_split(
        input_dir=input_dir,
        split=split,
        layers=layers,
        feature_name="features",
    )


def load_layer_feature_split(
    *,
    input_dir: str | Path,
    split: str,
    layers: list[int],
    feature_name: str,
) -> ActivationDataset:
    """Load cached per-layer features, labels, and ids for one split."""
    input_dir = Path(input_dir)
    features_by_layer = {
        layer: load_artifact(input_dir / f"{split}_layer_{layer}_{feature_name}.npz")
        for layer in layers
    }
    labels = load_artifact(input_dir / f"{split}_labels.npz")
    example_ids = load_artifact(input_dir / f"{split}_ids.npz").tolist()
    return ActivationDataset(
        features_by_layer=features_by_layer,
        labels=labels,
        example_ids=example_ids,
    )
