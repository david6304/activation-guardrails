"""Activation feature extraction for the MVP linear-probe experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


def extract_last_token_hidden_states(
    *,
    model: torch.nn.Module,
    tokenizer,
    examples: list[PromptExample],
    layers: list[int],
    batch_size: int,
    max_length: int,
) -> ActivationDataset:
    """Extract end-of-prompt hidden states for the requested layers."""
    all_features = {layer: [] for layer in layers}
    labels: list[int] = []
    example_ids: list[str] = []

    n_batches = (len(examples) + batch_size - 1) // batch_size
    for batch_examples in tqdm(batched(examples, batch_size), total=n_batches, desc="extracting activations"):
        prompts = [format_prompt(tokenizer, example.prompt) for example in batch_examples]
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
        last_positions = attention_mask.sum(dim=1) - 1

        for layer in layers:
            hidden = outputs.hidden_states[layer]
            batch_vectors = hidden[
                torch.arange(hidden.size(0), device=hidden.device),
                last_positions,
            ]
            all_features[layer].append(batch_vectors.detach().cpu().numpy().astype(np.float32))

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
) -> dict[int, Path]:
    """Persist activation features, labels, ids, and metadata for one split."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: dict[int, Path] = {}
    for layer, features in dataset.features_by_layer.items():
        saved_paths[layer] = save_artifact(
            features,
            output_dir / f"{split}_layer_{layer}_features",
        )

    labels_path = save_artifact(dataset.labels, output_dir / f"{split}_labels")
    ids_path = save_artifact(np.array(dataset.example_ids, dtype=str), output_dir / f"{split}_ids")
    save_metadata(
        output_dir / f"{split}_metadata.json",
        config_path=config_path,
        model_name=model_name,
        split=split,
        n_examples=len(dataset.example_ids),
        layers=sorted(dataset.features_by_layer.keys()),
        labels_path=str(labels_path),
        ids_path=str(ids_path),
        feature_paths={str(layer): str(path) for layer, path in sorted(saved_paths.items())},
    )
    return saved_paths


def load_activation_split(
    *,
    input_dir: str | Path,
    split: str,
    layers: list[int],
) -> ActivationDataset:
    """Load cached activation features, labels, and ids for one split."""
    input_dir = Path(input_dir)
    features_by_layer = {
        layer: load_artifact(input_dir / f"{split}_layer_{layer}_features.npz")
        for layer in layers
    }
    labels = load_artifact(input_dir / f"{split}_labels.npz")
    example_ids = load_artifact(input_dir / f"{split}_ids.npz").tolist()
    return ActivationDataset(
        features_by_layer=features_by_layer,
        labels=labels,
        example_ids=example_ids,
    )
