"""Activation feature extraction for the MVP linear-probe experiments."""

from __future__ import annotations

import sys
import warnings
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
    label_arrays: dict[str, np.ndarray] | None = None


TokenPosition = Literal["last", "last_instruction"]

_SOURCE_LABEL_TO_BINARY = {
    "harmful": 1,
    "unsafe": 1,
    "vanilla_harmful": 1,
    "adversarial_harmful": 1,
    "benign": 0,
    "safe": 0,
    "vanilla_benign": 0,
}


def _label_artifact_path(
    input_dir: str | Path,
    *,
    split: str,
    label_key: str,
) -> Path:
    input_dir = Path(input_dir)
    if label_key == "label":
        return input_dir / f"{split}_labels.npz"
    return input_dir / f"{split}_labels_{label_key}.npz"


def _load_label_arrays(input_dir: str | Path, *, split: str) -> dict[str, np.ndarray]:
    input_dir = Path(input_dir)
    label_arrays = {
        "label": load_artifact(
            _label_artifact_path(input_dir, split=split, label_key="label")
        )
    }
    prefix = f"{split}_labels_"
    for path in sorted(input_dir.glob(f"{prefix}*.npz")):
        key = path.stem.removeprefix(prefix)
        label_arrays[key] = load_artifact(path)
    return label_arrays


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
    """Locate the final token of the raw user instruction inside the chat template.

    First tries a direct subsequence match (works when standalone encoding
    matches the in-template encoding).  Falls back to character-offset mapping
    for tokenizers like SentencePiece where boundary context changes token IDs.
    """
    if not prompt:
        msg = "Cannot resolve last_instruction for an empty prompt."
        raise ValueError(msg)

    full_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=False,
    )

    # Fast path: direct subsequence match.
    content_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if content_ids:
        start = _find_subsequence_start(full_ids, content_ids)
        if start is not None:
            return start + len(content_ids) - 1

    # Slow path: diff the template-as-string with vs without content to find
    # the content character span, then use offset mapping to get the token.
    template_str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=False,
    )
    empty_template = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        tokenize=False,
        add_generation_prompt=False,
    )

    # Find common prefix and suffix to isolate the content span.
    prefix_len = 0
    for a, b in zip(template_str, empty_template):
        if a != b:
            break
        prefix_len += 1
    suffix_len = 0
    for a, b in zip(reversed(template_str), reversed(empty_template)):
        if a != b:
            break
        suffix_len += 1

    content_end = len(template_str) - suffix_len
    if content_end <= prefix_len:
        msg = "Could not locate prompt tokens inside the formatted user turn."
        raise ValueError(msg)
    # Last character of the content in the template string.
    last_char = content_end - 1

    encoding = tokenizer(template_str, add_special_tokens=False,
                         return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    # Find the token whose span contains last_char.
    for tok_idx, (start_c, end_c) in enumerate(offsets):
        if start_c <= last_char < end_c:
            return tok_idx

    msg = "Could not map prompt character offset to a token position."
    raise ValueError(msg)


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
    source_labels: list[int] = []
    has_binary_source_labels = True
    example_ids: list[str] = []
    n_skipped = 0

    n_batches = (len(examples) + batch_size - 1) // batch_size
    for batch_examples in tqdm(
        batched(examples, batch_size),
        total=n_batches,
        desc="extracting activations",
        file=sys.stdout,
        disable=False,
    ):
        prompts = [
            format_prompt(tokenizer, example.prompt) for example in batch_examples
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
            keep_mask = list(range(len(batch_examples)))
            kept_examples = list(batch_examples)
        else:
            resolved_positions = []
            keep_mask = []
            kept_examples = []
            for idx, (example, sequence_length, pad_offset) in enumerate(
                zip(
                    batch_examples,
                    sequence_lengths.tolist(),
                    pad_offsets.tolist(),
                    strict=True,
                )
            ):
                position = _resolve_last_instruction_position(tokenizer, example.prompt)
                if position >= int(sequence_length):
                    n_skipped += 1
                    warnings.warn(
                        f"Skipping {example.example_id}: truncated before "
                        f"token_position={token_position!r} "
                        f"(position {position} >= max_length {max_length})",
                        stacklevel=2,
                    )
                    continue
                resolved_positions.append(int(pad_offset) + position)
                keep_mask.append(idx)
                kept_examples.append(example)
            if not resolved_positions:
                continue
            positions = torch.tensor(resolved_positions, device=model.device)

        batch_indices = torch.tensor(keep_mask, device=model.device)
        for layer in layers:
            hidden = outputs.hidden_states[layer]
            batch_vectors = hidden[batch_indices, positions]
            # NumPy cannot reliably materialise some PyTorch dtypes such as
            # bfloat16 directly. Cast in torch first so model dtype choices
            # (for example Gemma in bfloat16) do not break extraction.
            all_features[layer].append(
                batch_vectors.detach().to(dtype=torch.float32).cpu().numpy()
            )

        labels.extend(example.label for example in kept_examples)
        for example in kept_examples:
            if not has_binary_source_labels:
                break
            binary_source_label = _SOURCE_LABEL_TO_BINARY.get(example.source_label)
            if binary_source_label is None:
                has_binary_source_labels = False
                source_labels.clear()
                break
            source_labels.append(binary_source_label)
        example_ids.extend(example.example_id for example in kept_examples)

    if n_skipped:
        print(
            f"Skipped {n_skipped}/{len(examples)} examples "
            f"(truncated before token_position={token_position!r})",
            flush=True,
        )

    features_by_layer = {
        layer: np.concatenate(chunks, axis=0) for layer, chunks in all_features.items()
    }
    label_arrays = {"label": np.array(labels, dtype=np.int64)}
    if has_binary_source_labels and len(source_labels) == len(example_ids):
        label_arrays["source_label"] = np.array(source_labels, dtype=np.int64)
    return ActivationDataset(
        features_by_layer=features_by_layer,
        labels=label_arrays["label"],
        example_ids=example_ids,
        label_arrays=label_arrays,
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

    label_arrays = {"label": dataset.labels}
    if dataset.label_arrays is not None:
        label_arrays.update(dataset.label_arrays)

    label_paths = {
        key: str(
            save_artifact(
                values,
                _label_artifact_path(
                    output_dir, split=split, label_key=key
                ).with_suffix(""),
            )
        )
        for key, values in sorted(label_arrays.items())
    }
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
        label_keys=sorted(label_arrays.keys()),
        layers=sorted(dataset.features_by_layer.keys()),
        labels_path=label_paths["label"],
        label_paths=label_paths,
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
    label_key: str = "label",
) -> ActivationDataset:
    """Load cached dense activation features, labels, and ids for one split."""
    return load_layer_feature_split(
        input_dir=input_dir,
        split=split,
        layers=layers,
        feature_name="features",
        label_key=label_key,
    )


def load_layer_feature_split(
    *,
    input_dir: str | Path,
    split: str,
    layers: list[int],
    feature_name: str,
    label_key: str = "label",
) -> ActivationDataset:
    """Load cached per-layer features, labels, and ids for one split."""
    input_dir = Path(input_dir)
    features_by_layer = {
        layer: load_artifact(input_dir / f"{split}_layer_{layer}_{feature_name}.npz")
        for layer in layers
    }
    label_arrays = _load_label_arrays(input_dir, split=split)
    if label_key not in label_arrays:
        available = sorted(label_arrays)
        msg = (
            f"Label key {label_key!r} not available for split={split!r}. "
            f"Available keys: {available}"
        )
        raise ValueError(msg)
    example_ids = load_artifact(input_dir / f"{split}_ids.npz").tolist()
    return ActivationDataset(
        features_by_layer=features_by_layer,
        labels=label_arrays[label_key],
        example_ids=example_ids,
        label_arrays=label_arrays,
    )
