"""Activation feature assembly and cache contracts for CC++ probes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from agguardrails.ccpp_data import NormalizedExchange


@dataclass(frozen=True)
class ActivationExample:
    example_id: str
    label: int
    split: str
    group_id: str
    features: np.ndarray
    token_ids: np.ndarray
    token_mask: np.ndarray
    layers: list[int]
    metadata: dict[str, Any]


def format_exchange_messages(
    messages: Sequence[Mapping[str, str]],
    *,
    tokenizer: Any | None = None,
) -> str:
    """Format chat messages for teacher-forced activation extraction."""

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                list(messages),
                tokenize=False,
                add_generation_prompt=False,
            )
        )

    chunks = []
    for message in messages:
        role = str(message["role"]).strip()
        content = str(message["content"]).strip()
        chunks.append(f"{role}: {content}")
    return "\n".join(chunks)


def resolve_layer_indices(
    hidden_state_count: int,
    layers: str | Sequence[int],
    *,
    include_embedding: bool = False,
) -> list[int]:
    """Resolve configured layer IDs into hidden_states tuple indices."""

    if hidden_state_count < 1:
        raise ValueError("hidden_state_count must be >= 1")
    if layers == "all":
        start = 0 if include_embedding else 1
        return list(range(start, hidden_state_count))
    if isinstance(layers, str):
        raise ValueError(f"unsupported layers setting: {layers!r}")

    resolved = [int(layer) for layer in layers]
    invalid = [layer for layer in resolved if layer < 0 or layer >= hidden_state_count]
    if invalid:
        raise ValueError(
            f"layer indices out of range for {hidden_state_count} hidden states: "
            f"{invalid}"
        )
    return resolved


def concatenate_hidden_layers(
    hidden_states: Sequence[Tensor],
    *,
    layers: str | Sequence[int],
    attention_mask: Tensor | None = None,
    include_embedding: bool = False,
) -> tuple[Tensor, Tensor]:
    """Concatenate selected hidden-state layers into per-token features."""

    if not hidden_states:
        raise ValueError("hidden_states must be non-empty")
    if hidden_states[0].ndim != 3 or hidden_states[0].shape[0] != 1:
        raise ValueError("hidden states must have shape [1, seq_len, hidden_dim]")

    resolved_layers = resolve_layer_indices(
        len(hidden_states),
        layers,
        include_embedding=include_embedding,
    )
    selected = [hidden_states[index].squeeze(0) for index in resolved_layers]
    features = torch.cat(selected, dim=-1)

    if attention_mask is None:
        token_mask = torch.ones(
            features.shape[0],
            dtype=torch.bool,
            device=features.device,
        )
    else:
        if attention_mask.ndim != 2 or attention_mask.shape[0] != 1:
            raise ValueError("attention_mask must have shape [1, seq_len]")
        token_mask = attention_mask.squeeze(0).to(dtype=torch.bool)

    return features[token_mask], torch.tensor(resolved_layers, device=features.device)


def build_activation_example(
    example: NormalizedExchange,
    *,
    hidden_states: Sequence[Tensor],
    token_ids: Tensor,
    attention_mask: Tensor,
    layers: str | Sequence[int],
    activation_source: str,
    include_embedding: bool = False,
) -> ActivationExample:
    features, resolved_layers = concatenate_hidden_layers(
        hidden_states,
        layers=layers,
        attention_mask=attention_mask,
        include_embedding=include_embedding,
    )
    token_ids_1d = token_ids.squeeze(0)[attention_mask.squeeze(0).to(dtype=torch.bool)]

    return ActivationExample(
        example_id=example.example_id,
        label=example.label,
        split=example.split,
        group_id=example.group_id,
        features=features.detach().cpu().to(dtype=torch.float32).numpy(),
        token_ids=token_ids_1d.detach().cpu().to(dtype=torch.int64).numpy(),
        token_mask=np.ones(features.shape[0], dtype=bool),
        layers=[int(layer) for layer in resolved_layers.detach().cpu().tolist()],
        metadata={
            "activation_source": activation_source,
            "source_dataset": example.source_dataset,
            "source_subset": example.source_subset,
            "completion_source": example.completion_source,
        },
    )


def save_activation_cache(
    examples: Sequence[ActivationExample],
    *,
    npz_path: Path,
    index_path: Path,
    metadata: Mapping[str, Any],
) -> None:
    """Write variable-length activation arrays plus a JSONL index."""

    if not examples:
        raise ValueError("examples must be non-empty")

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    index_rows = []
    for offset, example in enumerate(examples):
        key = f"example_{offset:06d}"
        arrays[f"{key}_features"] = example.features
        arrays[f"{key}_token_ids"] = example.token_ids
        arrays[f"{key}_token_mask"] = example.token_mask
        index_rows.append(
            {
                "array_key": key,
                "example_id": example.example_id,
                "label": example.label,
                "split": example.split,
                "group_id": example.group_id,
                "num_tokens": int(example.features.shape[0]),
                "feature_dim": int(example.features.shape[1]),
                "layers": example.layers,
                "metadata": example.metadata,
            }
        )
    arrays["metadata_json"] = np.asarray(json.dumps(dict(metadata), sort_keys=True))
    np.savez_compressed(npz_path, **arrays)

    with index_path.open("w", encoding="utf-8") as handle:
        for row in index_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_activation_cache(
    *,
    npz_path: Path,
    index_path: Path,
) -> tuple[list[ActivationExample], dict[str, Any]]:
    with np.load(npz_path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata_json"]))
        index_rows = [
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        examples = []
        for row in index_rows:
            key = row["array_key"]
            examples.append(
                ActivationExample(
                    example_id=row["example_id"],
                    label=int(row["label"]),
                    split=row["split"],
                    group_id=row["group_id"],
                    features=np.asarray(arrays[f"{key}_features"]),
                    token_ids=np.asarray(arrays[f"{key}_token_ids"]),
                    token_mask=np.asarray(arrays[f"{key}_token_mask"]),
                    layers=[int(layer) for layer in row["layers"]],
                    metadata=dict(row["metadata"]),
                )
            )
    return examples, metadata


def activation_index_rows(index_path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def activation_example_to_dict(example: ActivationExample) -> dict[str, Any]:
    values = asdict(example)
    values["features_shape"] = list(example.features.shape)
    values["token_ids_shape"] = list(example.token_ids.shape)
    values["token_mask_shape"] = list(example.token_mask.shape)
    values.pop("features")
    values.pop("token_ids")
    values.pop("token_mask")
    return values
