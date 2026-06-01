from __future__ import annotations

import numpy as np
import torch

from agguardrails.activations import (
    activation_index_rows,
    build_activation_example,
    concatenate_hidden_layers,
    format_exchange_messages,
    load_activation_cache,
    resolve_layer_indices,
    save_activation_cache,
)
from agguardrails.ccpp_data import normalize_record


def test_format_exchange_messages_uses_fallback_chat_format() -> None:
    text = format_exchange_messages(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    )

    assert text == "user: Hello\nassistant: Hi"


def test_resolve_layer_indices_excludes_embedding_for_all_layers() -> None:
    assert resolve_layer_indices(4, "all") == [1, 2, 3]
    assert resolve_layer_indices(4, "all", include_embedding=True) == [0, 1, 2, 3]
    assert resolve_layer_indices(4, [1, 3]) == [1, 3]


def test_concatenate_hidden_layers_masks_padding_tokens() -> None:
    hidden_states = [
        torch.full((1, 3, 2), float(layer_index))
        for layer_index in range(4)
    ]
    attention_mask = torch.tensor([[1, 1, 0]])

    features, layers = concatenate_hidden_layers(
        hidden_states,
        layers=[1, 3],
        attention_mask=attention_mask,
    )

    assert layers.tolist() == [1, 3]
    assert features.shape == (2, 4)
    torch.testing.assert_close(
        features,
        torch.tensor([[1.0, 1.0, 3.0, 3.0], [1.0, 1.0, 3.0, 3.0]]),
    )


def test_build_activation_example_and_cache_round_trip(tmp_path) -> None:
    exchange = normalize_record(
        {
            "example_id": "ex-1",
            "group_id": "group-1",
            "split": "train",
            "label": 1,
            "domain": "cbrn",
            "source_dataset": "fixture",
            "source_subset": "unit",
            "user_text": "Prompt",
            "assistant_text": "A non-refusal assistant completion.",
            "completion_source": "public",
        }
    )
    hidden_states = [
        torch.arange(6, dtype=torch.float32).reshape(1, 3, 2) + layer_index
        for layer_index in range(3)
    ]
    token_ids = torch.tensor([[10, 11, 0]])
    attention_mask = torch.tensor([[1, 1, 0]])
    activation = build_activation_example(
        exchange,
        hidden_states=hidden_states,
        token_ids=token_ids,
        attention_mask=attention_mask,
        layers="all",
        activation_source="hidden_residual",
    )

    npz_path = tmp_path / "activations.npz"
    index_path = tmp_path / "activations.index.jsonl"
    save_activation_cache(
        [activation],
        npz_path=npz_path,
        index_path=index_path,
        metadata={"model_id": "fixture-model"},
    )
    loaded, metadata = load_activation_cache(npz_path=npz_path, index_path=index_path)
    index_rows = activation_index_rows(index_path)

    assert metadata == {"model_id": "fixture-model"}
    assert index_rows[0]["example_id"] == "ex-1"
    assert index_rows[0]["num_tokens"] == 2
    assert loaded[0].layers == [1, 2]
    np.testing.assert_allclose(loaded[0].features, activation.features)
    np.testing.assert_array_equal(loaded[0].token_ids, np.asarray([10, 11]))

