"""Tests for agguardrails.features."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from agguardrails.data import PromptExample
from agguardrails.features import (
    ActivationDataset,
    extract_last_token_hidden_states,
    load_activation_split,
    load_layer_feature_split,
    save_activation_dataset,
    validate_layer_indices,
)


def _mock_model(num_hidden_layers: int) -> MagicMock:
    model = MagicMock()
    model.config.num_hidden_layers = num_hidden_layers
    return model


def test_validate_layer_indices_valid():
    # 28 transformer layers → hidden state indices 0–28
    validate_layer_indices([0, 8, 16, 24, 28], _mock_model(28))


def test_validate_layer_indices_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        validate_layer_indices([8, 16, 24, 32], _mock_model(28))


def test_save_and_load_activation_split_round_trip(tmp_path):
    dataset = ActivationDataset(
        features_by_layer={
            8: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            16: np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        },
        labels=np.array([0, 1], dtype=np.int64),
        example_ids=["a", "b"],
        label_arrays={
            "label": np.array([0, 1], dtype=np.int64),
            "source_label": np.array([1, 0], dtype=np.int64),
        },
    )

    save_activation_dataset(
        dataset,
        output_dir=tmp_path,
        split="train",
        config_path="configs/mvp/mvp.yaml",
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )
    loaded = load_activation_split(input_dir=tmp_path, split="train", layers=[8, 16])

    np.testing.assert_array_equal(
        loaded.features_by_layer[8], dataset.features_by_layer[8]
    )
    np.testing.assert_array_equal(
        loaded.features_by_layer[16], dataset.features_by_layer[16]
    )
    np.testing.assert_array_equal(loaded.labels, dataset.labels)
    np.testing.assert_array_equal(
        loaded.label_arrays["source_label"],
        dataset.label_arrays["source_label"],
    )
    assert loaded.example_ids == dataset.example_ids

    loaded_source = load_activation_split(
        input_dir=tmp_path,
        split="train",
        layers=[8, 16],
        label_key="source_label",
    )
    np.testing.assert_array_equal(
        loaded_source.labels,
        dataset.label_arrays["source_label"],
    )


def test_load_layer_feature_split_with_custom_feature_name(tmp_path):
    np.savez_compressed(
        tmp_path / "train_layer_9_sae_features.npz",
        data=np.array([[1.0, 2.0]], dtype=np.float32),
    )
    np.savez_compressed(
        tmp_path / "train_labels.npz",
        data=np.array([1], dtype=np.int64),
    )
    np.savez_compressed(
        tmp_path / "train_labels_source_label.npz",
        data=np.array([0], dtype=np.int64),
    )
    np.savez_compressed(
        tmp_path / "train_ids.npz",
        data=np.array(["ex-1"]),
    )

    loaded = load_layer_feature_split(
        input_dir=tmp_path,
        split="train",
        layers=[9],
        feature_name="sae_features",
    )

    np.testing.assert_array_equal(
        loaded.features_by_layer[9],
        np.array([[1.0, 2.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(loaded.labels, np.array([1], dtype=np.int64))
    np.testing.assert_array_equal(
        loaded.label_arrays["source_label"],
        np.array([0], dtype=np.int64),
    )
    assert loaded.example_ids == ["ex-1"]


def test_extract_last_token_hidden_states_casts_bfloat16_to_float32(monkeypatch):
    class FakeTokenizer:
        def __call__(self, prompts, return_tensors, padding, truncation, max_length):
            assert prompts == ["formatted::alpha", "formatted::beta"]
            return {
                "input_ids": torch.tensor(
                    [[10, 11, 12], [20, 21, 22]], dtype=torch.int64
                ),
                "attention_mask": torch.tensor(
                    [[1, 1, 1], [1, 1, 1]], dtype=torch.int64
                ),
            }

    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            batch = kwargs["input_ids"].shape[0]
            seq_len = kwargs["input_ids"].shape[1]
            hidden_states = []
            for layer_idx in range(3):
                values = (
                    torch.arange(
                        batch * seq_len * 2,
                        dtype=torch.float32,
                    ).reshape(batch, seq_len, 2)
                    + layer_idx
                )
                hidden_states.append(values.to(torch.bfloat16))
            return SimpleNamespace(hidden_states=tuple(hidden_states))

    examples = [
        PromptExample(
            "ex-1", "alpha", 1, "train", "wildjailbreak", "1", "vanilla_harmful"
        ),
        PromptExample(
            "ex-2", "beta", 0, "train", "wildjailbreak", "2", "vanilla_benign"
        ),
    ]

    monkeypatch.setattr(
        "agguardrails.features.format_prompt",
        lambda tokenizer, prompt: f"formatted::{prompt}",
    )

    dataset = extract_last_token_hidden_states(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        examples=examples,
        layers=[1, 2],
        batch_size=2,
        max_length=16,
    )

    assert dataset.features_by_layer[1].dtype == np.float32
    assert dataset.features_by_layer[2].dtype == np.float32
    assert dataset.features_by_layer[1].shape == (2, 2)
    np.testing.assert_array_equal(dataset.labels, np.array([1, 0], dtype=np.int64))
    assert dataset.example_ids == ["ex-1", "ex-2"]


def test_extract_last_token_hidden_states_handles_left_padding(monkeypatch):
    class FakeTokenizer:
        padding_side = "left"

        def __call__(self, prompts, return_tensors, padding, truncation, max_length):
            assert prompts == ["formatted::alpha", "formatted::beta"]
            return {
                "input_ids": torch.tensor(
                    [[10, 11, 12, 13, 14], [0, 20, 21, 22, 23]],
                    dtype=torch.int64,
                ),
                "attention_mask": torch.tensor(
                    [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]],
                    dtype=torch.int64,
                ),
            }

    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            batch = kwargs["input_ids"].shape[0]
            seq_len = kwargs["input_ids"].shape[1]
            hidden = torch.arange(batch * seq_len * 2, dtype=torch.float32).reshape(
                batch, seq_len, 2
            )
            return SimpleNamespace(hidden_states=(hidden,))

    examples = [
        PromptExample(
            "ex-1", "alpha", 1, "train", "wildjailbreak", "1", "vanilla_harmful"
        ),
        PromptExample(
            "ex-2", "beta", 0, "train", "wildjailbreak", "2", "vanilla_benign"
        ),
    ]

    monkeypatch.setattr(
        "agguardrails.features.format_prompt",
        lambda tokenizer, prompt: f"formatted::{prompt}",
    )

    dataset = extract_last_token_hidden_states(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        examples=examples,
        layers=[0],
        batch_size=2,
        max_length=16,
    )

    np.testing.assert_array_equal(
        dataset.features_by_layer[0],
        np.array([[8.0, 9.0], [18.0, 19.0]], dtype=np.float32),
    )


def test_extract_last_token_hidden_states_supports_last_instruction(monkeypatch):
    class FakeTokenizer:
        padding_side = "left"

        def __call__(self, prompts, return_tensors, padding, truncation, max_length):
            assert prompts == ["formatted::alpha", "formatted::beta"]
            return {
                "input_ids": torch.tensor(
                    [[101, 11, 12, 102, 103], [0, 101, 21, 102, 103]],
                    dtype=torch.int64,
                ),
                "attention_mask": torch.tensor(
                    [[1, 1, 1, 1, 1], [0, 1, 1, 1, 1]],
                    dtype=torch.int64,
                ),
            }

        def encode(self, prompt, add_special_tokens):
            assert add_special_tokens is False
            return {"alpha": [11, 12], "beta": [21]}[prompt]

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            assert tokenize is True
            assert add_generation_prompt is False
            prompt = messages[0]["content"]
            return {"alpha": [101, 11, 12, 102], "beta": [101, 21, 102]}[prompt]

    class FakeModel:
        device = torch.device("cpu")

        def __call__(self, **kwargs):
            batch = kwargs["input_ids"].shape[0]
            seq_len = kwargs["input_ids"].shape[1]
            hidden = torch.arange(batch * seq_len * 2, dtype=torch.float32).reshape(
                batch, seq_len, 2
            )
            return SimpleNamespace(hidden_states=(hidden,))

    examples = [
        PromptExample(
            "ex-1", "alpha", 1, "train", "wildjailbreak", "1", "vanilla_harmful"
        ),
        PromptExample(
            "ex-2", "beta", 0, "train", "wildjailbreak", "2", "vanilla_benign"
        ),
    ]

    monkeypatch.setattr(
        "agguardrails.features.format_prompt",
        lambda tokenizer, prompt: f"formatted::{prompt}",
    )

    dataset = extract_last_token_hidden_states(
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
        examples=examples,
        layers=[0],
        batch_size=2,
        max_length=16,
        token_position="last_instruction",
    )

    np.testing.assert_array_equal(
        dataset.features_by_layer[0],
        np.array([[4.0, 5.0], [14.0, 15.0]], dtype=np.float32),
    )
