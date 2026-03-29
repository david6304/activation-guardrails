"""Tests for agguardrails.features."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from agguardrails.features import ActivationDataset, load_activation_split, save_activation_dataset, validate_layer_indices


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
    )

    save_activation_dataset(
        dataset,
        output_dir=tmp_path,
        split="train",
        config_path="configs/mvp/mvp.yaml",
        model_name="Qwen/Qwen2.5-7B-Instruct",
    )
    loaded = load_activation_split(input_dir=tmp_path, split="train", layers=[8, 16])

    np.testing.assert_array_equal(loaded.features_by_layer[8], dataset.features_by_layer[8])
    np.testing.assert_array_equal(loaded.features_by_layer[16], dataset.features_by_layer[16])
    np.testing.assert_array_equal(loaded.labels, dataset.labels)
    assert loaded.example_ids == dataset.example_ids
