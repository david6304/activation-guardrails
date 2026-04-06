"""Tests for agguardrails.latent_guard."""

import numpy as np
import pytest

from agguardrails.eval import BinaryEvalResult
from agguardrails.latent_guard import (
    LatentGuardResult,
    fit_latent_guard_direction,
    fit_latent_guard_for_layer,
    score_latent_guard,
    select_best_latent_guard,
)


def test_fit_latent_guard_direction_normalizes_difference_in_means():
    x_train = np.array(
        [
            [0.0, 1.0],
            [0.0, 3.0],
            [2.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)

    direction = fit_latent_guard_direction(x_train, y_train)

    expected = np.array([3.0, -2.0], dtype=np.float32)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(direction, expected)
    np.testing.assert_allclose(np.linalg.norm(direction), 1.0)


def test_fit_latent_guard_direction_requires_both_classes():
    x_train = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    y_train = np.array([1, 1], dtype=np.int64)

    with pytest.raises(ValueError, match="requires both positive and negative"):
        fit_latent_guard_direction(x_train, y_train)


def test_fit_latent_guard_for_layer_uses_validation_threshold_on_test():
    x_train = np.array(
        [
            [0.0, 1.0],
            [0.0, 3.0],
            [2.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 1, 1], dtype=np.int64)
    x_val = np.array(
        [
            [0.0, 2.0],
            [0.0, 4.0],
            [3.0, 0.0],
            [5.0, 0.0],
        ],
        dtype=np.float32,
    )
    y_val = np.array([0, 0, 1, 1], dtype=np.int64)
    x_test = np.array(
        [
            [0.0, 1.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    y_test = np.array([0, 1], dtype=np.int64)

    result = fit_latent_guard_for_layer(
        layer=20,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        target_fpr=0.0,
    )

    expected_test_scores = score_latent_guard(x_test, result.direction)
    assert result.layer == 20
    assert result.val_result.threshold > 0.0
    assert result.test_result.threshold == result.val_result.threshold
    assert result.test_result.positive_predictions == int(
        (expected_test_scores >= result.val_result.threshold).sum()
    )
    assert result.test_result.roc_auc == 1.0


def test_select_best_latent_guard_prefers_validation_tpr_then_auc():
    result_low = LatentGuardResult(
        layer=9,
        direction=np.array([1.0, 0.0], dtype=np.float32),
        val_result=BinaryEvalResult(0.80, 0.5, 0.0, 0.60, 2),
        test_result=BinaryEvalResult(0.75, 0.5, 0.0, 0.55, 2),
    )
    result_high = LatentGuardResult(
        layer=20,
        direction=np.array([0.0, 1.0], dtype=np.float32),
        val_result=BinaryEvalResult(0.70, 0.5, 0.0, 0.70, 2),
        test_result=BinaryEvalResult(0.65, 0.5, 0.0, 0.65, 2),
    )

    best = select_best_latent_guard([result_low, result_high])

    assert best.layer == 20
