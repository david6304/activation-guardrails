from __future__ import annotations

import math

import numpy as np
import pytest

from agguardrails.metrics import (
    apply_threshold,
    fixed_fpr_point,
    flag_at_any_token,
    log_space_auc,
    roc_auc,
)


def test_roc_auc_returns_expected_value() -> None:
    assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)


def test_fixed_fpr_threshold_allows_only_requested_false_positive_rate() -> None:
    labels = [0, 0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.99, 0.25, 0.8, 0.9]

    point = fixed_fpr_point(labels, scores, max_fpr=0.25)

    assert point.false_positives == 1
    assert point.true_positives == 2
    assert point.fpr == pytest.approx(0.25)
    assert point.tpr == pytest.approx(2 / 3)


def test_fixed_fpr_threshold_handles_zero_false_positive_budget() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.4, 0.5, 0.6, 0.7]

    point = fixed_fpr_point(labels, scores, max_fpr=0.0)

    assert point.false_positives == 0
    assert point.true_positives == 2
    assert point.threshold > 0.5


def test_apply_frozen_validation_threshold_to_test_scores() -> None:
    validation = fixed_fpr_point([0, 0, 1, 1], [0.2, 0.8, 0.7, 0.9], max_fpr=0.0)
    test = apply_threshold(
        [0, 0, 1, 1],
        [0.1, 0.91, 0.92, 0.95],
        threshold=validation.threshold,
        max_fpr=0.0,
    )

    assert validation.false_positives == 0
    assert test.false_positives == 1
    assert test.tpr == 1.0


def test_log_space_auc_is_bounded() -> None:
    value = log_space_auc(
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0.1, 0.2, 0.3, 0.4, 0.55, 0.65, 0.75, 0.85],
        min_fpr=0.25,
        max_fpr=1.0,
        num_points=8,
    )

    assert 0.0 <= value <= 1.0
    assert not math.isnan(value)


def test_flag_at_any_token_uses_max_sequence_score() -> None:
    scores = flag_at_any_token([[0.1, 0.3, 0.2], [-1.0, -0.2]])

    np.testing.assert_allclose(scores, [0.3, -0.2])


def test_flag_at_any_token_rejects_empty_sequences() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        flag_at_any_token([[]])
