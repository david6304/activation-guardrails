import pytest

from agguardrails.metrics import (
    evaluate_binary_scores,
    evaluate_frozen_threshold_transfer,
    select_threshold_at_fpr,
)


def test_select_threshold_at_one_percent_fpr_prefers_highest_tpr_under_constraint():
    labels = [0] * 100 + [1] * 5
    scores = [index / 1000 for index in range(99)] + [0.3]
    scores += [0.25, 0.4, 0.7, 0.8, 0.9]

    selection = select_threshold_at_fpr(labels, scores, max_fpr=0.01)

    assert selection.threshold == 0.25
    assert selection.validation.false_positives == 1
    assert selection.validation.fpr == 0.01
    assert selection.validation.tpr == 1.0
    assert selection.validation.roc_auc == pytest.approx(0.998)


def test_evaluate_frozen_threshold_transfer_keeps_validation_threshold():
    validation_labels = [0] * 100 + [1, 1]
    validation_scores = [index / 1000 for index in range(100)] + [0.8, 0.9]
    transfer_labels = [0, 0, 1, 1]
    transfer_scores = [0.1, 0.85, 0.7, 0.95]

    selection, transfer = evaluate_frozen_threshold_transfer(
        validation_labels,
        validation_scores,
        transfer_labels,
        transfer_scores,
        max_fpr=0.01,
    )

    assert selection.threshold == 0.8
    assert transfer.threshold == selection.threshold
    assert transfer.true_positives == 1
    assert transfer.false_positives == 1
    assert transfer.tpr == 0.5
    assert transfer.fpr == 0.5


def test_one_class_subset_returns_no_roc_auc_but_keeps_count_metrics():
    metrics = evaluate_binary_scores([1, 1, 1], [0.2, 0.8, 0.9], threshold=0.5)

    assert metrics.roc_auc is None
    assert metrics.fpr is None
    assert metrics.tpr == pytest.approx(2 / 3)
    assert metrics.positives == 3
    assert metrics.negatives == 0


def test_threshold_selection_requires_negative_validation_examples():
    with pytest.raises(ValueError, match="without negative examples"):
        select_threshold_at_fpr([1, 1], [0.2, 0.3], max_fpr=0.01)
