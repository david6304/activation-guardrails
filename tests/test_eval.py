"""Tests for agguardrails.eval."""

import numpy as np

from agguardrails.eval import (
    evaluate_binary_classifier,
    format_results_row,
    summarize_scores_at_threshold,
    threshold_at_target_fpr,
)


def test_threshold_at_target_fpr_respects_constraint():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.05, 0.10, 0.20, 0.60, 0.80, 0.95])

    threshold, achieved_fpr, tpr = threshold_at_target_fpr(
        y_true,
        y_score,
        target_fpr=0.0,
    )

    assert achieved_fpr == 0.0
    assert threshold == 0.6
    assert tpr == 1.0


def test_evaluate_binary_classifier_returns_expected_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.05, 0.10, 0.20, 0.60, 0.80, 0.95])

    result = evaluate_binary_classifier(y_true, y_score, target_fpr=0.0)

    assert result.threshold == 0.6
    assert result.achieved_fpr == 0.0
    assert result.tpr_at_threshold == 1.0
    assert result.positive_predictions == 3
    assert result.roc_auc == 1.0


def test_format_results_row_flattens_result():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.7, 0.8])
    result = evaluate_binary_classifier(y_true, y_score, target_fpr=0.0)

    row = format_results_row(
        model_name="tfidf_lr",
        split="test",
        result=result,
        metadata={"layer": "none"},
    )

    assert row["model_name"] == "tfidf_lr"
    assert row["split"] == "test"
    assert row["layer"] == "none"
    assert row["roc_auc"] == 1.0


def test_summarize_scores_at_threshold_handles_single_class_split():
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.9, 0.2, 0.8])

    summary = summarize_scores_at_threshold(
        y_true,
        y_score,
        threshold=0.5,
    )

    assert summary["threshold"] == 0.5
    assert summary["roc_auc"] is None
    assert summary["achieved_fpr"] == 0.0
    assert summary["tpr_at_threshold"] == 2 / 3
    assert summary["positive_predictions"] == 2
    assert summary["positive_prediction_rate"] == 2 / 3
    assert summary["n_examples"] == 3
    assert summary["n_positive"] == 3
    assert summary["n_negative"] == 0
