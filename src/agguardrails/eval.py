"""Evaluation helpers for fixed-FPR operating points and report tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass(frozen=True)
class BinaryEvalResult:
    """Evaluation result for one binary harmful-intent detector."""

    roc_auc: float
    threshold: float
    achieved_fpr: float
    tpr_at_threshold: float
    positive_predictions: int


def threshold_at_target_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_fpr: float,
) -> tuple[float, float, float]:
    """Select the highest threshold whose FPR is at or below target_fpr."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        msg = "No threshold achieves the requested target_fpr."
        raise ValueError(msg)

    idx = valid[-1]
    return float(thresholds[idx]), float(fpr[idx]), float(tpr[idx])


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    target_fpr: float,
) -> BinaryEvalResult:
    """Compute ROC-AUC and TPR at a threshold chosen by target FPR."""
    threshold, achieved_fpr, tpr = threshold_at_target_fpr(
        y_true,
        y_score,
        target_fpr=target_fpr,
    )
    y_pred = y_score >= threshold

    return BinaryEvalResult(
        roc_auc=float(roc_auc_score(y_true, y_score)),
        threshold=threshold,
        achieved_fpr=achieved_fpr,
        tpr_at_threshold=tpr,
        positive_predictions=int(y_pred.sum()),
    )


def summarize_scores_at_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int | None]:
    """Summarise classifier behaviour at a fixed pre-selected threshold.

    This is used when a threshold is chosen on validation and then transferred
    unchanged to another split such as test or adversarial holdout.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    y_pred = y_score >= threshold

    n_examples = int(len(y_true))
    n_positive = int((y_true == 1).sum())
    n_negative = int((y_true == 0).sum())
    positive_predictions = int(y_pred.sum())
    tpr = float(((y_pred == 1) & (y_true == 1)).sum() / max(n_positive, 1))
    fpr = float(((y_pred == 1) & (y_true == 0)).sum() / max(n_negative, 1))

    roc_auc: float | None = None
    if len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_score))

    return {
        "threshold": float(threshold),
        "roc_auc": roc_auc,
        "achieved_fpr": fpr,
        "tpr_at_threshold": tpr,
        "positive_predictions": positive_predictions,
        "positive_prediction_rate": float(positive_predictions / max(n_examples, 1)),
        "n_examples": n_examples,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def format_results_row(
    *,
    model_name: str,
    split: str,
    result: BinaryEvalResult,
    metadata: dict[str, str | int | float] | None = None,
) -> dict[str, str | int | float]:
    """Create one flat result row suitable for JSON/CSV table output."""
    row: dict[str, str | int | float] = {
        "model_name": model_name,
        "split": split,
        "roc_auc": result.roc_auc,
        "threshold": result.threshold,
        "achieved_fpr": result.achieved_fpr,
        "tpr_at_threshold": result.tpr_at_threshold,
        "positive_predictions": result.positive_predictions,
    }
    if metadata:
        row.update(metadata)
    return row
