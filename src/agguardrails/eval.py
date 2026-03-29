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
