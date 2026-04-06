"""Difference-in-means Latent Guard baseline for activation features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agguardrails.eval import BinaryEvalResult, evaluate_binary_classifier


@dataclass(frozen=True)
class LatentGuardResult:
    """Artifacts and metrics from one fitted Latent Guard direction."""

    layer: int
    direction: np.ndarray
    val_result: BinaryEvalResult
    test_result: BinaryEvalResult


def fit_latent_guard_direction(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> np.ndarray:
    """Fit a unit-norm difference-in-means direction for the positive class."""
    x_train = np.asarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train)

    positive = x_train[y_train == 1]
    negative = x_train[y_train == 0]
    if len(positive) == 0 or len(negative) == 0:
        msg = "Latent Guard requires both positive and negative training examples."
        raise ValueError(msg)

    direction = positive.mean(axis=0) - negative.mean(axis=0)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        msg = "Latent Guard direction has zero norm."
        raise ValueError(msg)
    return direction / norm


def score_latent_guard(features: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project features onto a fitted Latent Guard direction."""
    features = np.asarray(features, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    return features @ direction


def fit_latent_guard_for_layer(
    *,
    layer: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    target_fpr: float,
) -> LatentGuardResult:
    """Fit and evaluate Latent Guard on one activation layer."""
    direction = fit_latent_guard_direction(x_train, y_train)

    val_scores = score_latent_guard(x_val, direction)
    val_result = evaluate_binary_classifier(y_val, val_scores, target_fpr=target_fpr)

    test_scores = score_latent_guard(x_test, direction)
    test_threshold = val_result.threshold
    test_predictions = test_scores >= test_threshold
    test_result = BinaryEvalResult(
        roc_auc=evaluate_binary_classifier(
            y_test, test_scores, target_fpr=target_fpr
        ).roc_auc,
        threshold=test_threshold,
        achieved_fpr=float(
            ((test_predictions == 1) & (y_test == 0)).sum()
            / max((y_test == 0).sum(), 1)
        ),
        tpr_at_threshold=float(
            ((test_predictions == 1) & (y_test == 1)).sum()
            / max((y_test == 1).sum(), 1)
        ),
        positive_predictions=int(test_predictions.sum()),
    )
    return LatentGuardResult(
        layer=layer,
        direction=direction,
        val_result=val_result,
        test_result=test_result,
    )


def select_best_latent_guard(results: list[LatentGuardResult]) -> LatentGuardResult:
    """Select the best layer using validation TPR, then ROC-AUC as a tiebreak."""
    if not results:
        raise ValueError("results must not be empty")
    return max(
        results,
        key=lambda result: (
            result.val_result.tpr_at_threshold,
            result.val_result.roc_auc,
            -result.layer,
        ),
    )
