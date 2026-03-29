"""Activation-based logistic probes for harmful-intent detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from agguardrails.eval import BinaryEvalResult, evaluate_binary_classifier


@dataclass(frozen=True)
class ProbeResult:
    """Artifacts and metrics from one fitted layer probe."""

    layer: int
    probe: LogisticRegression
    val_result: BinaryEvalResult
    test_result: BinaryEvalResult


def fit_probe_for_layer(
    *,
    layer: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c: float,
    max_iter: int,
    target_fpr: float,
    random_state: int = 42,
) -> ProbeResult:
    """Train and evaluate one logistic probe on a single layer."""
    probe = LogisticRegression(
        C=c,
        max_iter=max_iter,
        random_state=random_state,
        solver="liblinear",
    )
    probe.fit(x_train, y_train)

    val_scores = probe.predict_proba(x_val)[:, 1]
    val_result = evaluate_binary_classifier(y_val, val_scores, target_fpr=target_fpr)

    test_scores = probe.predict_proba(x_test)[:, 1]
    test_threshold = val_result.threshold
    test_predictions = test_scores >= test_threshold
    test_result = BinaryEvalResult(
        roc_auc=evaluate_binary_classifier(y_test, test_scores, target_fpr=target_fpr).roc_auc,
        threshold=test_threshold,
        achieved_fpr=float(
            ((test_predictions == 1) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1)
        ),
        tpr_at_threshold=float(
            ((test_predictions == 1) & (y_test == 1)).sum() / max((y_test == 1).sum(), 1)
        ),
        positive_predictions=int(test_predictions.sum()),
    )

    return ProbeResult(
        layer=layer,
        probe=probe,
        val_result=val_result,
        test_result=test_result,
    )


def select_best_probe(probe_results: list[ProbeResult]) -> ProbeResult:
    """Select the best layer using validation TPR, then ROC-AUC as a tiebreak."""
    if not probe_results:
        raise ValueError("probe_results must not be empty")
    return max(
        probe_results,
        key=lambda result: (
            result.val_result.tpr_at_threshold,
            result.val_result.roc_auc,
            -result.layer,
        ),
    )
