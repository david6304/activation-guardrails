"""Activation-based logistic probes for harmful-intent detection."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from agguardrails.eval import BinaryEvalResult, evaluate_binary_classifier


@dataclass(frozen=True)
class ProbeResult:
    """Artifacts and metrics from one fitted layer probe."""

    layer: int
    probe: LogisticRegression
    val_result: BinaryEvalResult
    test_result: BinaryEvalResult
    best_c: float | None = None


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
    penalty: str = "l2",
    random_state: int = 42,
) -> ProbeResult:
    """Train and evaluate one logistic probe on a single layer.

    Args:
        penalty: Regularisation type — ``"l1"`` or ``"l2"``.  SAE4Safety uses
            ``"l1"`` with ``C=500`` (penalty=0.002).  The main experiment uses
            ``"l2"`` with ``C=1.0``.  The ``"liblinear"`` solver supports both.

    Note:
        sklearn >= 1.8 deprecated the ``penalty`` parameter in favour of
        ``l1_ratio`` (0 = L2, 1 = L1).  We map internally to suppress the
        FutureWarning.
    """
    _l1_ratio_map = {"l1": 1.0, "l2": 0.0}
    if penalty not in _l1_ratio_map:
        msg = f"penalty must be 'l1' or 'l2', got {penalty!r}"
        raise ValueError(msg)
    probe = LogisticRegression(
        C=c,
        l1_ratio=_l1_ratio_map[penalty],
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


def fit_probe_for_layer_cv(
    *,
    layer: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_values: list[float],
    max_iter: int,
    target_fpr: float,
    penalty: str = "l2",
    random_state: int = 42,
    cv_folds: int = 5,
) -> ProbeResult:
    """Train and evaluate one logistic probe with cross-validated C selection.

    For each C in ``c_values``, runs stratified k-fold CV on ``x_train``/``y_train``
    and scores by mean ROC-AUC. Refits the best C on the full training set, then
    evaluates on val and test exactly as ``fit_probe_for_layer``.

    Args:
        c_values: Candidate inverse regularisation strengths to search.
        cv_folds: Number of stratified cross-validation folds.
    """
    _l1_ratio_map = {"l1": 1.0, "l2": 0.0}
    if penalty not in _l1_ratio_map:
        msg = f"penalty must be 'l1' or 'l2', got {penalty!r}"
        raise ValueError(msg)

    best_c = c_values[0]
    best_mean_auc = -1.0
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    for c in c_values:
        candidate = LogisticRegression(
            C=c,
            l1_ratio=_l1_ratio_map[penalty],
            max_iter=max_iter,
            random_state=random_state,
            solver="liblinear",
        )
        fold_aucs = cross_val_score(
            candidate, x_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
        mean_auc = float(fold_aucs.mean())
        if mean_auc > best_mean_auc:
            best_mean_auc = mean_auc
            best_c = c

    probe = LogisticRegression(
        C=best_c,
        l1_ratio=_l1_ratio_map[penalty],
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
        best_c=best_c,
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
