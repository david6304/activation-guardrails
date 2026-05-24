"""Metric helpers for CC++-style binary safety evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class BinaryMetrics:
    threshold: float
    tpr: float | None
    fpr: float | None
    roc_auc: float | None
    positives: int
    negatives: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    max_fpr: float
    validation: BinaryMetrics


def select_threshold_at_fpr(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    max_fpr: float = 0.01,
) -> ThresholdSelection:
    """Select the highest-recall threshold with validation FPR <= max_fpr."""
    label_array, score_array = _as_arrays(labels, scores)
    if not 0 <= max_fpr <= 1:
        raise ValueError("max_fpr must be between 0 and 1")
    if not np.any(label_array == 0):
        raise ValueError("Cannot select an FPR threshold without negative examples")

    candidates = [math.inf, *sorted(set(score_array.tolist()), reverse=True)]
    best_metrics: BinaryMetrics | None = None
    for threshold in candidates:
        metrics = evaluate_binary_scores(
            label_array,
            score_array,
            threshold=threshold,
        )
        if metrics.fpr is None or metrics.fpr > max_fpr:
            continue
        if best_metrics is None or _is_better_threshold(metrics, best_metrics):
            best_metrics = metrics

    if best_metrics is None:
        raise RuntimeError("No threshold candidate satisfied the FPR constraint")

    return ThresholdSelection(
        threshold=best_metrics.threshold,
        max_fpr=max_fpr,
        validation=best_metrics,
    )


def evaluate_binary_scores(
    labels: Iterable[int],
    scores: Iterable[float],
    *,
    threshold: float,
) -> BinaryMetrics:
    """Evaluate binary labels where larger scores are more harmful."""
    label_array, score_array = _as_arrays(labels, scores)
    predictions = score_array >= threshold
    positives = int(np.sum(label_array == 1))
    negatives = int(np.sum(label_array == 0))
    true_positives = int(np.sum(predictions & (label_array == 1)))
    false_positives = int(np.sum(predictions & (label_array == 0)))
    true_negatives = int(np.sum(~predictions & (label_array == 0)))
    false_negatives = int(np.sum(~predictions & (label_array == 1)))

    return BinaryMetrics(
        threshold=float(threshold),
        tpr=true_positives / positives if positives else None,
        fpr=false_positives / negatives if negatives else None,
        roc_auc=_roc_auc_or_none(label_array, score_array),
        positives=positives,
        negatives=negatives,
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
    )


def evaluate_frozen_threshold_transfer(
    validation_labels: Iterable[int],
    validation_scores: Iterable[float],
    transfer_labels: Iterable[int],
    transfer_scores: Iterable[float],
    *,
    max_fpr: float = 0.01,
) -> tuple[ThresholdSelection, BinaryMetrics]:
    selection = select_threshold_at_fpr(
        validation_labels,
        validation_scores,
        max_fpr=max_fpr,
    )
    transfer = evaluate_binary_scores(
        transfer_labels,
        transfer_scores,
        threshold=selection.threshold,
    )
    return selection, transfer


def _as_arrays(
    labels: Iterable[int],
    scores: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    label_array = np.asarray(list(labels), dtype=int)
    score_array = np.asarray(list(scores), dtype=float)
    if label_array.shape != score_array.shape:
        raise ValueError("labels and scores must have the same length")
    if label_array.ndim != 1:
        raise ValueError("labels and scores must be one-dimensional")
    if label_array.size == 0:
        raise ValueError("labels and scores must be non-empty")
    invalid_labels = set(label_array.tolist()) - {0, 1}
    if invalid_labels:
        raise ValueError(f"labels must be binary 0/1, got {sorted(invalid_labels)}")
    if not np.all(np.isfinite(score_array)):
        raise ValueError("scores must be finite")
    return label_array, score_array


def _roc_auc_or_none(label_array: np.ndarray, score_array: np.ndarray) -> float | None:
    if len(set(label_array.tolist())) < 2:
        return None
    return float(roc_auc_score(label_array, score_array))


def _is_better_threshold(candidate: BinaryMetrics, current: BinaryMetrics) -> bool:
    candidate_tpr = -1.0 if candidate.tpr is None else candidate.tpr
    current_tpr = -1.0 if current.tpr is None else current.tpr
    if candidate_tpr != current_tpr:
        return candidate_tpr > current_tpr
    candidate_fpr = 1.0 if candidate.fpr is None else candidate.fpr
    current_fpr = 1.0 if current.fpr is None else current.fpr
    if candidate_fpr != current_fpr:
        return candidate_fpr < current_fpr
    return candidate.threshold < current.threshold
