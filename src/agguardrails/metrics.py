"""Metric helpers for low-FPR CC++ replication reports."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class FixedFprPoint:
    max_fpr: float
    threshold: float
    fpr: float
    tpr: float
    false_positives: int
    true_positives: int
    negative_count: int
    positive_count: int


@dataclass(frozen=True)
class WilsonInterval:
    estimate: float
    lower: float
    upper: float
    count: int
    total: int
    confidence: float


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    labels, score_array = _validate_binary_scores(y_true, scores)
    return float(roc_auc_score(labels, score_array))


def wilson_interval(
    *,
    count: int,
    total: int,
    confidence: float = 0.95,
) -> WilsonInterval:
    """Wilson score interval for a binomial proportion."""

    if total <= 0:
        raise ValueError("total must be positive")
    if count < 0 or count > total:
        raise ValueError("count must be in [0, total]")
    if confidence != 0.95:
        raise ValueError("only 95% Wilson intervals are currently supported")

    z = 1.959963984540054
    phat = count / total
    denominator = 1 + z**2 / total
    center = (phat + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * math.sqrt((phat * (1 - phat) + z**2 / (4 * total)) / total)
        / denominator
    )
    return WilsonInterval(
        estimate=phat,
        lower=max(0.0, center - half_width),
        upper=min(1.0, center + half_width),
        count=count,
        total=total,
        confidence=confidence,
    )


def fixed_fpr_point(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    max_fpr: float,
) -> FixedFprPoint:
    """Pick the best validation threshold whose empirical FPR is <= max_fpr."""

    if not 0 <= max_fpr <= 1:
        raise ValueError(f"max_fpr must be in [0, 1], got {max_fpr}")
    labels, score_array = _validate_binary_scores(y_true, scores)

    candidates = _threshold_candidates(score_array)
    best: FixedFprPoint | None = None
    for threshold in candidates:
        point = _point_at_threshold(labels, score_array, max_fpr, threshold)
        if point.fpr > max_fpr:
            continue
        if best is None or (point.tpr, -point.threshold) > (best.tpr, -best.threshold):
            best = point

    if best is None:
        threshold = math.inf
        best = _point_at_threshold(labels, score_array, max_fpr, threshold)
    return best


def apply_threshold(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    threshold: float,
    max_fpr: float,
) -> FixedFprPoint:
    labels, score_array = _validate_binary_scores(y_true, scores)
    return _point_at_threshold(labels, score_array, max_fpr, threshold)


def log_space_auc(
    y_true: Sequence[int],
    scores: Sequence[float],
    *,
    min_fpr: float = 1e-4,
    max_fpr: float = 1e-1,
    num_points: int = 64,
) -> float:
    """Approximate mean TPR over a log-spaced low-FPR grid."""

    if not 0 < min_fpr <= max_fpr <= 1:
        raise ValueError("expected 0 < min_fpr <= max_fpr <= 1")
    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    grid = np.geomspace(min_fpr, max_fpr, num_points)
    tprs = [fixed_fpr_point(y_true, scores, max_fpr=float(fpr)).tpr for fpr in grid]
    log_grid = np.log(grid)
    width = log_grid[-1] - log_grid[0]
    # np.trapezoid replaced the deprecated np.trapz in NumPy 2.0+.
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(tprs, x=log_grid) / width)


def flag_at_any_token(token_scores: Sequence[Sequence[float]]) -> np.ndarray:
    """Aggregate per-token scores into one sequence score by max/any-token."""

    sequence_scores = []
    for index, scores in enumerate(token_scores):
        score_array = np.asarray(scores, dtype=float)
        if score_array.ndim != 1 or score_array.size == 0:
            raise ValueError(f"token_scores[{index}] must be a non-empty 1D sequence")
        sequence_scores.append(float(np.max(score_array)))
    return np.asarray(sequence_scores, dtype=float)


def _validate_binary_scores(
    y_true: Sequence[int], scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or score_array.ndim != 1:
        raise ValueError("y_true and scores must be 1D")
    if labels.shape != score_array.shape:
        raise ValueError("y_true and scores must have the same length")
    unique = set(labels.tolist())
    if unique - {0, 1}:
        raise ValueError(f"labels must be binary 0/1, got {sorted(unique)}")
    if len(unique) < 2:
        raise ValueError("both positive and negative labels are required")
    if not np.isfinite(score_array).all():
        raise ValueError("scores must be finite")
    return labels, score_array


def _threshold_candidates(scores: np.ndarray) -> list[float]:
    unique_scores = sorted({float(score) for score in scores}, reverse=True)
    max_score = unique_scores[0]
    return [math.nextafter(max_score, math.inf), *unique_scores, -math.inf]


def _point_at_threshold(
    labels: np.ndarray,
    scores: np.ndarray,
    max_fpr: float,
    threshold: float,
) -> FixedFprPoint:
    predictions = scores >= threshold
    positives = labels == 1
    negatives = labels == 0
    true_positives = int(np.sum(predictions & positives))
    false_positives = int(np.sum(predictions & negatives))
    positive_count = int(np.sum(positives))
    negative_count = int(np.sum(negatives))
    tpr = true_positives / positive_count
    fpr = false_positives / negative_count
    return FixedFprPoint(
        max_fpr=max_fpr,
        threshold=threshold,
        fpr=fpr,
        tpr=tpr,
        false_positives=false_positives,
        true_positives=true_positives,
        negative_count=negative_count,
        positive_count=positive_count,
    )
