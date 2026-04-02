"""Helpers for frozen-model transfer evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from agguardrails.baselines import examples_to_text_and_labels
from agguardrails.data import PromptExample
from agguardrails.eval import summarize_scores_at_threshold


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    with Path(path).open() as f:
        return json.load(f)


def get_text_threshold(metrics: dict[str, Any]) -> float:
    """Read the validation-selected threshold from text-baseline metrics."""
    return float(metrics["val_metrics"]["threshold"])


def get_probe_selection(metrics: dict[str, Any]) -> tuple[int, float]:
    """Read best layer and validation threshold from probe metrics."""
    best_layer = int(metrics["best_layer"])
    threshold = float(metrics["per_layer"][str(best_layer)]["val"]["threshold"])
    return best_layer, threshold


def summarize_text_transfer(
    *,
    examples: list[PromptExample],
    pipeline,
    threshold: float,
) -> dict[str, float | int | None]:
    """Evaluate a frozen text pipeline on a new prompt dataset."""
    texts, labels = examples_to_text_and_labels(examples)
    scores = pipeline.predict_proba(texts)[:, 1]
    return summarize_scores_at_threshold(labels, scores, threshold=threshold)


def summarize_feature_transfer(
    *,
    labels: np.ndarray,
    features: np.ndarray,
    classifier,
    threshold: float,
) -> dict[str, float | int | None]:
    """Evaluate a frozen classifier on cached features."""
    scores = classifier.predict_proba(features)[:, 1]
    return summarize_scores_at_threshold(labels, scores, threshold=threshold)
