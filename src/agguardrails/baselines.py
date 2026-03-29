"""Text-only baselines for the MVP harmful-intent detection task."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from agguardrails.data import PromptExample
from agguardrails.eval import BinaryEvalResult, evaluate_binary_classifier


@dataclass(frozen=True)
class BaselineArtifacts:
    """Artifacts and metrics from a fitted text baseline."""

    pipeline: Pipeline
    val_result: BinaryEvalResult
    test_result: BinaryEvalResult


def examples_to_text_and_labels(
    examples: list[PromptExample],
) -> tuple[list[str], np.ndarray]:
    """Convert canonical prompt examples into raw texts and label arrays."""
    texts = [example.prompt for example in examples]
    labels = np.array([example.label for example in examples], dtype=np.int64)
    return texts, labels


def build_tfidf_lr_pipeline(
    *,
    max_features: int,
    c: float,
    max_iter: int = 1000,
    random_state: int = 42,
) -> Pipeline:
    """Construct the frozen MVP TF-IDF + LogisticRegression baseline."""
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    max_features=max_features,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=c,
                    max_iter=max_iter,
                    random_state=random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )


def fit_text_baseline(
    *,
    train_examples: list[PromptExample],
    val_examples: list[PromptExample],
    test_examples: list[PromptExample],
    max_features: int,
    c: float,
    target_fpr: float,
    random_state: int = 42,
) -> BaselineArtifacts:
    """Train TF-IDF + LR and evaluate on validation and test splits."""
    pipeline = build_tfidf_lr_pipeline(
        max_features=max_features,
        c=c,
        random_state=random_state,
    )

    train_texts, y_train = examples_to_text_and_labels(train_examples)
    val_texts, y_val = examples_to_text_and_labels(val_examples)
    test_texts, y_test = examples_to_text_and_labels(test_examples)

    pipeline.fit(train_texts, y_train)
    val_scores = pipeline.predict_proba(val_texts)[:, 1]
    val_result = evaluate_binary_classifier(y_val, val_scores, target_fpr=target_fpr)

    test_scores = pipeline.predict_proba(test_texts)[:, 1]
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

    return BaselineArtifacts(
        pipeline=pipeline,
        val_result=val_result,
        test_result=test_result,
    )
