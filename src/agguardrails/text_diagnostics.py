"""Text-only separability diagnostics for CC++ dataset acceptance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from agguardrails.ccpp_data import NormalizedExchange
from agguardrails.metrics import roc_auc


@dataclass(frozen=True)
class TextSeparabilityConfig:
    max_allowed_roc_auc: float
    text_fields: tuple[str, ...] = ("user_text", "assistant_text")


def exchange_text(
    example: NormalizedExchange,
    *,
    fields: Sequence[str] = ("user_text", "assistant_text"),
) -> str:
    chunks = []
    for field in fields:
        value = getattr(example, field)
        chunks.append(str(value))
    return "\n".join(chunks)


def train_text_baseline(
    examples: Sequence[NormalizedExchange],
    *,
    fields: Sequence[str] = ("user_text", "assistant_text"),
) -> Pipeline:
    train_examples = [example for example in examples if example.split == "train"]
    if len({example.label for example in train_examples}) < 2:
        raise ValueError("train split must contain both labels")

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=50000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    solver="liblinear",
                    random_state=0,
                ),
            ),
        ]
    )
    model.fit(
        [exchange_text(example, fields=fields) for example in train_examples],
        [example.label for example in train_examples],
    )
    return model


def text_separability_report(
    examples: Sequence[NormalizedExchange],
    *,
    config: TextSeparabilityConfig,
) -> dict[str, Any]:
    model = train_text_baseline(examples, fields=config.text_fields)
    report: dict[str, Any] = {
        "max_allowed_roc_auc": config.max_allowed_roc_auc,
        "text_fields": list(config.text_fields),
        "splits": {},
        "gate": "passed",
    }
    for split in sorted({example.split for example in examples}):
        split_examples = [example for example in examples if example.split == split]
        labels = [example.label for example in split_examples]
        split_report: dict[str, Any] = {"count": len(split_examples)}
        if len(set(labels)) == 2:
            texts = [
                exchange_text(example, fields=config.text_fields)
                for example in split_examples
            ]
            scores = model.predict_proba(texts)[:, 1]
            split_report["roc_auc"] = roc_auc(labels, scores)
        else:
            split_report["status"] = "missing_binary_labels"
        report["splits"][split] = split_report

    gate_splits = [
        split_report
        for split, split_report in report["splits"].items()
        if split != "train" and "roc_auc" in split_report
    ]
    near_ceiling = [
        split_report["roc_auc"]
        for split_report in gate_splits
        if split_report["roc_auc"] >= config.max_allowed_roc_auc
    ]
    if near_ceiling:
        report["gate"] = "failed"
        report["blocked_reason"] = (
            "text baseline near ceiling; dataset may be surface-separable"
        )
    return report
