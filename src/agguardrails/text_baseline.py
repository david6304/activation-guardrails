"""TF-IDF text baseline for WildJailbreak harmfulness detection."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from agguardrails.metrics import (
    BinaryMetrics,
    ThresholdSelection,
    evaluate_binary_scores,
    select_threshold_at_fpr,
)
from agguardrails.response_cache import (
    current_git_commit,
    load_normalized_examples,
    sha256_file,
    sha256_text,
)

SCHEMA_VERSION = "agguardrails.text_baseline.v1"
BASELINE_NAME = "tfidf_logistic_regression"


def run_tfidf_logistic_baseline(
    *,
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    config_path: str | Path,
    dataset_artifact_path: str | Path,
    model_path: str | Path,
    scores_path: str | Path,
    metrics_path: str | Path,
    table_path: str | Path,
    created_at: str,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Train on vanilla train, select on vanilla val, evaluate frozen threshold."""
    baseline_config = baseline_settings(config)
    seed = int(baseline_config["seed"])
    max_fpr = float(baseline_config["threshold"]["max_fpr"])
    groups = split_baseline_examples(examples)

    pipeline = build_tfidf_logistic_pipeline(baseline_config, seed=seed)
    pipeline.fit(
        [row["prompt"] for row in groups["train"]],
        [int(row["label"]) for row in groups["train"]],
    )

    scored_groups = {
        split: score_examples(pipeline, rows) for split, rows in groups.items()
    }
    selection = select_threshold_at_fpr(
        [row["label"] for row in groups["val"]],
        [row["score"] for row in scored_groups["val"]],
        max_fpr=max_fpr,
    )
    metrics = {
        "validation": selection.validation,
        "vanilla_test": evaluate_binary_scores(
            [row["label"] for row in groups["test"]],
            [row["score"] for row in scored_groups["test"]],
            threshold=selection.threshold,
        ),
        "adversarial_transfer": evaluate_binary_scores(
            [row["label"] for row in groups["transfer"]],
            [row["score"] for row in scored_groups["transfer"]],
            threshold=selection.threshold,
        ),
    }

    metadata = build_text_baseline_metadata(
        config=config,
        baseline_config=baseline_config,
        config_path=config_path,
        dataset_artifact_path=dataset_artifact_path,
        model_path=model_path,
        scores_path=scores_path,
        metrics_path=metrics_path,
        table_path=table_path,
        created_at=created_at,
        selection=selection,
        metrics=metrics,
        groups=groups,
        git_commit=git_commit,
    )
    write_text_baseline_outputs(
        pipeline=pipeline,
        scored_groups=scored_groups,
        metadata=metadata,
        model_path=model_path,
        scores_path=scores_path,
        metrics_path=metrics_path,
        table_path=table_path,
    )
    return metadata


def load_and_run_tfidf_logistic_baseline(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    created_at: str,
) -> dict[str, Any]:
    outputs = config["outputs"]
    dataset_path = Path(outputs["data_path"])
    examples = load_normalized_examples(dataset_path)
    return run_tfidf_logistic_baseline(
        examples=examples,
        config=config,
        config_path=config_path,
        dataset_artifact_path=dataset_path,
        model_path=outputs["text_baseline_model_path"],
        scores_path=outputs["text_baseline_scores_path"],
        metrics_path=outputs["text_baseline_metrics_path"],
        table_path=outputs["text_baseline_table_path"],
        created_at=created_at,
    )


def baseline_settings(config: dict[str, Any]) -> dict[str, Any]:
    baseline_config = dict(config.get("text_baseline", {}))
    sampling_seed = config.get("sampling", {}).get("seed", 0)
    threshold = dict(baseline_config.get("threshold", {}))
    tfidf = dict(baseline_config.get("tfidf", {}))
    logistic = dict(baseline_config.get("logistic_regression", {}))
    return {
        "name": baseline_config.get("name", BASELINE_NAME),
        "seed": int(baseline_config.get("seed", sampling_seed)),
        "threshold": {
            "max_fpr": float(threshold.get("max_fpr", 0.01)),
            "selection_split": threshold.get("selection_split", "val"),
            "selection_family": threshold.get("selection_family", "vanilla"),
            "rule": threshold.get("rule", "maximize_tpr_under_fpr"),
        },
        "tfidf": {
            "lowercase": bool(tfidf.get("lowercase", True)),
            "ngram_range": list(tfidf.get("ngram_range", [1, 2])),
            "min_df": tfidf.get("min_df", 1),
            "max_df": tfidf.get("max_df", 1.0),
            "max_features": tfidf.get("max_features"),
            "sublinear_tf": bool(tfidf.get("sublinear_tf", True)),
        },
        "logistic_regression": {
            "C": float(logistic.get("C", 1.0)),
            "max_iter": int(logistic.get("max_iter", 1000)),
            "solver": logistic.get("solver", "liblinear"),
            "class_weight": logistic.get("class_weight"),
        },
    }


def build_tfidf_logistic_pipeline(
    baseline_config: dict[str, Any],
    *,
    seed: int,
) -> Pipeline:
    tfidf_config = dict(baseline_config["tfidf"])
    tfidf_config["ngram_range"] = tuple(tfidf_config["ngram_range"])
    logistic_config = dict(baseline_config["logistic_regression"])
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**tfidf_config)),
            (
                "classifier",
                LogisticRegression(random_state=seed, **logistic_config),
            ),
        ]
    )


def split_baseline_examples(
    examples: Iterable[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups = {
        "train": [
            row
            for row in examples
            if row["split"] == "train" and row["source_family"] == "vanilla"
        ],
        "val": [
            row
            for row in examples
            if row["split"] == "val" and row["source_family"] == "vanilla"
        ],
        "test": [
            row
            for row in examples
            if row["split"] == "test" and row["source_family"] == "vanilla"
        ],
        "transfer": [
            row
            for row in examples
            if row["split"] == "transfer"
            and row["source_family"] == "adversarial"
        ],
    }
    for split, rows in groups.items():
        _require_binary_labels(rows, split=split)
    return {
        split: sorted(rows, key=lambda row: row["example_id"])
        for split, rows in groups.items()
    }


def score_examples(
    pipeline: Pipeline,
    examples: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = list(examples)
    harmful_class_index = list(pipeline.classes_).index(1)
    scores = pipeline.predict_proba([row["prompt"] for row in rows])[
        :,
        harmful_class_index,
    ]
    return [
        {
            "example_id": str(row["example_id"]),
            "split": str(row["split"]),
            "label": int(row["label"]),
            "data_type": str(row["data_type"]),
            "source_family": str(row["source_family"]),
            "tactics": [str(item) for item in row.get("tactics", [])],
            "score": float(score),
            "prompt_sha256": sha256_text(str(row["prompt"])),
        }
        for row, score in zip(rows, scores, strict=True)
    ]


def build_text_baseline_metadata(
    *,
    config: dict[str, Any],
    baseline_config: dict[str, Any],
    config_path: str | Path,
    dataset_artifact_path: str | Path,
    model_path: str | Path,
    scores_path: str | Path,
    metrics_path: str | Path,
    table_path: str | Path,
    created_at: str,
    selection: ThresholdSelection,
    metrics: dict[str, BinaryMetrics],
    groups: dict[str, list[dict[str, Any]]],
    git_commit: str | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_artifact_path)
    model = dict(config.get("model", {}))
    dataset = dict(config.get("dataset", {}))
    return {
        "schema_version": SCHEMA_VERSION,
        "baseline": baseline_config["name"],
        "config_path": str(config_path),
        "git_commit": git_commit if git_commit is not None else current_git_commit(),
        "created_at": created_at,
        "seed": baseline_config["seed"],
        "dataset_artifact": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "model": {
            "id": model.get("id"),
            "revision": model.get("revision"),
        },
        "dataset": {
            "id": dataset.get("id"),
            "revision": dataset.get("revision"),
            "name": dataset.get("name"),
            "split": dataset.get("split", dataset.get("subset")),
        },
        "artifacts": {
            "model_path": str(model_path),
            "scores_path": str(scores_path),
            "metrics_path": str(metrics_path),
            "table_path": str(table_path),
        },
        "training": {
            "split": "train",
            "source_family": "vanilla",
            "counts": _counts_for_rows(groups["train"]),
        },
        "evaluation": {
            "validation": {
                "split": "val",
                "source_family": "vanilla",
                "counts": _counts_for_rows(groups["val"]),
            },
            "vanilla_test": {
                "split": "test",
                "source_family": "vanilla",
                "counts": _counts_for_rows(groups["test"]),
            },
            "adversarial_transfer": {
                "split": "transfer",
                "source_family": "adversarial",
                "counts": _counts_for_rows(groups["transfer"]),
            },
        },
        "threshold_rule": {
            "rule": baseline_config["threshold"]["rule"],
            "selection_split": baseline_config["threshold"]["selection_split"],
            "selection_family": baseline_config["threshold"]["selection_family"],
            "max_fpr": selection.max_fpr,
            "frozen_threshold": selection.threshold,
            "score_direction": "larger_is_more_harmful",
        },
        "classifier": {
            "tfidf": baseline_config["tfidf"],
            "logistic_regression": baseline_config["logistic_regression"],
        },
        "metrics": {name: _metric_dict(metric) for name, metric in metrics.items()},
        "package_versions": package_versions(
            ["numpy", "scipy", "scikit-learn", "joblib"]
        ),
    }


def write_text_baseline_outputs(
    *,
    pipeline: Pipeline,
    scored_groups: dict[str, list[dict[str, Any]]],
    metadata: dict[str, Any],
    model_path: str | Path,
    scores_path: str | Path,
    metrics_path: str | Path,
    table_path: str | Path,
) -> None:
    model_output = Path(model_path)
    scores_output = Path(scores_path)
    metrics_output = Path(metrics_path)
    table_output = Path(table_path)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    scores_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    table_output.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_output)
    with scores_output.open("w", encoding="utf-8") as handle:
        for split in ("train", "val", "test", "transfer"):
            for row in scored_groups[split]:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    with metrics_output.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_result_table(metadata, table_output)


def write_result_table(metadata: dict[str, Any], table_path: str | Path) -> None:
    rows = []
    for name, split_label in (
        ("validation", "vanilla_val"),
        ("vanilla_test", "vanilla_test"),
        ("adversarial_transfer", "adversarial_transfer"),
    ):
        metric = metadata["metrics"][name]
        rows.append(
            {
                "baseline": metadata["baseline"],
                "split": split_label,
                "validation_max_fpr": metadata["threshold_rule"]["max_fpr"],
                "threshold": metric["threshold"],
                "tpr_at_frozen_threshold": metric["tpr"],
                "fpr": metric["fpr"],
                "roc_auc": metric["roc_auc"],
                "positives": metric["positives"],
                "negatives": metric["negatives"],
            }
        )

    with Path(table_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _metric_dict(metric: BinaryMetrics) -> dict[str, Any]:
    row = asdict(metric)
    for key, value in row.items():
        if isinstance(value, np.floating):
            row[key] = float(value)
    return row


def _counts_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(rows),
        "by_label": {
            str(label): count
            for label, count in sorted(
                Counter(int(row["label"]) for row in rows).items()
            )
        },
        "by_data_type": dict(
            sorted(Counter(str(row["data_type"]) for row in rows).items())
        ),
    }


def _require_binary_labels(rows: list[dict[str, Any]], *, split: str) -> None:
    if not rows:
        raise ValueError(f"No examples found for baseline split {split!r}")
    labels = {int(row["label"]) for row in rows}
    if labels != {0, 1}:
        raise ValueError(
            f"Baseline split {split!r} must contain both labels, got {sorted(labels)}"
        )
