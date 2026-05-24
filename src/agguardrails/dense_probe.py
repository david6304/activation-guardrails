"""Dense activation logistic probe for WildJailbreak harmfulness detection."""

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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from agguardrails.activation_cache import (
    ActivationCacheRecord,
    activation_settings,
    load_activation_cache_index,
    validate_activation_cache,
)
from agguardrails.metrics import (
    BinaryMetrics,
    ThresholdSelection,
    evaluate_binary_scores,
    select_threshold_at_fpr,
)
from agguardrails.response_cache import current_git_commit, sha256_file

SCHEMA_VERSION = "agguardrails.dense_probe.v1"
PROBE_NAME = "dense_prompt_final_logistic_regression"


def run_dense_probe(
    *,
    activations: np.ndarray,
    records: list[ActivationCacheRecord],
    config: dict[str, Any],
    config_path: str | Path,
    dataset_artifact_path: str | Path,
    activation_cache_path: str | Path,
    activation_index_path: str | Path,
    model_path: str | Path,
    scores_path: str | Path,
    metrics_path: str | Path,
    table_path: str | Path,
    created_at: str,
    response_cache_path: str | Path | None = None,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Train on vanilla train, select on vanilla val, evaluate frozen threshold."""
    validate_activation_cache(records=records, activations=activations)
    probe_config = dense_probe_settings(config)
    seed = int(probe_config["seed"])
    max_fpr = float(probe_config["threshold"]["max_fpr"])
    groups = split_activation_records(records)

    scaler = StandardScaler()
    train_indices = [record.vector_index for record in groups["train"]]
    train_x = scaler.fit_transform(activations[train_indices])
    train_y = np.asarray([record.label for record in groups["train"]], dtype=int)

    classifier = LogisticRegression(
        random_state=seed,
        **probe_config["logistic_regression"],
    )
    classifier.fit(train_x, train_y)

    scored_groups = {
        split: score_activation_records(
            classifier,
            scaler,
            activations,
            rows,
        )
        for split, rows in groups.items()
    }
    selection = select_threshold_at_fpr(
        [record.label for record in groups["val"]],
        [row["score"] for row in scored_groups["val"]],
        max_fpr=max_fpr,
    )
    metrics = {
        "validation": selection.validation,
        "vanilla_test": evaluate_binary_scores(
            [record.label for record in groups["test"]],
            [row["score"] for row in scored_groups["test"]],
            threshold=selection.threshold,
        ),
        "adversarial_transfer": evaluate_binary_scores(
            [record.label for record in groups["transfer"]],
            [row["score"] for row in scored_groups["transfer"]],
            threshold=selection.threshold,
        ),
    }

    metadata = build_dense_probe_metadata(
        config=config,
        probe_config=probe_config,
        config_path=config_path,
        dataset_artifact_path=dataset_artifact_path,
        response_cache_path=response_cache_path,
        activation_cache_path=activation_cache_path,
        activation_index_path=activation_index_path,
        model_path=model_path,
        scores_path=scores_path,
        metrics_path=metrics_path,
        table_path=table_path,
        created_at=created_at,
        selection=selection,
        metrics=metrics,
        groups=groups,
        activations=activations,
        git_commit=git_commit,
    )
    write_dense_probe_outputs(
        classifier=classifier,
        scaler=scaler,
        scored_groups=scored_groups,
        metadata=metadata,
        model_path=model_path,
        scores_path=scores_path,
        metrics_path=metrics_path,
        table_path=table_path,
    )
    return metadata


def load_and_run_dense_probe(
    *,
    config: dict[str, Any],
    config_path: str | Path,
    created_at: str,
) -> dict[str, Any]:
    outputs = config["outputs"]
    activation_cache_path = Path(outputs["activation_cache_path"])
    activation_index_path = Path(outputs["activation_index_path"])
    activations, records = load_activation_cache_arrays(
        activation_cache_path,
        activation_index_path,
    )
    return run_dense_probe(
        activations=activations,
        records=records,
        config=config,
        config_path=config_path,
        dataset_artifact_path=outputs["data_path"],
        response_cache_path=outputs.get("response_cache_path"),
        activation_cache_path=activation_cache_path,
        activation_index_path=activation_index_path,
        model_path=outputs["dense_probe_model_path"],
        scores_path=outputs["dense_probe_scores_path"],
        metrics_path=outputs["dense_probe_metrics_path"],
        table_path=outputs["dense_probe_table_path"],
        created_at=created_at,
    )


def dense_probe_settings(config: dict[str, Any]) -> dict[str, Any]:
    probe_config = dict(config.get("dense_probe", {}))
    sampling_seed = config.get("sampling", {}).get("seed", 0)
    threshold = dict(probe_config.get("threshold", {}))
    logistic = dict(probe_config.get("logistic_regression", {}))
    return {
        "name": probe_config.get("name", PROBE_NAME),
        "seed": int(probe_config.get("seed", sampling_seed)),
        "feature_source": probe_config.get("feature_source", "activation_cache"),
        "threshold": {
            "max_fpr": float(threshold.get("max_fpr", 0.01)),
            "selection_split": threshold.get("selection_split", "val"),
            "selection_family": threshold.get("selection_family", "vanilla"),
            "rule": threshold.get("rule", "maximize_tpr_under_fpr"),
        },
        "logistic_regression": {
            "C": float(logistic.get("C", 1.0)),
            "max_iter": int(logistic.get("max_iter", 1000)),
            "solver": logistic.get("solver", "liblinear"),
            "class_weight": logistic.get("class_weight"),
        },
    }


def load_activation_cache_arrays(
    activation_cache_path: str | Path,
    activation_index_path: str | Path,
) -> tuple[np.ndarray, list[ActivationCacheRecord]]:
    records = load_activation_cache_index(activation_index_path)
    with np.load(activation_cache_path) as cache:
        activations = np.asarray(cache["activations"], dtype=np.float32)
        labels = np.asarray(cache["labels"], dtype=int)
        example_ids = [str(item) for item in cache["example_ids"].tolist()]
    validate_activation_cache(records=records, activations=activations)
    if labels.tolist() != [record.label for record in records]:
        raise ValueError("activation labels do not match index records")
    if example_ids != [record.example_id for record in records]:
        raise ValueError("activation example_ids do not match index records")
    return activations, records


def split_activation_records(
    records: Iterable[ActivationCacheRecord],
) -> dict[str, list[ActivationCacheRecord]]:
    rows = list(records)
    groups = {
        "train": [
            row
            for row in rows
            if row.split == "train" and row.source_family == "vanilla"
        ],
        "val": [
            row for row in rows if row.split == "val" and row.source_family == "vanilla"
        ],
        "test": [
            row
            for row in rows
            if row.split == "test" and row.source_family == "vanilla"
        ],
        "transfer": [
            row
            for row in rows
            if row.split == "transfer" and row.source_family == "adversarial"
        ],
    }
    for split, split_rows in groups.items():
        _require_binary_labels(split_rows, split=split)
    return {
        split: sorted(split_rows, key=lambda row: row.example_id)
        for split, split_rows in groups.items()
    }


def score_activation_records(
    classifier: LogisticRegression,
    scaler: StandardScaler,
    activations: np.ndarray,
    records: Iterable[ActivationCacheRecord],
) -> list[dict[str, Any]]:
    rows = list(records)
    if not rows:
        return []
    indices = [record.vector_index for record in rows]
    harmful_class_index = list(classifier.classes_).index(1)
    scores = classifier.predict_proba(scaler.transform(activations[indices]))[
        :,
        harmful_class_index,
    ]
    return [
        {
            "example_id": record.example_id,
            "split": record.split,
            "label": int(record.label),
            "data_type": record.data_type,
            "source_family": record.source_family,
            "tactics": list(record.tactics),
            "score": float(score),
            "vector_index": int(record.vector_index),
            "activation": dict(record.activation),
            "token": dict(record.token),
            "hashes": dict(record.hashes),
        }
        for record, score in zip(rows, scores, strict=True)
    ]


def build_dense_probe_metadata(
    *,
    config: dict[str, Any],
    probe_config: dict[str, Any],
    config_path: str | Path,
    dataset_artifact_path: str | Path,
    response_cache_path: str | Path | None,
    activation_cache_path: str | Path,
    activation_index_path: str | Path,
    model_path: str | Path,
    scores_path: str | Path,
    metrics_path: str | Path,
    table_path: str | Path,
    created_at: str,
    selection: ThresholdSelection,
    metrics: dict[str, BinaryMetrics],
    groups: dict[str, list[ActivationCacheRecord]],
    activations: np.ndarray,
    git_commit: str | None = None,
) -> dict[str, Any]:
    dataset_path = Path(dataset_artifact_path)
    activation_path = Path(activation_cache_path)
    index_path = Path(activation_index_path)
    model = dict(config.get("model", {}))
    tokenizer = dict(config.get("tokenizer") or model)
    dataset = dict(config.get("dataset", {}))
    return {
        "schema_version": SCHEMA_VERSION,
        "probe": probe_config["name"],
        "config_path": str(config_path),
        "git_commit": git_commit if git_commit is not None else current_git_commit(),
        "created_at": created_at,
        "seed": probe_config["seed"],
        "dataset_artifact": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "response_cache_artifact": _artifact_hash(response_cache_path),
        "activation_cache_artifact": {
            "path": str(activation_path),
            "sha256": sha256_file(activation_path),
        },
        "activation_index_artifact": {
            "path": str(index_path),
            "sha256": sha256_file(index_path),
        },
        "model": {
            "id": model.get("id"),
            "revision": model.get("revision"),
        },
        "tokenizer": {
            "id": tokenizer.get("id"),
            "revision": tokenizer.get("revision"),
        },
        "dataset": {
            "id": dataset.get("id"),
            "revision": dataset.get("revision"),
            "name": dataset.get("name"),
            "split": dataset.get("split", dataset.get("subset")),
        },
        "activation": activation_settings(config),
        "activation_array": {
            "shape": list(activations.shape),
            "dtype": str(activations.dtype),
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
            "counts": _counts_for_records(groups["train"]),
        },
        "evaluation": {
            "validation": {
                "split": "val",
                "source_family": "vanilla",
                "counts": _counts_for_records(groups["val"]),
            },
            "vanilla_test": {
                "split": "test",
                "source_family": "vanilla",
                "counts": _counts_for_records(groups["test"]),
            },
            "adversarial_transfer": {
                "split": "transfer",
                "source_family": "adversarial",
                "counts": _counts_for_records(groups["transfer"]),
            },
        },
        "threshold_rule": {
            "rule": probe_config["threshold"]["rule"],
            "selection_split": probe_config["threshold"]["selection_split"],
            "selection_family": probe_config["threshold"]["selection_family"],
            "max_fpr": selection.max_fpr,
            "frozen_threshold": selection.threshold,
            "score_direction": "larger_is_more_harmful",
        },
        "classifier": {
            "standardize": True,
            "logistic_regression": probe_config["logistic_regression"],
        },
        "metrics": {name: _metric_dict(metric) for name, metric in metrics.items()},
        "package_versions": package_versions(
            ["numpy", "scipy", "scikit-learn", "joblib"]
        ),
    }


def write_dense_probe_outputs(
    *,
    classifier: LogisticRegression,
    scaler: StandardScaler,
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

    joblib.dump(
        {
            "classifier": classifier,
            "scaler": scaler,
            "metadata": {
                "schema_version": metadata["schema_version"],
                "probe": metadata["probe"],
                "threshold_rule": metadata["threshold_rule"],
            },
        },
        model_output,
    )
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
                "probe": metadata["probe"],
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


def _artifact_hash(path: str | Path | None) -> dict[str, str | None]:
    if path is None:
        return {"path": None, "sha256": None}
    artifact_path = Path(path)
    return {
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path) if artifact_path.exists() else None,
    }


def _metric_dict(metric: BinaryMetrics) -> dict[str, Any]:
    row = asdict(metric)
    for key, value in row.items():
        if isinstance(value, np.floating):
            row[key] = float(value)
    return row


def _counts_for_records(records: list[ActivationCacheRecord]) -> dict[str, Any]:
    return {
        "records": len(records),
        "by_label": {
            str(label): count
            for label, count in sorted(
                Counter(record.label for record in records).items()
            )
        },
        "by_data_type": dict(
            sorted(Counter(record.data_type for record in records).items())
        ),
    }


def _require_binary_labels(
    records: list[ActivationCacheRecord],
    *,
    split: str,
) -> None:
    if not records:
        raise ValueError(f"No activation records found for probe split {split!r}")
    labels = {record.label for record in records}
    if labels != {0, 1}:
        raise ValueError(
            f"Probe split {split!r} must contain both labels, got {sorted(labels)}"
        )
