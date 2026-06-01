#!/usr/bin/env python
"""Train and evaluate the CC++ SWiM linear activation probe."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from agguardrails.activations import load_activation_cache
from agguardrails.metrics import (
    apply_threshold,
    fixed_fpr_point,
    roc_auc,
    wilson_interval,
)
from agguardrails.swim_probe import (
    SwimTrainingConfig,
    score_activation_examples,
    train_linear_swim_probe,
)


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    examples, activation_metadata = load_activation_cache(
        npz_path=args.activation_npz,
        index_path=args.activation_index,
    )
    train_examples = apply_training_sampling(
        [example for example in examples if example.split == "train"],
        config=config,
    )
    if not train_examples:
        raise ValueError("activation cache contains no train examples")

    model, losses = train_linear_swim_probe(
        train_examples,
        config=training_config(config),
    )
    score_rows = score_activation_examples(
        model,
        examples,
        ema_gamma=config["swim_probe"].get("ema_gamma"),
    )
    metrics = evaluate_scores(score_rows, config=config)
    metrics["training"] = {
        "loss_count": len(losses),
        "final_loss": losses[-1] if losses else None,
    }
    metrics["provenance"] = {
        "git_commit": current_git_commit(),
        "config_path": str(args.config),
        "activation_metadata": activation_metadata,
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feature_dim": next(model.parameters()).shape[1],
            "config": config["swim_probe"],
        },
        args.model_output,
    )
    write_jsonl(args.scores_output, score_rows)
    write_json(args.metrics_output, metrics)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ccpp/gemma3_4b_it_public_cbrn_probe.yaml"),
    )
    parser.add_argument("--activation-npz", type=Path, default=None)
    parser.add_argument("--activation-index", type=Path, default=None)
    parser.add_argument("--model-output", type=Path, default=None)
    parser.add_argument("--scores-output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    args = parser.parse_args()

    config = load_yaml(args.config)
    activation_dir = Path(config["activation"]["artifact_dir"])
    probe_dir = Path(config["swim_probe"]["artifact_dir"])
    args.activation_npz = args.activation_npz or activation_dir / "activations.npz"
    args.activation_index = (
        args.activation_index or activation_dir / "activations.index.jsonl"
    )
    args.model_output = args.model_output or probe_dir / "linear_swim_probe.pt"
    args.scores_output = args.scores_output or Path(config["swim_probe"]["scores_path"])
    args.metrics_output = args.metrics_output or Path(
        config["swim_probe"]["metrics_path"]
    )
    return args


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return loaded


def training_config(config: dict[str, Any]) -> SwimTrainingConfig:
    probe_config = config["swim_probe"]
    train_config = probe_config["train"]
    return SwimTrainingConfig(
        seed=int(probe_config["seed"]),
        epochs=int(train_config["epochs"]),
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        batch_size=int(train_config["batch_size"]),
        window_size=int(probe_config["window_size"]),
        softmax_temperature=float(probe_config["softmax_temperature"]),
    )


def apply_training_sampling(examples, *, config: dict[str, Any]):
    sampling = config["swim_probe"]["train"].get("sampling", {})
    if sampling.get("strategy") != "balanced_by_label":
        return examples
    max_ratio = float(sampling["max_negative_to_positive_ratio"])
    positives = [example for example in examples if example.label == 1]
    negatives = [example for example in examples if example.label == 0]
    if not positives or not negatives:
        return examples
    max_negatives = max(1, int(len(positives) * max_ratio))
    if len(negatives) <= max_negatives:
        return examples
    return positives + negatives[:max_negatives]


def evaluate_scores(
    rows: list[dict[str, float | int | str]],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    row_by_split = split_rows(rows)
    evaluation_config = config["swim_probe"].get("evaluation", {})
    group_rows = aggregate_group_scores(
        rows,
        aggregation=evaluation_config.get(
            "group_score_aggregation",
            "mean",
        ),
    )
    group_by_split = split_rows(group_rows)
    metrics: dict[str, Any] = {
        "row_level": {"splits": split_metrics(row_by_split)},
        "group_level": {"splits": split_metrics(group_by_split)},
        "headline_level": "group_level",
    }
    metrics["splits"] = metrics["group_level"]["splits"]

    threshold_config = config["swim_probe"]["threshold"]
    threshold_metrics: dict[str, Any] = {}
    for name, max_fpr in {
        "primary": float(threshold_config["primary_max_fpr"]),
        "secondary": float(threshold_config["secondary_max_fpr"]),
    }.items():
        threshold_metrics[name] = threshold_report(
            group_by_split.get("val", []),
            group_by_split,
            max_fpr=max_fpr,
        )
    metrics["thresholds"] = threshold_metrics
    return metrics


def split_rows(
    rows: list[dict[str, float | int | str]],
) -> dict[str, list[dict[str, float | int | str]]]:
    return {
        split: [row for row in rows if row["split"] == split]
        for split in sorted({str(row["split"]) for row in rows})
    }


def split_metrics(
    by_split: dict[str, list[dict[str, float | int | str]]],
) -> dict[str, Any]:
    metrics = {}
    for split, split_rows in by_split.items():
        labels = [int(row["label"]) for row in split_rows]
        scores = [float(row["score"]) for row in split_rows]
        split_metrics: dict[str, Any] = {"count": len(split_rows)}
        split_metrics["positive_count"] = sum(label == 1 for label in labels)
        split_metrics["negative_count"] = sum(label == 0 for label in labels)
        if len(set(labels)) == 2:
            split_metrics["roc_auc"] = roc_auc(labels, scores)
        metrics[split] = split_metrics
    return metrics


def aggregate_group_scores(
    rows: list[dict[str, float | int | str]],
    *,
    aggregation: str,
) -> list[dict[str, float | int | str]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    for row in rows:
        group_id = str(row.get("group_id") or row["example_id"])
        grouped.setdefault(group_id, []).append(row)

    group_rows = []
    for group_id, group in grouped.items():
        labels = {int(row["label"]) for row in group}
        splits = {str(row["split"]) for row in group}
        if len(labels) != 1:
            raise ValueError(f"group {group_id} contains multiple labels")
        if len(splits) != 1:
            raise ValueError(f"group {group_id} appears in multiple splits")
        scores = [float(row["score"]) for row in group]
        if aggregation == "mean":
            score = sum(scores) / len(scores)
        elif aggregation == "max":
            score = max(scores)
        else:
            raise ValueError(f"unsupported group_score_aggregation: {aggregation}")
        group_rows.append(
            {
                "example_id": group_id,
                "group_id": group_id,
                "split": next(iter(splits)),
                "label": next(iter(labels)),
                "score": score,
                "row_count": len(group),
            }
        )
    return group_rows


def threshold_report(
    val_rows: list[dict[str, float | int | str]],
    by_split: dict[str, list[dict[str, float | int | str]]],
    *,
    max_fpr: float,
) -> dict[str, Any]:
    if not val_rows or len({int(row["label"]) for row in val_rows}) < 2:
        return {"max_fpr": max_fpr, "status": "missing_binary_validation_split"}

    validation = fixed_fpr_point(
        [int(row["label"]) for row in val_rows],
        [float(row["score"]) for row in val_rows],
        max_fpr=max_fpr,
    )
    negative_count = validation.negative_count
    minimum_negative_count = math.ceil(1 / max_fpr) if max_fpr > 0 else None
    fpr_warning = None
    if minimum_negative_count is not None and negative_count < minimum_negative_count:
        fpr_warning = (
            "validation negatives are too few to resolve this FPR with one "
            "false positive"
        )
    report: dict[str, Any] = {
        "max_fpr": max_fpr,
        "threshold": validation.threshold,
        "validation": validation.__dict__,
        "minimum_negative_count_for_one_false_positive": minimum_negative_count,
        "fpr_resolution_warning": fpr_warning,
        "splits": {},
    }
    for split, rows in by_split.items():
        if len({int(row["label"]) for row in rows}) < 2:
            continue
        point = apply_threshold(
            [int(row["label"]) for row in rows],
            [float(row["score"]) for row in rows],
            threshold=validation.threshold,
            max_fpr=max_fpr,
        )
        split_report = dict(point.__dict__)
        split_report["tpr_wilson_95"] = wilson_interval(
            count=point.true_positives,
            total=point.positive_count,
        ).__dict__
        split_report["fpr_wilson_95"] = wilson_interval(
            count=point.false_positives,
            total=point.negative_count,
        ).__dict__
        report["splits"][split] = split_report
    return report


def write_jsonl(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def current_git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
