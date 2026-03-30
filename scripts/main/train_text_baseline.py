"""Train the TF-IDF + LogisticRegression baseline on the main experiment dataset.

Usage:
    python scripts/main/train_text_baseline.py --config configs/main/main.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agguardrails.baselines import examples_to_text_and_labels, fit_text_baseline
from agguardrails.data import read_prompt_dataset, split_prompt_dataset
from agguardrails.eval import format_results_row, summarize_scores_at_threshold
from agguardrails.io import save_artifact, save_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + LR text baseline on the main experiment dataset."
    )
    parser.add_argument("--config", default="configs/main/main.yaml")
    parser.add_argument("--dataset", default="data/processed/main_prompts.jsonl")
    parser.add_argument(
        "--adversarial-dataset",
        default=None,
        help=(
            "Optional adversarial harmful-only JSONL. If provided, report recall "
            "at the validation-selected threshold."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/main/text_baseline",
        help="Directory for saved pipeline artifact.",
    )
    parser.add_argument(
        "--metrics-dir",
        default="results/main/metrics",
        help="Directory for committed metrics JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    dataset = read_prompt_dataset(args.dataset)
    splits = split_prompt_dataset(dataset)

    baseline_cfg = config["baseline"]
    eval_cfg = config["eval"]

    artifacts = fit_text_baseline(
        train_examples=splits["train"],
        val_examples=splits["val"],
        test_examples=splits["test"],
        max_features=int(baseline_cfg["max_features"]),
        c=float(baseline_cfg["C"]),
        target_fpr=float(eval_cfg["target_fpr"]),
        random_state=int(config["seed"]),
    )

    output_dir = Path(args.output_dir)
    pipeline_path = save_artifact(artifacts.pipeline, output_dir / "tfidf_lr_pipeline")

    val_row = format_results_row(
        model_name="tfidf_lr",
        split="val",
        result=artifacts.val_result,
    )
    test_row = format_results_row(
        model_name="tfidf_lr",
        split="test",
        result=artifacts.test_result,
    )
    adversarial_row = None
    if args.adversarial_dataset is not None:
        adversarial = read_prompt_dataset(args.adversarial_dataset)
        adv_texts, y_adv = examples_to_text_and_labels(adversarial)
        adv_scores = artifacts.pipeline.predict_proba(adv_texts)[:, 1]
        adversarial_row = {
            "model_name": "tfidf_lr",
            "split": "adversarial",
            **summarize_scores_at_threshold(
                y_adv,
                adv_scores,
                threshold=artifacts.val_result.threshold,
            ),
        }

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metadata_kwargs = {
        "config_path": args.config,
        "dataset_path": args.dataset,
        "pipeline_path": str(pipeline_path),
        "val_metrics": val_row,
        "test_metrics": test_row,
    }
    if args.adversarial_dataset is not None:
        metadata_kwargs["adversarial_dataset_path"] = args.adversarial_dataset
        metadata_kwargs["adversarial_metrics"] = adversarial_row
    metadata_path = save_metadata(
        metrics_dir / "text_baseline_metrics.json",
        **metadata_kwargs,
    )

    print(f"Pipeline saved: {pipeline_path}")
    print(f"Metrics saved:  {metadata_path}")
    print(f"Val  ROC-AUC: {artifacts.val_result.roc_auc:.4f}  TPR@1%FPR: {artifacts.val_result.tpr_at_threshold:.4f}")
    print(f"Test ROC-AUC: {artifacts.test_result.roc_auc:.4f}  TPR@1%FPR: {artifacts.test_result.tpr_at_threshold:.4f}")
    if adversarial_row is not None:
        print(
            "Adv  Recall@val-threshold:"
            f" {adversarial_row['tpr_at_threshold']:.4f}"
            f"  Positives: {adversarial_row['positive_predictions']}/{adversarial_row['n_examples']}"
        )


if __name__ == "__main__":
    main()
