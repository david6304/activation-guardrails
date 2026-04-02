"""Evaluate a frozen text baseline on a transfer dataset without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path

from agguardrails.data import read_prompt_dataset
from agguardrails.io import load_artifact, save_metadata
from agguardrails.transfer import load_json, summarize_text_transfer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen text baseline on a transfer dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Transfer evaluation dataset JSONL.",
    )
    parser.add_argument(
        "--pipeline-path",
        default="artifacts/models/refusal/text_baseline/tfidf_lr_pipeline.joblib",
        help="Saved TF-IDF + LR pipeline.",
    )
    parser.add_argument(
        "--plain-metrics",
        default="results/refusal/metrics/text_baseline_metrics.json",
        help="Plain-text text-baseline metrics JSON used to recover the val threshold.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output metrics JSON path.",
    )
    parser.add_argument(
        "--eval-name",
        default="transfer",
        help="Name recorded in the output row, for example rot13 or reverse.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = read_prompt_dataset(args.dataset)
    pipeline = load_artifact(args.pipeline_path)
    plain_metrics = load_json(args.plain_metrics)
    threshold = float(plain_metrics["val_metrics"]["threshold"])

    transfer_metrics = summarize_text_transfer(
        examples=dataset,
        pipeline=pipeline,
        threshold=threshold,
    )
    transfer_metrics.update(
        {
            "model_name": "tfidf_lr",
            "split": args.eval_name,
        }
    )

    output_path = save_metadata(
        args.output,
        dataset_path=args.dataset,
        pipeline_path=args.pipeline_path,
        plain_metrics_path=args.plain_metrics,
        transfer_metrics=transfer_metrics,
    )
    print(f"Transfer metrics saved: {output_path}")


if __name__ == "__main__":
    main()
