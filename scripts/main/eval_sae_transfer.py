"""Evaluate a frozen SAE probe on cached transfer SAE features."""

from __future__ import annotations

import argparse
from pathlib import Path

from agguardrails.features import load_layer_feature_split
from agguardrails.io import load_artifact, save_metadata
from agguardrails.transfer import load_json, summarize_feature_transfer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen SAE probe on transfer SAE features."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory with cached transfer SAE features.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split name to evaluate from the SAE feature cache.",
    )
    parser.add_argument(
        "--probe-dir",
        default="artifacts/models/refusal/sae_probes",
        help="Directory with fitted plain-text SAE probes.",
    )
    parser.add_argument(
        "--plain-metrics",
        default="results/refusal/metrics/sae_probe_metrics.json",
        help="Plain-text SAE metrics JSON used to recover best layer and threshold.",
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

    plain_metrics = load_json(args.plain_metrics)
    best_layer = int(plain_metrics["best_layer"])
    threshold = float(plain_metrics["per_layer"][str(best_layer)]["val"]["threshold"])

    dataset = load_layer_feature_split(
        input_dir=args.input_dir,
        split=args.split,
        layers=[best_layer],
        feature_name="sae_features",
    )
    probe = load_artifact(Path(args.probe_dir) / f"layer_{best_layer}_sae_probe.joblib")
    transfer_metrics = summarize_feature_transfer(
        labels=dataset.labels,
        features=dataset.features_by_layer[best_layer],
        classifier=probe,
        threshold=threshold,
    )
    transfer_metrics.update(
        {
            "model_name": "sae_probe",
            "split": args.eval_name,
            "layer": best_layer,
        }
    )

    output_path = save_metadata(
        args.output,
        input_dir=args.input_dir,
        split=args.split,
        probe_dir=args.probe_dir,
        plain_metrics_path=args.plain_metrics,
        transfer_metrics=transfer_metrics,
    )
    print(f"Transfer metrics saved: {output_path}")


if __name__ == "__main__":
    main()
