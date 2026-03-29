"""Train one logistic activation probe per layer and select the best layer."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agguardrails.eval import format_results_row
from agguardrails.features import load_activation_split
from agguardrails.io import save_artifact, save_metadata
from agguardrails.probes import fit_probe_for_layer, select_best_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp/mvp.yaml")
    parser.add_argument("--input-dir", default="artifacts/activations/mvp")
    parser.add_argument("--output-dir", default="artifacts/models/activation_probes")
    parser.add_argument("--metrics-dir", default="results/metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    layers = [int(layer) for layer in config["features"]["layers"]]
    train = load_activation_split(input_dir=args.input_dir, split="train", layers=layers)
    val = load_activation_split(input_dir=args.input_dir, split="val", layers=layers)
    test = load_activation_split(input_dir=args.input_dir, split="test", layers=layers)

    probe_cfg = config["probe"]
    eval_cfg = config["eval"]

    probe_results = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        result = fit_probe_for_layer(
            layer=layer,
            x_train=train.features_by_layer[layer],
            y_train=train.labels,
            x_val=val.features_by_layer[layer],
            y_val=val.labels,
            x_test=test.features_by_layer[layer],
            y_test=test.labels,
            c=float(probe_cfg["C"]),
            max_iter=int(probe_cfg["max_iter"]),
            target_fpr=float(eval_cfg["target_fpr"]),
            random_state=int(config["seed"]),
        )
        probe_results.append(result)
        save_artifact(result.probe, output_dir / f"layer_{layer}_probe")

    best = select_best_probe(probe_results)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(
        metrics_dir / "probe_metrics.json",
        config_path=args.config,
        input_dir=args.input_dir,
        best_layer=best.layer,
        per_layer={
            str(result.layer): {
                "val": format_results_row(
                    model_name="activation_probe",
                    split="val",
                    result=result.val_result,
                    metadata={"layer": result.layer},
                ),
                "test": format_results_row(
                    model_name="activation_probe",
                    split="test",
                    result=result.test_result,
                    metadata={"layer": result.layer},
                ),
            }
            for result in probe_results
        },
    )

    print(f"Best layer: {best.layer}")
    print(f"Validation TPR@FPR: {best.val_result.tpr_at_threshold:.4f}")
    print(f"Test TPR@val-threshold: {best.test_result.tpr_at_threshold:.4f}")


if __name__ == "__main__":
    main()
