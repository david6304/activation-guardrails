"""Train one logistic activation probe per layer and select the best layer.

Sweeps all layers defined in the config, selects the best layer on val TPR,
and saves per-layer metrics and the fitted probes.

Usage:
    python scripts/main/train_activation_probes.py --config configs/main/main.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agguardrails.eval import format_results_row, summarize_scores_at_threshold
from agguardrails.features import load_activation_split
from agguardrails.io import save_artifact, save_metadata
from agguardrails.probes import fit_probe_for_layer, select_best_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train logistic activation probes for the main experiment."
    )
    parser.add_argument("--config", default="configs/main/main.yaml")
    parser.add_argument("--input-dir", default="artifacts/activations/main")
    parser.add_argument(
        "--output-dir",
        default="artifacts/models/main/activation_probes",
        help="Directory for fitted probe artifacts.",
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

    layers = [int(layer) for layer in config["features"]["layers"]]
    probe_cfg = config["probe"]
    eval_cfg = config["eval"]
    seed = int(config["seed"])

    train = load_activation_split(input_dir=args.input_dir, split="train", layers=layers)
    val = load_activation_split(input_dir=args.input_dir, split="val", layers=layers)
    test = load_activation_split(input_dir=args.input_dir, split="test", layers=layers)
    adversarial = None
    adversarial_labels_path = Path(args.input_dir) / "adversarial_labels.npz"
    if adversarial_labels_path.exists():
        adversarial = load_activation_split(
            input_dir=args.input_dir,
            split="adversarial",
            layers=layers,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_results = []
    per_layer_metrics: dict[str, dict[str, dict[str, str | int | float | None]]] = {}
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
            penalty=str(probe_cfg.get("penalty", "l2")),
            random_state=seed,
        )
        probe_results.append(result)
        save_artifact(result.probe, output_dir / f"layer_{layer}_probe")
        layer_metrics: dict[str, dict[str, str | int | float | None]] = {
            "val": format_results_row(
                model_name="dense_probe",
                split="val",
                result=result.val_result,
                metadata={"layer": result.layer},
            ),
            "test": format_results_row(
                model_name="dense_probe",
                split="test",
                result=result.test_result,
                metadata={"layer": result.layer},
            ),
        }
        if adversarial is not None:
            adv_scores = result.probe.predict_proba(
                adversarial.features_by_layer[layer]
            )[:, 1]
            layer_metrics["adversarial"] = {
                "model_name": "dense_probe",
                "split": "adversarial",
                "layer": result.layer,
                **summarize_scores_at_threshold(
                    adversarial.labels,
                    adv_scores,
                    threshold=result.val_result.threshold,
                ),
            }
        per_layer_metrics[str(result.layer)] = layer_metrics
        print(
            f"Layer {layer:2d}: val ROC-AUC={result.val_result.roc_auc:.4f}"
            f"  val TPR@1%FPR={result.val_result.tpr_at_threshold:.4f}"
        )

    best = select_best_probe(probe_results)

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metadata_kwargs: dict[str, object] = {
        "config_path": args.config,
        "input_dir": args.input_dir,
        "best_layer": best.layer,
        "per_layer": per_layer_metrics,
    }
    if adversarial is not None:
        metadata_kwargs["best_adversarial_metrics"] = per_layer_metrics[str(best.layer)][
            "adversarial"
        ]
    save_metadata(
        metrics_dir / "probe_metrics.json",
        **metadata_kwargs,
    )

    print(f"\nBest layer: {best.layer}")
    print(f"Val  ROC-AUC: {best.val_result.roc_auc:.4f}  TPR@1%FPR: {best.val_result.tpr_at_threshold:.4f}")
    print(f"Test ROC-AUC: {best.test_result.roc_auc:.4f}  TPR@1%FPR: {best.test_result.tpr_at_threshold:.4f}")
    if adversarial is not None:
        best_adv = per_layer_metrics[str(best.layer)]["adversarial"]
        print(
            "Adv  Recall@val-threshold:"
            f" {best_adv['tpr_at_threshold']:.4f}"
            f"  Positives: {best_adv['positive_predictions']}/{best_adv['n_examples']}"
        )


if __name__ == "__main__":
    main()
