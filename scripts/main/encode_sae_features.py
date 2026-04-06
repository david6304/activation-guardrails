"""Encode cached hidden states into SAE feature vectors for the main experiment.

Usage:
    python scripts/main/encode_sae_features.py --config configs/main/main.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml

from agguardrails.features import load_activation_split
from agguardrails.io import save_artifact, save_metadata
from agguardrails.sae import (
    build_pretrained_sae_spec,
    encode_with_sae,
    load_pretrained_sae,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode cached hidden states with pretrained SAEs."
    )
    parser.add_argument("--config", default="configs/main/main.yaml")
    parser.add_argument("--input-dir", default="artifacts/activations/main")
    parser.add_argument("--output-dir", default="artifacts/features/main/sae")
    parser.add_argument(
        "--metrics-dir",
        default="results/main/metrics",
        help="Directory for small tracked SAE encoding summaries.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        metavar="SPLIT",
        help="Splits to encode. Default: train val test, plus adversarial if present.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="SAE encoding batch size.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device passed to SAELens when loading the SAE.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Torch dtype string passed to SAELens when loading the SAE.",
    )
    return parser.parse_args()


def _default_splits(input_dir: Path) -> list[str]:
    splits = ["train", "val", "test"]
    if (input_dir / "adversarial_labels.npz").exists():
        splits.append("adversarial")
    return splits


def main() -> None:
    args = parse_args()

    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    sae_cfg = config["sae"]
    layers = [int(layer) for layer in sae_cfg["layers"]]
    variant = sae_cfg.get("variant", "average_l0_14")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    splits = args.splits or _default_splits(input_dir)

    specs = [
        build_pretrained_sae_spec(
            release=sae_cfg["release"],
            layer=layer,
            width=int(sae_cfg["width"]),
            variant=variant,
        )
        for layer in layers
    ]
    saes = {}
    for spec in specs:
        print(
            "Loading SAE for layer "
            f"{spec.layer}: release={spec.release} sae_id={spec.sae_id}",
            flush=True,
        )
        saes[spec.layer] = load_pretrained_sae(
            release=spec.release,
            sae_id=spec.sae_id,
            device=args.device,
            dtype=args.dtype,
        )

    run_summary: dict[str, object] = {
        "config_path": args.config,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "release": sae_cfg["release"],
        "width": int(sae_cfg["width"]),
        "variant": variant,
        "layers": layers,
        "batch_size": int(args.batch_size),
        "device": args.device,
        "dtype": args.dtype,
        "splits": {},
    }

    for split in splits:
        dataset = load_activation_split(input_dir=input_dir, split=split, layers=layers)
        feature_paths: dict[str, str] = {}
        split_summary: dict[str, object] = {
            "n_examples": len(dataset.example_ids),
            "feature_paths": {},
            "feature_shapes": {},
        }
        for spec in specs:
            print(
                f"Encoding split={split} layer={spec.layer} "
                f"into SAE width {spec.width}",
                flush=True,
            )
            encoded = encode_with_sae(
                sae=saes[spec.layer],
                activations=dataset.features_by_layer[spec.layer],
                batch_size=args.batch_size,
            )
            path = save_artifact(
                encoded,
                output_dir / f"{split}_layer_{spec.layer}_sae_features",
            )
            feature_paths[str(spec.layer)] = str(path)
            split_summary["feature_paths"][str(spec.layer)] = str(path)
            split_summary["feature_shapes"][str(spec.layer)] = list(encoded.shape)

        label_arrays = {"label": dataset.labels}
        if dataset.label_arrays is not None:
            label_arrays.update(dataset.label_arrays)
        label_paths = {}
        for label_key, label_values in sorted(label_arrays.items()):
            label_stem = (
                f"{split}_labels"
                if label_key == "label"
                else f"{split}_labels_{label_key}"
            )
            label_paths[label_key] = str(
                save_artifact(label_values, output_dir / label_stem)
            )
        ids_path = save_artifact(
            np.array(dataset.example_ids, dtype=str),
            output_dir / f"{split}_ids",
        )
        save_metadata(
            output_dir / f"{split}_metadata.json",
            config_path=args.config,
            input_dir=str(input_dir),
            split=split,
            release=sae_cfg["release"],
            width=int(sae_cfg["width"]),
            variant=variant,
            layers=layers,
            batch_size=int(args.batch_size),
            device=args.device,
            dtype=args.dtype,
            label_keys=sorted(label_arrays.keys()),
            labels_path=label_paths["label"],
            label_paths=label_paths,
            ids_path=str(ids_path),
            feature_paths=feature_paths,
        )
        split_summary["label_paths"] = label_paths
        split_summary["ids_path"] = str(ids_path)
        run_summary["splits"][split] = split_summary
        print(f"Saved SAE features for split={split}", flush=True)

    summary_path = save_metadata(
        metrics_dir / "sae_encoding_summary.json",
        **run_summary,
    )
    print(f"SAE encoding summary saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
