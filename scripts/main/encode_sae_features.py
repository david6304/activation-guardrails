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
from agguardrails.sae import build_pretrained_sae_spec, encode_with_sae, load_pretrained_sae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode cached hidden states with pretrained SAEs."
    )
    parser.add_argument("--config", default="configs/main/main.yaml")
    parser.add_argument("--input-dir", default="artifacts/activations/main")
    parser.add_argument("--output-dir", default="artifacts/features/main/sae")
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
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = args.splits or _default_splits(input_dir)

    specs = [
        build_pretrained_sae_spec(
            release=sae_cfg["release"],
            layer=layer,
            width=int(sae_cfg["width"]),
        )
        for layer in layers
    ]
    saes = {}
    for spec in specs:
        print(
            f"Loading SAE for layer {spec.layer}: release={spec.release} sae_id={spec.sae_id}",
            flush=True,
        )
        saes[spec.layer] = load_pretrained_sae(
            release=spec.release,
            sae_id=spec.sae_id,
            device=args.device,
            dtype=args.dtype,
        )

    for split in splits:
        dataset = load_activation_split(input_dir=input_dir, split=split, layers=layers)
        feature_paths: dict[str, str] = {}
        for spec in specs:
            print(
                f"Encoding split={split} layer={spec.layer} into SAE width {spec.width}",
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

        labels_path = save_artifact(dataset.labels, output_dir / f"{split}_labels")
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
            layers=layers,
            batch_size=int(args.batch_size),
            device=args.device,
            dtype=args.dtype,
            labels_path=str(labels_path),
            ids_path=str(ids_path),
            feature_paths=feature_paths,
        )
        print(f"Saved SAE features for split={split}", flush=True)


if __name__ == "__main__":
    main()
