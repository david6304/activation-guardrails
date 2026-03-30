"""Prefetch configured pretrained SAEs into the local Hugging Face cache.

Usage:
    python scripts/main/cache_sae_models.py --config configs/main/main.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from agguardrails.sae import build_pretrained_sae_spec, load_pretrained_sae


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download configured pretrained SAEs into the local cache."
    )
    parser.add_argument("--config", default="configs/main/main.yaml")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device passed to SAELens while prefetching.",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Torch dtype string passed to SAELens while prefetching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    sae_cfg = config["sae"]
    layers = [int(layer) for layer in sae_cfg["layers"]]
    variant = sae_cfg.get("variant", "average_l0_14")

    specs = [
        build_pretrained_sae_spec(
            release=sae_cfg["release"],
            layer=layer,
            width=int(sae_cfg["width"]),
            variant=variant,
        )
        for layer in layers
    ]

    for spec in specs:
        print(
            f"Caching SAE for layer {spec.layer}: release={spec.release} sae_id={spec.sae_id}",
            flush=True,
        )
        load_pretrained_sae(
            release=spec.release,
            sae_id=spec.sae_id,
            device=args.device,
            dtype=args.dtype,
        )

    print(f"Cached {len(specs)} SAE(s) for release={sae_cfg['release']}", flush=True)


if __name__ == "__main__":
    main()
