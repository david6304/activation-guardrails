"""Assemble a one-table main-experiment comparison from saved metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agguardrails.io import save_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main/main.yaml")
    parser.add_argument(
        "--text-metrics",
        default="results/main/metrics/text_baseline_metrics.json",
    )
    parser.add_argument(
        "--probe-metrics",
        default="results/main/metrics/probe_metrics.json",
    )
    parser.add_argument(
        "--sae-probe-metrics",
        default="results/main/metrics/sae_probe_metrics.json",
    )
    parser.add_argument(
        "--output",
        default="results/main/results_table.csv",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def maybe_text_rows(path: str | Path) -> list[dict[str, str | int | float | None]]:
    metrics_path = Path(path)
    if not metrics_path.exists():
        return []

    metrics = load_json(metrics_path)
    rows = [metrics["val_metrics"], metrics["test_metrics"]]
    if "adversarial_metrics" in metrics:
        rows.append(metrics["adversarial_metrics"])
    return rows


def maybe_best_layer_rows(path: str | Path) -> list[dict[str, str | int | float | None]]:
    metrics_path = Path(path)
    if not metrics_path.exists():
        return []

    metrics = load_json(metrics_path)
    best_layer = str(metrics["best_layer"])
    per_split = metrics["per_layer"][best_layer]
    ordered_splits = ["val", "test", "adversarial"]
    return [per_split[split] for split in ordered_splits if split in per_split]


def main() -> None:
    args = parse_args()
    rows = [
        *maybe_text_rows(args.text_metrics),
        *maybe_best_layer_rows(args.probe_metrics),
        *maybe_best_layer_rows(args.sae_probe_metrics),
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    save_metadata(
        output_path.with_suffix(".metadata.json"),
        config_path=args.config,
        text_metrics_path=args.text_metrics,
        probe_metrics_path=args.probe_metrics,
        sae_probe_metrics_path=args.sae_probe_metrics,
        n_rows=len(rows),
    )

    print(f"Wrote results table to {output_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
