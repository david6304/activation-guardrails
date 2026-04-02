"""Assemble plain-text and cipher-transfer metrics into one CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agguardrails.io import save_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a combined plain-text and cipher-transfer results table."
    )
    parser.add_argument(
        "--text-metrics",
        default="results/refusal/metrics/text_baseline_metrics.json",
    )
    parser.add_argument(
        "--probe-metrics",
        default="results/refusal/metrics/probe_metrics.json",
    )
    parser.add_argument(
        "--sae-probe-metrics",
        default="results/refusal/metrics/sae_probe_metrics.json",
    )
    parser.add_argument(
        "--transfer-metrics-dir",
        default="results/refusal/cipher_transfer",
        help="Directory searched recursively for *_transfer_metrics.json files.",
    )
    parser.add_argument(
        "--output",
        default="results/refusal/cipher_transfer/results_table.csv",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def plain_rows(path: str | Path) -> list[dict[str, str | int | float | None]]:
    metrics_path = Path(path)
    if not metrics_path.exists():
        return []

    metrics = load_json(metrics_path)
    rows = []
    if "val_metrics" in metrics:
        rows.append(metrics["val_metrics"])
    if "test_metrics" in metrics:
        rows.append(metrics["test_metrics"])
    else:
        best_layer = str(metrics["best_layer"])
        per_split = metrics["per_layer"][best_layer]
        rows.extend(per_split[split] for split in ("val", "test") if split in per_split)
    return rows


def transfer_rows(path: Path) -> list[dict[str, str | int | float | None]]:
    rows: list[dict[str, str | int | float | None]] = []
    for metrics_path in sorted(path.rglob("*_transfer_metrics.json")):
        metrics = load_json(metrics_path)
        row = dict(metrics["transfer_metrics"])
        row["metrics_path"] = str(metrics_path)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()

    rows = [
        *plain_rows(args.text_metrics),
        *plain_rows(args.probe_metrics),
        *plain_rows(args.sae_probe_metrics),
        *transfer_rows(Path(args.transfer_metrics_dir)),
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
        text_metrics_path=args.text_metrics,
        probe_metrics_path=args.probe_metrics,
        sae_probe_metrics_path=args.sae_probe_metrics,
        transfer_metrics_dir=args.transfer_metrics_dir,
        n_rows=len(rows),
    )

    print(f"Wrote cipher-transfer table to {output_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
