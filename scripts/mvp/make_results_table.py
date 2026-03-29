"""Assemble a one-table MVP comparison from saved baseline and probe metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from agguardrails.io import save_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvp/mvp.yaml")
    parser.add_argument(
        "--text-metrics",
        default="results/metrics/text_baseline_metrics.json",
    )
    parser.add_argument(
        "--probe-metrics",
        default="results/metrics/probe_metrics.json",
    )
    parser.add_argument(
        "--output",
        default="results/mvp_results.csv",
    )
    return parser.parse_args()


def load_json(path: str | Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def maybe_probe_test_row(path: str | Path) -> dict[str, str | int | float] | None:
    metrics_path = Path(path)
    if not metrics_path.exists():
        return None

    metrics = load_json(metrics_path)
    best_layer = str(metrics["best_layer"])
    return metrics["per_layer"][best_layer]["test"]


def main() -> None:
    args = parse_args()
    text_metrics = load_json(args.text_metrics)
    rows = [text_metrics["test_metrics"]]

    probe_row = maybe_probe_test_row(args.probe_metrics)
    if probe_row is not None:
        rows.append(probe_row)

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
        n_rows=len(rows),
    )

    print(f"Wrote results table to {output_path}")
    print(f"Rows: {len(rows)}")


if __name__ == "__main__":
    main()
