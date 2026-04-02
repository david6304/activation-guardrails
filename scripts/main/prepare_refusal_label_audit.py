"""Prepare a manual audit sample from judged refusal labels.

Reads the full labelled response file from ``artifacts/`` and writes a
reviewable CSV sample under ``results/`` so it is not hidden by gitignore.

Sampling policy:
- prioritise unexpected cells first:
  - harmful prompt judged as compliance
  - benign prompt judged as refusal
- fill remaining budget with expected cells:
  - harmful prompt judged as refusal
  - benign prompt judged as compliance

This keeps the manual audit focused on likely judge mistakes while still
checking a slice of ordinary cases.

Usage:
    python scripts/main/prepare_refusal_label_audit.py

    python scripts/main/prepare_refusal_label_audit.py \
        --labelled-responses artifacts/responses/refusal/labelled_responses.jsonl \
        --output results/refusal/validation/label_audit_sample.csv \
        --sample-size 100
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter
from pathlib import Path

from agguardrails.io import read_jsonl, save_metadata


UNEXPECTED_CELLS = (("harmful", 0), ("benign", 1))
EXPECTED_CELLS = (("harmful", 1), ("benign", 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a manual-audit sample from judged refusal labels."
    )
    parser.add_argument(
        "--labelled-responses",
        default="artifacts/responses/refusal/labelled_responses.jsonl",
        help="Full judged response JSONL from label_refusals.py.",
    )
    parser.add_argument(
        "--output",
        default="results/refusal/validation/label_audit_sample.csv",
        help="CSV output path for manual review.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Total number of examples to sample for manual audit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling.",
    )
    return parser.parse_args()


def _sample_from_cells(
    pools: dict[tuple[str, int], list[dict]],
    cells: tuple[tuple[str, int], ...],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Sample up to *n* records across *cells*, as evenly as possible."""
    if n <= 0:
        return []

    available = {cell: list(pools.get(cell, [])) for cell in cells}
    for records in available.values():
        rng.shuffle(records)

    selected: list[dict] = []
    while len(selected) < n:
        progressed = False
        for cell in cells:
            records = available[cell]
            if records and len(selected) < n:
                selected.append(records.pop())
                progressed = True
        if not progressed:
            break
    return selected


def build_audit_sample(
    records: list[dict],
    *,
    sample_size: int,
    seed: int,
) -> tuple[list[dict], dict[str, object]]:
    """Return sampled records plus a small summary dict."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    rng = random.Random(seed)
    pools: dict[tuple[str, int], list[dict]] = {
        cell: [] for cell in (*UNEXPECTED_CELLS, *EXPECTED_CELLS)
    }
    for record in records:
        cell = (str(record["source_label"]), int(record["refusal_label"]))
        pools.setdefault(cell, []).append(record)

    unexpected_total = sum(len(pools[cell]) for cell in UNEXPECTED_CELLS)
    unexpected_budget = min(sample_size // 2, unexpected_total)
    expected_budget = sample_size - unexpected_budget

    sample: list[dict] = []
    sample.extend(_sample_from_cells(pools, UNEXPECTED_CELLS, unexpected_budget, rng))
    sample.extend(_sample_from_cells(pools, EXPECTED_CELLS, expected_budget, rng))

    if len(sample) < min(sample_size, len(records)):
        remaining = [
            record
            for cell_records in pools.values()
            for record in cell_records
        ]
        rng.shuffle(remaining)
        sample.extend(remaining[: min(sample_size, len(records)) - len(sample)])

    for record in sample:
        record["audit_priority"] = (
            "unexpected"
            if (record["source_label"], int(record["refusal_label"])) in UNEXPECTED_CELLS
            else "expected"
        )
        record["manual_label"] = ""
        record["manual_notes"] = ""

    sampled_counts = Counter(
        (record["source_label"], int(record["refusal_label"])) for record in sample
    )
    full_counts = Counter(
        (record["source_label"], int(record["refusal_label"])) for record in records
    )

    summary = {
        "n_total_records": len(records),
        "sample_size": len(sample),
        "requested_sample_size": sample_size,
        "seed": seed,
        "full_counts": {f"{source}:{label}": count for (source, label), count in sorted(full_counts.items())},
        "sample_counts": {f"{source}:{label}": count for (source, label), count in sorted(sampled_counts.items())},
    }
    return sample, summary


def write_audit_csv(path: Path, records: list[dict]) -> None:
    """Write sampled audit records to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "example_id",
        "source",
        "source_label",
        "refusal_label",
        "audit_priority",
        "prompt",
        "response",
        "judge_output",
        "manual_label",
        "manual_notes",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def main() -> None:
    args = parse_args()

    records = read_jsonl(args.labelled_responses)
    sample, summary = build_audit_sample(
        records,
        sample_size=int(args.sample_size),
        seed=int(args.seed),
    )

    output_path = Path(args.output)
    write_audit_csv(output_path, sample)
    metadata_path = save_metadata(
        output_path.with_suffix(".metadata.json"),
        labelled_responses_path=args.labelled_responses,
        output_path=str(output_path),
        **summary,
    )

    print(f"Audit sample written to: {output_path}")
    print(f"Summary metadata written to: {metadata_path}")
    print(f"Sampled {summary['sample_size']} of {summary['n_total_records']} labelled responses")
    print("Cell counts in audit sample:")
    for cell, count in summary["sample_counts"].items():
        print(f"  {cell}: {count}")


if __name__ == "__main__":
    main()
