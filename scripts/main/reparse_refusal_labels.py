"""Repair refusal labels by reparsing existing judge outputs.

This is a cheap recovery path when the raw judge generations already exist but
the label parser has changed or was previously buggy. No model inference is
performed: the script reads the existing JSONL, recomputes ``refusal_label``
from ``judge_output``, and writes a corrected JSONL plus metadata summary.

Usage:
    python scripts/main/reparse_refusal_labels.py

    python scripts/main/reparse_refusal_labels.py \
        --input artifacts/responses/refusal/labelled_responses.jsonl \
        --output artifacts/responses/refusal/labelled_responses_reparsed.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from agguardrails.io import read_jsonl, save_metadata, write_jsonl
from agguardrails.judge import _parse_refusal_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute refusal labels from existing judge outputs."
    )
    parser.add_argument(
        "--input",
        default="artifacts/responses/refusal/labelled_responses.jsonl",
        help="Existing JSONL with judge_output fields.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/responses/refusal/labelled_responses_reparsed.jsonl",
        help="Corrected JSONL path.",
    )
    parser.add_argument(
        "--fallback-label",
        type=int,
        default=0,
        choices=[0, 1],
        help="Label used when judge_output is still unparseable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = read_jsonl(args.input)
    repaired = []
    n_changed = 0
    n_unparseable = 0

    for record in records:
        parsed = _parse_refusal_label(str(record.get("judge_output", "")))
        if parsed is None:
            n_unparseable += 1
            parsed = int(args.fallback_label)
        old_label = int(record.get("refusal_label", parsed))
        if parsed != old_label:
            n_changed += 1
        repaired.append({**record, "refusal_label": parsed})

    output_path = write_jsonl(args.output, repaired)
    metadata_path = save_metadata(
        Path(args.output).with_suffix(".metadata.json"),
        input_path=args.input,
        output_path=str(output_path),
        n_records=len(records),
        n_changed=n_changed,
        n_unparseable=n_unparseable,
        fallback_label=int(args.fallback_label),
    )

    print(f"Corrected labels written to: {output_path}")
    print(f"Metadata written to:        {metadata_path}")
    print(f"Records changed:            {n_changed}/{len(records)}")
    if n_unparseable:
        print(f"Unparseable judge outputs:  {n_unparseable}")


if __name__ == "__main__":
    main()
