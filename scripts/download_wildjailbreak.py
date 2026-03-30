"""Download and normalise the WildJailbreak dataset to local JSONL.

Usage (run once, requires internet access — use cluster head node or local):
    python scripts/download_wildjailbreak.py --output data/raw/wildjailbreak/wildjailbreak.jsonl

This script downloads ``allenai/wildjailbreak`` from HuggingFace, normalises
it into the schema expected by ``agguardrails.data.load_wildjailbreak_examples``,
and saves a single JSONL file with one record per prompt.

Normalised schema (one record per row):
    {"id": str, "prompt": str, "data_type": str, "source": "wildjailbreak"}

data_type values:
    - "vanilla_harmful"      — direct harmful request
    - "vanilla_benign"       — benign prompt surface-similar to harmful one
    - "adversarial_harmful"  — jailbroken version of a vanilla harmful prompt
    - "adversarial_benign"   — jailbreak-styled benign prompt

The current Hugging Face dataset card exposes configs ``train`` and ``eval``
and TSV-backed rows with columns including:
    - "vanilla"
    - "adversarial"
    - "data_type"
    - "completion"
    - "tactics"

Rows are single prompt-response pairs. We therefore choose the prompt field
based on ``data_type``:
    - vanilla_*      -> use the "vanilla" column
    - adversarial_*  -> use the "adversarial" column
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from repo root without installing package
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agguardrails.io import write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download WildJailbreak dataset to local JSONL.")
    parser.add_argument(
        "--output",
        default="data/raw/wildjailbreak/wildjailbreak.jsonl",
        help="Output path for normalised JSONL.",
    )
    parser.add_argument(
        "--hf-dataset",
        default="allenai/wildjailbreak",
        help="HuggingFace dataset identifier.",
    )
    parser.add_argument(
        "--config-name",
        default="train",
        choices=["train", "eval"],
        help="HuggingFace dataset config to download (default: train).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="HuggingFace dataset split to download within the selected config (default: train).",
    )
    parser.add_argument(
        "--print-schema-only",
        action="store_true",
        help="Print dataset schema and first row then exit without writing.",
    )
    return parser.parse_args()


def normalise_record(record: dict, idx: int) -> dict | None:
    """Convert one HuggingFace WildJailbreak row into one normalised record.

    The upstream dataset already carries the canonical ``data_type`` for each
    row, so the main risk is selecting the wrong prompt column when Hugging Face
    schema details drift. This function makes that mapping explicit.
    """
    data_type = str(record.get("data_type", "") or "").strip()
    if not data_type:
        msg = f"WildJailbreak row {idx} is missing required field 'data_type'."
        raise ValueError(msg)

    prompt_column_by_type = {
        "vanilla_harmful": "vanilla",
        "vanilla_benign": "vanilla",
        "adversarial_harmful": "adversarial",
        "adversarial_benign": "adversarial",
    }
    try:
        prompt_column = prompt_column_by_type[data_type]
    except KeyError as exc:
        msg = f"Unsupported WildJailbreak data_type {data_type!r} at row {idx}."
        raise ValueError(msg) from exc

    prompt = str(record.get(prompt_column, "") or "").strip()
    if not prompt:
        return None

    row_id = str(record.get("id", idx))
    return {
        "id": f"{row_id}_{data_type}",
        "prompt": prompt,
        "data_type": data_type,
        "source": "wildjailbreak",
    }


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {args.hf_dataset} (config={args.config_name}, split={args.split}) …")
    ds = load_dataset(
        args.hf_dataset,
        args.config_name,
        split=args.split,
        delimiter="\t",
        keep_default_na=False,
        trust_remote_code=False,
    )

    print(f"\nDataset schema:")
    print(f"  Features: {ds.features}")
    print(f"  Num rows: {len(ds)}")
    print(f"\nFirst row:")
    first = dict(ds[0])
    for key, val in first.items():
        display = str(val)[:120] + ("…" if len(str(val)) > 120 else "")
        print(f"  {key!r}: {display!r}")

    if args.print_schema_only:
        print("\n--print-schema-only: exiting without writing.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for idx, row in enumerate(ds):
        normalised = normalise_record(dict(row), idx)
        if normalised is not None:
            records.append(normalised)

    write_jsonl(output_path, records)

    # Count by data_type
    from collections import Counter
    counts = Counter(r["data_type"] for r in records)
    print(f"\nWrote {len(records)} records to {output_path}")
    print("  Breakdown:")
    for dt, n in sorted(counts.items()):
        print(f"    {dt}: {n}")


if __name__ == "__main__":
    main()
