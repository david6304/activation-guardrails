"""Download and normalise the WildJailbreak dataset to local JSONL.

Usage (run once, requires internet access — use cluster head node or local):
    python scripts/download_wildjailbreak.py --output data/raw/wildjailbreak/wildjailbreak.jsonl

This script downloads ``allenai/wildjailbreak`` from HuggingFace, normalises
it into the schema expected by ``agguardrails.data.load_wildjailbreak_examples``,
and saves a single JSONL file with one record per prompt.

Normalised schema (one record per row):
    {"id": str, "prompt": str, "data_type": str, "source": "wildjailbreak"}

data_type values:
    - "vanilla_harmful"     — direct harmful request
    - "vanilla_benign"      — benign prompt surface-similar to harmful one
    - "adversarial_harmful" — jailbroken version of a vanilla harmful prompt

The HuggingFace dataset (Jiang et al., NeurIPS 2024) has a train split where
each row contains parallel variants of a prompt:
    - "vanilla": the direct harmful request
    - "adversarial": the jailbroken version
    - "prompt": the corresponding benign prompt (surface-similar)
    - "label": "harmful" or "benign" indicating whether the row is a harmful
       or benign entry (some datasets include both in separate rows; see note
       below)

NOTE: The exact column names are printed below when the script runs. If the
schema differs from what this script expects, adjust the COLUMN MAPPING section
and rerun.
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
        "--split",
        default="train",
        help="HuggingFace dataset split to download (default: train).",
    )
    parser.add_argument(
        "--print-schema-only",
        action="store_true",
        help="Print dataset schema and first row then exit without writing.",
    )
    return parser.parse_args()


def normalise_record(record: dict, idx: int) -> list[dict]:
    """Convert one HuggingFace WildJailbreak row into normalised records.

    WildJailbreak rows are structured as parallel data:
        - "vanilla": direct harmful request (label=1)
        - "adversarial": jailbroken version of "vanilla" (label=1)
        - "prompt": benign prompt surface-similar to "vanilla" (label=0)

    Each row yields up to three normalised records (one per data_type).
    Records with empty prompts are skipped.

    See the print-schema-only output to verify column names before trusting this.
    """
    # ---------------------------------------------------------------------------
    # COLUMN MAPPING — adjust here if the actual schema differs
    # ---------------------------------------------------------------------------
    vanilla_col = "vanilla"       # direct harmful prompt
    adversarial_col = "adversarial"  # jailbroken version
    benign_col = "prompt"            # benign surface-similar prompt
    id_col = "id"                    # row identifier (may be absent)
    # ---------------------------------------------------------------------------

    row_id = str(record.get(id_col, idx))
    results = []

    for col, data_type in [
        (vanilla_col, "vanilla_harmful"),
        (adversarial_col, "adversarial_harmful"),
        (benign_col, "vanilla_benign"),
    ]:
        text = str(record.get(col, "") or "").strip()
        if not text:
            continue
        results.append(
            {
                "id": f"{row_id}_{data_type}",
                "prompt": text,
                "data_type": data_type,
                "source": "wildjailbreak",
            }
        )

    return results


def main() -> None:
    args = parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"Downloading {args.hf_dataset} (split={args.split}) …")
    ds = load_dataset(args.hf_dataset, split=args.split, trust_remote_code=False)

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
        records.extend(normalise_record(dict(row), idx))

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
