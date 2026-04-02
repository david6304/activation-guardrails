"""Download AdvBench and Alpaca datasets for the refusal-probing experiment.

AdvBench (Zou et al., 2023): 520 harmful behavioural prompts.
  Source: llm-attacks/llm-attacks GitHub repo.

Alpaca (Taori et al., 2023): ~52K benign instruction-following examples.
  Source: tatsu-lab/alpaca on HuggingFace (direct JSON download).

Both are saved to data/raw/ for use by build_refusal_dataset.py.

Usage:
    python scripts/main/download_advbench_alpaca.py
    python scripts/main/download_advbench_alpaca.py --output-dir data/raw
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main"
    "/data/advbench/harmful_behaviors.csv"
)
ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/alpaca/main/alpaca_data.json"
)


def download_file(url: str, dest: Path) -> None:
    """Download *url* to *dest*, skipping if *dest* already exists."""
    if dest.exists():
        print(f"  Already exists, skipping: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"  Saved: {dest} ({dest.stat().st_size:,} bytes)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download AdvBench and Alpaca for the refusal-probing experiment."
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Root directory for raw data (default: data/raw).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)

    print("Downloading AdvBench...")
    download_file(ADVBENCH_URL, root / "advbench" / "harmful_behaviors.csv")

    print("Downloading Alpaca...")
    download_file(ALPACA_URL, root / "alpaca" / "alpaca_data.json")

    print("Done.")


if __name__ == "__main__":
    main()
