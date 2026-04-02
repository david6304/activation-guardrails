"""Label model responses as refusal or compliance using an LLM judge.

Reads generated responses from generate_responses.py and adds a
``refusal_label`` field (1=refusal, 0=compliance) determined by the model
judging its own output.

Prints a confusion matrix (source_label × refusal_label) for sanity checking.
Expected distribution: AdvBench (harmful) → ~95%+ refusal, Alpaca (benign)
→ ~99% compliance.  Significant deviation warrants manual inspection.

Usage:
    python scripts/main/label_refusals.py \\
        --config configs/main/refusal.yaml

    # Inspect output:
    python -c "
    from agguardrails.io import read_jsonl
    recs = read_jsonl('artifacts/responses/refusal/labelled_responses.jsonl')
    print(recs[0])
    "
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

from agguardrails.io import read_jsonl, save_metadata, write_jsonl
from agguardrails.judge import label_refusals
from agguardrails.models import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Label generated responses as refusal or compliance."
    )
    parser.add_argument(
        "--config",
        default="configs/main/refusal.yaml",
        help="Path to refusal experiment config.",
    )
    parser.add_argument(
        "--responses",
        default="artifacts/responses/refusal/responses.jsonl",
        help="Generated responses JSONL from generate_responses.py.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/responses/refusal/labelled_responses.jsonl",
        help="Output JSONL with refusal_label and judge_output fields added.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with Path(args.config).open() as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    judge_cfg = config["judge"]

    response_records = read_jsonl(args.responses)
    print(f"Loaded {len(response_records)} response records.", flush=True)

    print(f"Loading judge model: {model_cfg['name']}", flush=True)
    model, tokenizer = load_model_and_tokenizer(
        model_cfg["name"],
        torch_dtype=model_cfg["dtype"],
    )
    print("Judge model loaded.", flush=True)

    labelled = label_refusals(
        model,
        tokenizer,
        response_records,
        max_new_tokens=int(judge_cfg["max_new_tokens"]),
        batch_size=int(judge_cfg["batch_size"]),
        temperature=float(judge_cfg["temperature"]),
    )

    output_path = Path(args.output)
    write_jsonl(output_path, labelled)

    # Confusion matrix: source_label × refusal_label.
    counts: Counter = Counter(
        (rec["source_label"], rec["refusal_label"]) for rec in labelled
    )
    print("\nRefusal label distribution (source_label × refusal_label):")
    print(f"  {'source_label':<20} {'refusal=0':>12} {'refusal=1':>12}")
    for source_label in sorted({rec["source_label"] for rec in labelled}):
        n0 = counts.get((source_label, 0), 0)
        n1 = counts.get((source_label, 1), 0)
        print(f"  {source_label:<20} {n0:>12} {n1:>12}")

    n_unparseable = sum(
        1 for rec in labelled if rec.get("judge_output", "") == ""
    )

    save_metadata(
        output_path.with_suffix(".metadata.json"),
        config_path=args.config,
        responses_path=args.responses,
        output_path=str(output_path),
        model_name=model_cfg["name"],
        n_labelled=len(labelled),
        refusal_counts={
            f"{sl}:{rl}": c for (sl, rl), c in sorted(counts.items())
        },
    )

    print(f"\nLabelled responses written to: {output_path}")
    if n_unparseable:
        print(
            f"Warning: {n_unparseable} records have empty judge_output. "
            "Check for truncation issues."
        )


if __name__ == "__main__":
    main()
