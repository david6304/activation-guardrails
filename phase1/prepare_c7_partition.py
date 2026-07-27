"""Freeze the C7 external tune/test partition and draw the hand-check sample.

Thresholds are set on external *tune* negatives only; TPR and realised FPR are
reported on the external *test* partition. Positives play no part in calibration, so
they all go to test. Also writes a stratified hand-check CSV covering both directions
of Aegis-versus-judge disagreement.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from probe_prompt import load_judged_rows, strings_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", nargs="+", default=[
        "data/c7_judged_shard0.jsonl", "data/c7_judged_shard1.jsonl"])
    parser.add_argument("--out", default="data/c7_partition.json")
    parser.add_argument("--handcheck", default="data/c7_handcheck.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-handcheck", type=int, default=50)
    parser.add_argument("--merged", default="data/c7_judged_all.jsonl")
    args = parser.parse_args()

    merged = Path(args.merged)
    with merged.open("w") as out:
        for path in args.judged:
            out.write(Path(path).read_text())

    rows, parse_errors, malformed, pilot_overlap, excluded = load_judged_rows(
        merged, 0, args.seed, keep_protected_group=False
    )

    labels = np.array([bool(row["harmful"]) for row in rows])
    positives = [row for row, label in zip(rows, labels) if label]
    negatives = [row for row, label in zip(rows, labels) if not label]

    generator = np.random.default_rng(args.seed)
    order = generator.permutation(len(negatives))
    half = len(negatives) // 2
    tune = [negatives[int(i)] for i in order[:half]]
    test_negatives = [negatives[int(i)] for i in order[half:]]

    partition = {
        "seed": args.seed,
        "judge_parse_errors": parse_errors,
        "malformed": malformed,
        "pilot_overlap_dropped": pilot_overlap,
        "non_operational_positives_excluded": excluded,
        "counts": {
            "rows": len(rows),
            "positives": len(positives),
            "negatives": len(negatives),
            "tune_negatives": len(tune),
            "test_negatives": len(test_negatives),
            "test_positives": len(positives),
        },
        "tune_ids": sorted(row["id"] for row in tune),
        "test_ids": sorted(row["id"] for row in test_negatives + positives),
    }
    partition["tune_ids_sha256"] = strings_sha256(partition["tune_ids"])
    partition["test_ids_sha256"] = strings_sha256(partition["test_ids"])
    Path(args.out).write_text(json.dumps(partition, indent=2) + "\n")

    strata = {
        "aegis_unsafe_judge_harmful": [
            r for r in rows if r["aegis_prompt_label"] == "unsafe" and r["harmful"]],
        "aegis_unsafe_judge_benign": [
            r for r in rows if r["aegis_prompt_label"] == "unsafe" and not r["harmful"]],
        "aegis_safe_judge_benign": [
            r for r in rows if r["aegis_prompt_label"] == "safe" and not r["harmful"]],
        "aegis_safe_judge_harmful": [
            r for r in rows if r["aegis_prompt_label"] == "safe" and r["harmful"]],
    }
    per_stratum = args.n_handcheck // len(strata)
    import csv

    with Path(args.handcheck).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "stratum", "aegis_label", "aegis_categories",
                         "judge_harmful", "judge_category", "my_label",
                         "judge_reason", "prompt"])
        for name, subset in strata.items():
            take = min(per_stratum, len(subset))
            picks = generator.choice(len(subset), size=take, replace=False)
            for index in picks:
                row = subset[int(index)]
                writer.writerow([row["id"], name, row["aegis_prompt_label"],
                                 row["aegis_categories"], int(row["harmful"]),
                                 row["category"], "", row["judge_reason"],
                                 row["prompt"]])
            print(f"  {name}: {len(subset)} available, sampled {take}")

    print(json.dumps(partition["counts"], indent=2))
    print(f"[done] {args.out} tune_sha256={partition['tune_ids_sha256'][:16]} "
          f"test_sha256={partition['test_ids_sha256'][:16]}")


if __name__ == "__main__":
    main()
