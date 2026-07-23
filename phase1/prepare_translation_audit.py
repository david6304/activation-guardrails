"""Create a blinded, stratified English--Swahili fidelity-audit sheet."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from probe_prompt import load_judged_rows, split_rows


WEAPONS_LONG = (
    "weapons & CBRN (chemical, biological, radiological, nuclear, explosive)"
)
HARMFUL_ALLOCATION = {
    "fraud, scams & deception": 10,
    "privacy violation / doxxing / PII extraction": 10,
    "non-violent crime": 8,
    "cyber-intrusion & malware": 7,
    "sex crimes & child exploitation": 5,
    "self-harm & suicide": 4,
    "violent crime": 3,
    "weapons & CBRN": 3,
}


def normalise_category(category):
    return "weapons & CBRN" if category == WEAPONS_LONG else category


def load_translations(path):
    translations = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                translations[row["prompt"]] = row["translation"]
    return translations


def sample_rows(rows, seed):
    rng = np.random.default_rng(seed)
    harmful_by_category = defaultdict(list)
    benign_by_stratum = defaultdict(list)
    for row in rows:
        if row["harmful"]:
            harmful_by_category[normalise_category(str(row["category"]))].append(row)
        else:
            benign_by_stratum[str(row["data_type"])].append(row)

    selected = []
    for category, count in HARMFUL_ALLOCATION.items():
        candidates = harmful_by_category[category]
        if len(candidates) < count:
            raise ValueError(
                f"category {category!r} has {len(candidates)} rows, needs {count}"
            )
        indices = rng.choice(len(candidates), size=count, replace=False)
        selected.extend(candidates[index] for index in indices)

    benign_total = 50
    strata = sorted(benign_by_stratum)
    base, remainder = divmod(benign_total, len(strata))
    for stratum_index, stratum in enumerate(strata):
        count = base + int(stratum_index < remainder)
        candidates = benign_by_stratum[stratum]
        if len(candidates) < count:
            raise ValueError(
                f"benign stratum {stratum!r} has {len(candidates)} rows, needs {count}"
            )
        indices = rng.choice(len(candidates), size=count, replace=False)
        selected.extend(candidates[index] for index in indices)

    order = rng.permutation(len(selected))
    return [selected[index] for index in order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", default="data/judged_main_prompts.jsonl")
    parser.add_argument(
        "--translations", default="data/phase1_translations/swahili.jsonl"
    )
    parser.add_argument("--audit-out", default="data/swahili_fidelity_audit.csv")
    parser.add_argument("--key-out", default="data/swahili_fidelity_audit_key.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows, *_ = load_judged_rows(
        Path(args.inputs), limit=0, seed=args.seed, keep_protected_group=False
    )
    _, _, test_rows = split_rows(rows, args.seed)
    selected = sample_rows(test_rows, args.seed)
    translations = load_translations(Path(args.translations))
    if any(row["prompt"] not in translations for row in selected):
        raise ValueError("frozen manifest is missing a selected prompt")

    audit_path = Path(args.audit_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "audit_id",
                "english_prompt",
                "swahili_prompt",
                "intent_preserved",
                "target_preserved",
                "severity_preserved",
                "operational_category_preserved",
                "overall_preserved",
                "notes",
            ),
        )
        writer.writeheader()
        for audit_index, row in enumerate(selected, start=1):
            writer.writerow(
                {
                    "audit_id": f"SWA-{audit_index:03d}",
                    "english_prompt": row["prompt"],
                    "swahili_prompt": translations[row["prompt"]],
                    "intent_preserved": "",
                    "target_preserved": "",
                    "severity_preserved": "",
                    "operational_category_preserved": "",
                    "overall_preserved": "",
                    "notes": "",
                }
            )

    key = {
        f"SWA-{audit_index:03d}": {
            "id": str(row["id"]),
            "harmful": bool(row["harmful"]),
            "category": normalise_category(str(row["category"])),
            "source_stratum": str(row["data_type"]),
        }
        for audit_index, row in enumerate(selected, start=1)
    }
    key_path = Path(args.key_out)
    key_path.write_text(json.dumps(key, indent=2) + "\n")
    harmful_count = sum(row["harmful"] for row in selected)
    print(
        f"[audit] n={len(selected)} harmful={harmful_count} "
        f"benign={len(selected) - harmful_count}"
    )
    print(f"[done] blinded sheet -> {audit_path}")
    print(f"[done] hidden key -> {key_path}")


if __name__ == "__main__":
    main()
