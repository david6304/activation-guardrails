"""Analyse recovered WebQuestions capability-gate outputs.

The established automatic substring metric is read unchanged. Confidence intervals
resample paired question IDs, and the manual audit is stored separately.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCES = {
    "plain": ("data/cap_qa_base64_27b.jsonl", "plain"),
    "base64": ("data/cap_qa_base64_27b.jsonl", "base64"),
    "reverse": ("data/cap_qa_27b.jsonl", "reverse"),
    "nato": ("data/cap_qa_27b.jsonl", "nato"),
    "morse": ("data/cap_qa_27b.jsonl", "morse"),
    "zulu": ("data/cap_qa_27b.jsonl", "zulu"),
    "french": ("data/cap_qa_langs_27b.jsonl", "french"),
    "hindi": ("data/cap_qa_langs_27b.jsonl", "hindi"),
    "swahili": ("data/cap_qa_langs_27b.jsonl", "swahili"),
}


def load_jsonl(path):
    with Path(path).open() as source:
        return [json.loads(line) for line in source if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_by_condition(path):
    grouped = {}
    for row in load_jsonl(path):
        grouped.setdefault(row["cipher"], {})[int(row["qidx"])] = row
    return grouped


def percentile_interval(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def analyse_transform(path, condition, resamples, seed, tokenizer, max_new_tokens):
    grouped = rows_by_condition(path)
    condition_rows = grouped[condition]
    plain_rows = grouped["plain"]
    ids = sorted(condition_rows)
    if ids != sorted(plain_rows):
        raise ValueError(f"unpaired question IDs for {condition} in {path}")

    correct = np.asarray([bool(condition_rows[i]["correct"]) for i in ids], dtype=float)
    plain = np.asarray([bool(plain_rows[i]["correct"]) for i in ids], dtype=float)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(ids), size=(resamples, len(ids)))
    boot_accuracy = correct[sampled].mean(axis=1)
    boot_plain = plain[sampled].mean(axis=1)
    boot_difference = (correct - plain)[sampled].mean(axis=1)
    boot_retention = np.divide(
        boot_accuracy,
        boot_plain,
        out=np.full_like(boot_accuracy, np.nan),
        where=boot_plain > 0,
    )

    accuracy = float(correct.mean())
    plain_accuracy = float(plain.mean())
    ceiling_outputs = sum(
        len(tokenizer.encode(row["answer"], add_special_tokens=False))
        == max_new_tokens
        for row in condition_rows.values()
    )
    return {
        "source": path,
        "n": len(ids),
        "correct": int(correct.sum()),
        "accuracy": accuracy,
        "accuracy_ci": percentile_interval(boot_accuracy),
        "plain_correct": int(plain.sum()),
        "plain_accuracy": plain_accuracy,
        "paired_difference": accuracy - plain_accuracy,
        "paired_difference_ci": percentile_interval(boot_difference),
        "retention": accuracy / plain_accuracy,
        "retention_ci": percentile_interval(boot_retention[~np.isnan(boot_retention)]),
        "empty_responses": sum(not row["answer"].strip() for row in condition_rows.values()),
        "truncation_count": None,
        "retokenised_at_64_token_ceiling": ceiling_outputs,
        "truncation_note": (
            "exact flag not recorded; the ceiling count re-tokenises decoded text with "
            "the pinned tokenizer and is a conservative truncation indicator"
        ),
    }


def make_audit(path, per_stratum, seed):
    if Path(path).exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    rng = np.random.default_rng(seed)
    selected = []
    for condition, (source, source_condition) in SOURCES.items():
        rows = [
            row
            for row in load_jsonl(source)
            if row["cipher"] == source_condition
        ]
        for automatic_correct in (True, False):
            stratum = [row for row in rows if bool(row["correct"]) == automatic_correct]
            take = min(per_stratum, len(stratum))
            indices = sorted(rng.choice(len(stratum), size=take, replace=False).tolist())
            for index in indices:
                row = stratum[index]
                selected.append(
                    {
                        "audit_id": f"{condition}-{row['qidx']}",
                        "source": source,
                        "qidx": int(row["qidx"]),
                        "condition": condition,
                        "question": row["question"],
                        "golds": row["golds"],
                        "answer": row["answer"],
                        "automatic_correct": automatic_correct,
                        "manual_correct": None,
                        "agreement": None,
                        "audit_note": "",
                    }
                )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as output:
        for row in selected:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(selected)


def manual_audit_summary(path, labels_path, labelled_path):
    rows = load_jsonl(path)
    labels_file = Path(labels_path)
    if not labels_file.exists():
        return None
    labels = json.loads(labels_file.read_text())
    audit_ids = {row["audit_id"] for row in rows}
    if set(labels) != audit_ids:
        missing = sorted(audit_ids - set(labels))
        extra = sorted(set(labels) - audit_ids)
        raise ValueError(f"manual label mismatch: missing={missing}, extra={extra}")
    labelled = []
    for row in rows:
        label = labels[row["audit_id"]]
        row["manual_correct"] = bool(label["manual_correct"])
        row["agreement"] = bool(row["automatic_correct"]) == row["manual_correct"]
        row["audit_note"] = label.get("audit_note", "")
        labelled.append(row)
    with Path(labelled_path).open("w") as output:
        for row in labelled:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "n": len(labelled),
        "automatic_correct": sum(bool(row["automatic_correct"]) for row in labelled),
        "manual_correct": sum(bool(row["manual_correct"]) for row in labelled),
        "agreement": sum(bool(row["agreement"]) for row in labelled),
        "agreement_rate": sum(bool(row["agreement"]) for row in labelled) / len(labelled),
        "false_positives": sum(
            bool(row["automatic_correct"]) and not bool(row["manual_correct"])
            for row in labelled
        ),
        "false_negatives": sum(
            not bool(row["automatic_correct"]) and bool(row["manual_correct"])
            for row in labelled
        ),
    }
    summary["by_condition"] = {}
    for condition in sorted({row["condition"] for row in labelled}):
        condition_rows = [row for row in labelled if row["condition"] == condition]
        summary["by_condition"][condition] = {
            "n": len(condition_rows),
            "automatic_correct": sum(
                bool(row["automatic_correct"]) for row in condition_rows
            ),
            "manual_correct": sum(bool(row["manual_correct"]) for row in condition_rows),
            "agreement": sum(bool(row["agreement"]) for row in condition_rows),
            "false_positives": sum(
                bool(row["automatic_correct"]) and not bool(row["manual_correct"])
                for row in condition_rows
            ),
            "false_negatives": sum(
                not bool(row["automatic_correct"]) and bool(row["manual_correct"])
                for row in condition_rows
            ),
        }
    return summary


def metric_summary(values, sampled):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "ci": percentile_interval(values[sampled].mean(axis=1)),
    }


def analyse_fidelity(path, resamples, seed):
    rows = load_jsonl(path)
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate fidelity IDs")
    if any("decoded" not in row or not row["decoded"].strip() for row in rows):
        raise ValueError("missing or empty fidelity response")

    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(rows), size=(resamples, len(rows)))

    def stratum(keep):
        indices = np.flatnonzero(keep)
        stratum_sampled = rng.integers(
            0, len(indices), size=(resamples, len(indices))
        )
        selected = [rows[int(index)] for index in indices]
        return {
            "n": len(selected),
            "exact_match": metric_summary(
                [row["exact"] for row in selected], stratum_sampled
            ),
            "token_f1": metric_summary(
                [row["token_f1"] for row in selected], stratum_sampled
            ),
            "character_similarity": metric_summary(
                [row["character_similarity"] for row in selected], stratum_sampled
            ),
            "truncated": sum(bool(row["truncated"]) for row in selected),
        }

    lengths = np.asarray([len(row["plaintext"]) for row in rows])
    return {
        "source": path,
        "source_sha256": sha256(path),
        "n": len(rows),
        "exact_match": metric_summary([row["exact"] for row in rows], sampled),
        "token_f1": metric_summary([row["token_f1"] for row in rows], sampled),
        "character_similarity": metric_summary(
            [row["character_similarity"] for row in rows], sampled
        ),
        "truncated": sum(bool(row["truncated"]) for row in rows),
        "duplicate_ids": 0,
        "missing_or_empty_responses": 0,
        "at_most_90_characters": stratum(lengths <= 90),
        "over_90_characters": stratum(lengths > 90),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/capability_gate_analysis.json")
    parser.add_argument(
        "--audit-out", default="data/capability_gate_manual_audit.jsonl"
    )
    parser.add_argument(
        "--audit-labels", default="data/capability_gate_manual_labels.json"
    )
    parser.add_argument(
        "--labelled-audit-out",
        default="data/capability_gate_manual_audit_labelled.jsonl",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--audit-per-stratum", type=int, default=5)
    parser.add_argument("--audit-seed", type=int, default=0)
    parser.add_argument("--make-audit", action="store_true")
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument(
        "--model-revision", default="005ad3404e59d6023443cb575daa05336842228a"
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--fidelity", default="data/p1_decode_fidelity_57329555.jsonl"
    )
    args = parser.parse_args()

    if args.make_audit:
        audit_n = make_audit(args.audit_out, args.audit_per_stratum, args.audit_seed)
        print(f"[audit] wrote {audit_n} rows to {args.audit_out}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.model_revision, local_files_only=True
    )
    results = {
        "metric": "WebQuestions normalised gold-substring accuracy",
        "bootstrap_unit": "paired qidx",
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_resamples": args.bootstrap_resamples,
        "model": args.model,
        "model_revision": args.model_revision,
        "inputs": {path: sha256(path) for path, _ in sorted(set(SOURCES.values()))},
        "transformations": {
            condition: analyse_transform(
                source,
                source_condition,
                args.bootstrap_resamples,
                args.bootstrap_seed,
                tokenizer,
                args.max_new_tokens,
            )
            for condition, (source, source_condition) in SOURCES.items()
        },
        "decode_fidelity": analyse_fidelity(
            args.fidelity, args.bootstrap_resamples, args.bootstrap_seed
        ),
        "manual_audit_design": {
            "seed": args.audit_seed,
            "maximum_per_transformation_and_automatic_label": args.audit_per_stratum,
            "sampling": "without replacement within each transformation/label stratum",
        },
        "manual_audit": manual_audit_summary(
            args.audit_out, args.audit_labels, args.labelled_audit_out
        ),
    }
    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    print(f"[analysis] wrote {args.out}")


if __name__ == "__main__":
    main()
