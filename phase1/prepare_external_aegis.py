"""Build the C7 external-source judge input from Aegis 2.0 and freeze its manifest.

Cleans the Nvidia Aegis 2.0 prompts (drop redacted/empty, dedupe on normalised text,
drop anything NLLB would truncate at 256 tokens so the English and Swahili arms cover
identical rows), checks for overlap against every Phase 1 prompt, and writes a judge
input JSONL plus a manifest recording counts and SHA-256 checksums.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from probe_prompt import file_sha256, normalised_hash

AEGIS_DATASET = "nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
AEGIS_REVISION = "d86bb8bedff51d25ac834ab7838f1cc61acb7a2c"
AEGIS_FILES = ("train.json", "validation.json", "test.json")
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
MAX_NLLB_TOKENS = 256


def load_aegis():
    from huggingface_hub import snapshot_download

    root = Path(
        snapshot_download(
            AEGIS_DATASET,
            revision=AEGIS_REVISION,
            repo_type="dataset",
            local_files_only=True,
        )
    )
    rows = []
    for name in AEGIS_FILES:
        for row in json.loads((root / name).read_text()):
            row["aegis_split"] = name.removesuffix(".json")
            rows.append(row)
    return rows


def nllb_token_lengths(prompts):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        NLLB_MODEL, revision=NLLB_REVISION, src_lang="eng_Latn"
    )
    encodings = tok(prompts, truncation=False, add_special_tokens=True)["input_ids"]
    return [len(ids) for ids in encodings]


def phase1_hashes(paths):
    hashes = set()
    for path in paths:
        with Path(path).open() as f:
            for line in f:
                if line.strip():
                    hashes.add(normalised_hash(json.loads(line)["prompt"]))
    return hashes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/c7_aegis_judge_input.jsonl")
    parser.add_argument("--manifest", default="data/c7_external_manifest.json")
    parser.add_argument(
        "--phase1",
        nargs="+",
        default=["data/judged_main_prompts.jsonl", "data/judged_val_prompts.jsonl"],
    )
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()

    rows = load_aegis()
    counts = {"raw": len(rows)}

    kept = []
    for row in rows:
        prompt = (row.get("prompt") or "").strip()
        if not prompt or prompt == "REDACTED":
            continue
        row["prompt"] = prompt
        kept.append(row)
    counts["after_redacted_and_empty"] = len(kept)

    deduped = {}
    for row in kept:
        deduped.setdefault(normalised_hash(row["prompt"]), row)
    kept = list(deduped.values())
    counts["after_dedupe"] = len(kept)

    existing = phase1_hashes(args.phase1)
    overlap = [row for row in kept if normalised_hash(row["prompt"]) in existing]
    counts["phase1_overlap_dropped"] = len(overlap)
    kept = [row for row in kept if normalised_hash(row["prompt"]) not in existing]

    lengths = nllb_token_lengths([row["prompt"] for row in kept])
    long_rows = sum(length > MAX_NLLB_TOKENS for length in lengths)
    kept = [
        row for row, length in zip(kept, lengths) if length <= MAX_NLLB_TOKENS
    ]
    counts["nllb_truncated_dropped"] = long_rows
    counts["final"] = len(kept)

    counts["non_latin_script_kept"] = sum(
        sum(ord(c) < 128 for c in row["prompt"]) / len(row["prompt"]) < 0.9
        for row in kept
    )

    kept.sort(key=lambda row: row["id"])
    records = [
        json.dumps(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "aegis_prompt_label": row["prompt_label"],
                "aegis_categories": row["violated_categories"],
                "aegis_split": row["aegis_split"],
            }
        )
        for row in kept
    ]
    Path(args.out).write_text("\n".join(records) + "\n")

    # The judge runs ~10k prompts per 100 GPU-minutes, so shard to keep each job
    # well inside Eddie's wall-clock limit.
    shard_paths = []
    for index in range(args.shards):
        path = Path(args.out).with_name(
            Path(args.out).stem + f"_shard{index}" + Path(args.out).suffix
        )
        path.write_text("\n".join(records[index :: args.shards]) + "\n")
        shard_paths.append(path)

    manifest = {
        "dataset": AEGIS_DATASET,
        "revision": AEGIS_REVISION,
        "nllb_model": NLLB_MODEL,
        "nllb_revision": NLLB_REVISION,
        "max_nllb_tokens": MAX_NLLB_TOKENS,
        "counts": counts,
        "aegis_prompt_label": dict(
            Counter(row["prompt_label"] for row in kept).most_common()
        ),
        "judge_input": args.out,
        "judge_input_sha256": file_sha256(Path(args.out)),
        "shards": {
            str(path): {"rows": len(path.read_text().splitlines()),
                        "sha256": file_sha256(path)}
            for path in shard_paths
        },
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(counts, indent=2))
    print(f"[done] {args.out} sha256={manifest['judge_input_sha256']}")


if __name__ == "__main__":
    main()
