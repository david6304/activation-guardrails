"""Materialise the frozen Phase 1 French, Hindi and Zulu tune/test translations."""

import argparse
import json
from pathlib import Path

from capability_qa import LANGS
from probe_prompt import (
    file_sha256,
    load_judged_rows,
    load_or_translate,
    split_rows,
)


NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
NEW_LANGUAGES = ("french", "hindi", "zulu")
EXPECTED_SPLIT = (5341, 1781, 1781)


def verified_cached_revision(model_id, revision):
    from transformers import AutoConfig

    pinned = AutoConfig.from_pretrained(
        model_id, revision=revision, local_files_only=True
    )
    default = AutoConfig.from_pretrained(model_id, local_files_only=True)
    pinned_hash = str(getattr(pinned, "_commit_hash", "") or "")
    default_hash = str(getattr(default, "_commit_hash", "") or "")
    if pinned_hash != revision or default_hash != revision:
        raise RuntimeError(
            f"cached NLLB revision mismatch: pinned={pinned_hash!r}, "
            f"default={default_hash!r}, expected={revision!r}"
        )
    return pinned_hash


def write_metadata(path, metadata):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="data/judged_main_prompts.jsonl")
    parser.add_argument("--translations-dir", default="data/phase1_translations")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.seed != 0:
        raise ValueError("the frozen Phase 1 split requires --seed 0")
    translations_dir = Path(args.translations_dir)
    metadata_path = translations_dir / "metadata.json"
    swahili_path = translations_dir / "swahili.jsonl"
    if not metadata_path.exists() or not swahili_path.exists():
        raise FileNotFoundError(
            "the frozen Swahili manifest and metadata must exist before extending them"
        )

    metadata = json.loads(metadata_path.read_text())
    if metadata.get("nllb_model") != NLLB_MODEL:
        raise ValueError("frozen Swahili metadata uses a different NLLB model")
    if metadata.get("nllb_revision") != NLLB_REVISION:
        raise ValueError("frozen Swahili metadata uses a different NLLB revision")
    swahili_sha256 = file_sha256(swahili_path)
    if metadata.get("swahili_sha256") != swahili_sha256:
        raise ValueError("frozen Swahili manifest checksum mismatch")
    verified_cached_revision(NLLB_MODEL, NLLB_REVISION)

    rows, _, _, _, _ = load_judged_rows(
        Path(args.inp), 0, args.seed, keep_protected_group=False
    )
    train_rows, tune_rows, test_rows = split_rows(rows, args.seed)
    split_sizes = (len(train_rows), len(tune_rows), len(test_rows))
    if split_sizes != EXPECTED_SPLIT:
        raise ValueError(f"unexpected frozen split sizes: {split_sizes}")
    prompts = [row["prompt"] for row in tune_rows + test_rows]

    manifests = dict(metadata.get("manifests", {}))
    manifests["swahili"] = {
        "code": LANGS["swahili"],
        "sha256": swahili_sha256,
        "inputs_exceeding_256_tokens": int(
            metadata.get("inputs_exceeding_256_tokens", 0)
        ),
    }
    for language in NEW_LANGUAGES:
        path = translations_dir / f"{language}.jsonl"
        _, truncated_hashes = load_or_translate(
            prompts,
            path,
            NLLB_MODEL,
            LANGS[language],
            allow_translate=True,
        )
        manifests[language] = {
            "code": LANGS[language],
            "sha256": file_sha256(path),
            "inputs_exceeding_256_tokens": len(truncated_hashes),
        }
        print(
            f"[manifest] {language} rows={len(prompts)} "
            f"sha256={manifests[language]['sha256']} "
            f"truncated_256={len(truncated_hashes)}",
            flush=True,
        )

    metadata["manifests"] = manifests
    metadata["split"] = {
        "seed": args.seed,
        "train": len(train_rows),
        "tune": len(tune_rows),
        "test": len(test_rows),
        "translated_splits": ["tune", "test"],
    }
    write_metadata(metadata_path, metadata)
    print(f"[done] {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
