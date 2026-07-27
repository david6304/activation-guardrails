"""Translate the frozen C7 external prompts to Swahili with the pinned NLLB revision."""

import argparse
import json
from pathlib import Path

from capability_qa import LANGS
from phase1.prepare_external_aegis import NLLB_MODEL, NLLB_REVISION
from probe_prompt import file_sha256, load_or_translate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", default="data/c7_judged_all.jsonl")
    parser.add_argument("--partition", default="data/c7_partition.json")
    parser.add_argument("--out", default="data/c7_translations/swahili.jsonl")
    parser.add_argument("--manifest", default="data/c7_external_manifest.json")
    args = parser.parse_args()

    partition = json.loads(Path(args.partition).read_text())
    keep = set(partition["tune_ids"]) | set(partition["test_ids"])
    prompts = []
    with Path(args.judged).open() as f:
        for line in f:
            row = json.loads(line)
            if row["id"] in keep:
                prompts.append(row["prompt"])
    if len(prompts) != len(keep):
        raise ValueError(f"expected {len(keep)} partition prompts, found {len(prompts)}")

    path = Path(args.out)
    _, truncated = load_or_translate(
        prompts, path, NLLB_MODEL, LANGS["swahili"], allow_translate=True
    )
    # The input pool was filtered to <=256 NLLB tokens, so nothing may truncate here.
    if truncated:
        raise ValueError(f"{len(truncated)} external prompts truncated at 256 tokens")

    manifest = json.loads(Path(args.manifest).read_text())
    manifest["swahili"] = {
        "code": LANGS["swahili"],
        "path": str(path),
        "rows": len(prompts),
        "sha256": file_sha256(path),
        "nllb_model": NLLB_MODEL,
        "nllb_revision": NLLB_REVISION,
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[done] {path} rows={len(prompts)} sha256={manifest['swahili']['sha256']}")


if __name__ == "__main__":
    main()
