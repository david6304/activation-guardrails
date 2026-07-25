"""Translate the C3 WildChat pool's English rows to Swahili (optional arm).

A shifted background lets the matched operating point be set on shifted
traffic. Two constraints make this a sensitivity analysis rather than a
matching-size counterpart to the plain pool, and both must be reported with any
number it produces:

1. Only 47% of the pool is labelled English, and NLLB is invoked with
   `src_lang="eng_Latn"`. Translating the whole pool would mis-specify the
   source language for most of it, so only English rows are translated and the
   Swahili background is correspondingly smaller.
2. NLLB truncates its input at 256 tokens and WildChat's length tail is heavy,
   so long prompts are translated from a prefix. The truncated fraction is
   recorded.
"""

import argparse
import json
from pathlib import Path

from guard_screen import translate_nllb
from phase1.extend_multilingual_guards import resolve_cached_snapshot
from probe_prompt import file_sha256, translation_truncation_flags

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
TARGET = "swh_Latn"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", default="data/c3_wildchat_prompts.jsonl")
    parser.add_argument("--out", default="data/c3_wildchat_swahili.jsonl")
    parser.add_argument("--manifest", default="data/c3_wildchat_swahili_manifest.json")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    rows = [json.loads(line) for line in open(args.prompts) if line.strip()]
    total = len(rows)
    rows = [row for row in rows if row["language"] == "English"]
    print(f"[input] {len(rows)}/{total} English-labelled prompts", flush=True)
    if args.limit:
        rows = rows[: args.limit]
    snapshot, revision = resolve_cached_snapshot(NLLB_MODEL)
    if revision != NLLB_REVISION:
        raise RuntimeError(f"cached NLLB revision {revision!r} != {NLLB_REVISION!r}")

    texts = [row["prompt"].rstrip() for row in rows]
    flags = translation_truncation_flags(texts, str(snapshot))
    print(f"[translate] {len(rows)} prompts -> {TARGET}", flush=True)
    translations = translate_nllb(texts, str(snapshot), TARGET)

    truncated = 0
    empty = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row, text in zip(rows, texts):
            translation = translations[text]
            truncated += int(flags[text])
            empty += int(not translation.strip())
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "prompt": row["prompt"],
                        "translation": translation,
                        "truncated_256": bool(flags[text]),
                    }
                )
                + "\n"
            )

    manifest = {
        "nllb_model": NLLB_MODEL,
        "nllb_revision": revision,
        "target": TARGET,
        "n": len(rows),
        "pool_n": total,
        "selection": "rows whose WildChat language label is English",
        "truncated_256": truncated,
        "empty_translations": empty,
        "translations_path": str(output_path),
        "translations_sha256": file_sha256(output_path),
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"[done] {output_path}", flush=True)


if __name__ == "__main__":
    main()
