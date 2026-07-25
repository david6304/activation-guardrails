"""Translate the frozen C3 WildChat pool to Swahili with the pinned NLLB revision.

Gives a shifted background of the same size as the plain pool, so the matched
0.1% operating point can be set on shifted traffic. NLLB truncates its input at
256 tokens; WildChat's length tail is heavy, so the truncated fraction is
recorded and must be reported beside any Swahili background alert rate.
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
