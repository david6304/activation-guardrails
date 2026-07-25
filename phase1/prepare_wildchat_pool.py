"""Freeze the C3 WildChat background pool: 50,000 first-user-turn prompts.

Unlabelled real traffic, used as a large negative pool so the 1% and 0.1%
operating points are estimated from more than a dozen tune negatives. Nothing
here reuses `data/wildchat_scores.jsonl`, which holds response-level EMA scores
from `probe_v1`.

No language filter (the pool is meant to look like traffic, not like the
English test split). Prompts are deduplicated on normalised text and checked
against every Phase 1 prompt so the pool cannot overlap train, tune or test.
"""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from load_wildchat import cached_shards

PHASE1_PROMPTS = "data/judged_main_prompts.jsonl"
WHITESPACE = re.compile(r"\s+")


def normalise(text):
    return WHITESPACE.sub(" ", text).strip().casefold()


def first_user_prompt(conversation):
    if len(conversation) == 0 or conversation[0]["role"] != "user":
        return None
    prompt = conversation[0]["content"]
    return prompt if prompt.strip() else None


def phase1_keys(path):
    keys = set()
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            prompt = row.get("prompt")
            if prompt:
                keys.add(normalise(prompt))
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50000)
    parser.add_argument("--phase1-prompts", default=PHASE1_PROMPTS)
    parser.add_argument("--out", default="data/c3_wildchat_prompts.jsonl")
    parser.add_argument("--manifest", default="data/c3_wildchat_manifest.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    excluded = phase1_keys(args.phase1_prompts)
    shards = cached_shards()
    print(f"[shards] {len(shards)} cached parquet shards", flush=True)

    candidates = []
    seen = set()
    overlaps = 0
    duplicates = 0
    used_shards = []
    for shard in shards:
        frame = pd.read_parquet(
            shard, columns=["conversation_hash", "conversation", "language"]
        )
        used_shards.append(shard.name)
        for digest, conversation, language in zip(
            frame["conversation_hash"], frame["conversation"], frame["language"]
        ):
            prompt = first_user_prompt(conversation)
            if prompt is None:
                continue
            key = normalise(prompt)
            if key in excluded:
                overlaps += 1
                continue
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            candidates.append((str(digest), prompt, str(language)))
        print(f"  {shard.name}: {len(candidates)} candidates", flush=True)
        if len(candidates) >= args.n * 3:
            break

    if len(candidates) < args.n:
        raise RuntimeError(f"only {len(candidates)} candidates for --n {args.n}")

    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(candidates))[: args.n]
    picks = [candidates[index] for index in sorted(order)]

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for digest, prompt, language in picks:
            handle.write(
                json.dumps({"id": digest, "prompt": prompt, "language": language})
                + "\n"
            )

    lengths = np.array([len(prompt) for _, prompt, _ in picks])
    manifest = {
        "source": "allenai/WildChat-1M",
        "turn": "first user turn, verbatim",
        "n": len(picks),
        "seed": args.seed,
        "shards": used_shards,
        "candidates": len(candidates),
        "dropped_duplicates": duplicates,
        "dropped_phase1_overlap": overlaps,
        "prompts_path": str(output_path),
        "prompts_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "language_counts": dict(
            Counter(language for _, _, language in picks).most_common(15)
        ),
        "char_length_quantiles": {
            str(q): float(np.quantile(lengths, q))
            for q in (0.5, 0.75, 0.9, 0.99, 1.0)
        },
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    print(f"[done] {output_path}", flush=True)


if __name__ == "__main__":
    main()
