"""Sample benign production exchanges from WildChat-1M for FPR-threshold calibration.

The 0.1%-FPR detection threshold is the 99.9th percentile of the probe's per-exchange
max-EMA score over normal traffic. We take the first user->assistant exchange of each
English conversation, as-is (WildChat's own responses), to be scored teacher-forced
through the protected model later (calibration/eval is scored on the fly, not cached).

Reads the parquet shards straight from the local HF cache (offline): the cluster's
compute/head nodes cannot stream from the HF CDN, so prefetch the shards on a networked
node first with e.g.
  hf download allenai/WildChat-1M --repo-type dataset --include "data/train-0000[0-3]-of-00014.parquet"
then run this. No toxicity filter: real traffic includes a borderline tail and the
99.9th-percentile threshold is meant to sit above it (slightly conservative for v1).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def cached_shards():
    # Glob the cached snapshot directly (list_repo_files is a live API call that fails
    # offline); same approach as generate_responses.load_wildjailbreak.
    from huggingface_hub import constants

    snaps = Path(constants.HF_HUB_CACHE) / "datasets--allenai--WildChat-1M" / "snapshots"
    shards = sorted(p for p in snaps.glob("*/**/*.parquet") if "train" in p.name)
    if not shards:
        raise RuntimeError(f"no cached WildChat-1M train parquet under {snaps}; "
                           "prefetch shards on a networked node first (see module docstring)")
    return shards


def first_exchange(conv):
    if len(conv) < 2 or conv[0]["role"] != "user" or conv[1]["role"] != "assistant":
        return None
    prompt, response = conv[0]["content"], conv[1]["content"]
    if not prompt.strip() or not response.strip():
        return None
    return prompt, response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--out", default="data/wildchat_calib.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    shards = cached_shards()
    print(f"[shards] {len(shards)} cached parquet shards")

    # Collect candidate exchanges across shards, stopping once we have a comfortable
    # multiple of n, then shuffle so we don't bias toward the first shards' time/model
    # ordering (WildChat is roughly time-ordered).
    target = args.n * 3
    cands = []
    for sh in shards:
        df = pd.read_parquet(sh, columns=["conversation_hash", "conversation", "language"])
        df = df[df["language"] == "English"]
        for h, conv in zip(df["conversation_hash"], df["conversation"]):
            ex = first_exchange(conv)
            if ex is not None:
                cands.append((h, ex[0], ex[1]))
        print(f"  {sh.name}: {len(cands)} candidates so far", flush=True)
        if len(cands) >= target:
            break

    rng = np.random.default_rng(args.seed)
    rng.shuffle(cands)
    picks = cands[: args.n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for h, prompt, response in picks:
            f.write(json.dumps({"id": h, "prompt": prompt,
                                "response": response, "data_type": "wildchat"}) + "\n")
    print(f"[done] wrote {len(picks)} exchanges -> {out} "
          f"(from {len(cands)} candidates across {len(shards)} shards)")


if __name__ == "__main__":
    main()
