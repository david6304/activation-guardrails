"""Sample benign production exchanges from WildChat-1M for FPR-threshold calibration.

The 0.1%-FPR detection threshold is the 99.9th percentile of the probe's per-exchange
max-EMA score over normal traffic. We take the first user->assistant exchange of each
English conversation, as-is (WildChat's own responses), to be scored teacher-forced
through the protected model later (calibration/eval is scored on the fly, not cached).

No toxicity filter: real traffic includes a borderline tail, and the 99.9th-percentile
threshold is meant to sit above it (slightly conservative for v1).
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


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
    ap.add_argument("--shuffle-buffer", type=int, default=50000,
                    help="streaming shuffle buffer, to avoid time/model ordering bias")
    args = ap.parse_args()

    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = scanned = 0
    with out.open("w") as f:
        for r in ds:
            scanned += 1
            if r.get("language") != "English":
                continue
            ex = first_exchange(r["conversation"])
            if ex is None:
                continue
            f.write(json.dumps({"id": r["conversation_hash"], "prompt": ex[0],
                                "response": ex[1], "data_type": "wildchat"}) + "\n")
            n += 1
            if n % 5000 == 0:
                print(f"  {n} written ({scanned} scanned)", flush=True)
            if n >= args.n:
                break
    print(f"[done] wrote {n} exchanges -> {out} (scanned {scanned})")


if __name__ == "__main__":
    main()
