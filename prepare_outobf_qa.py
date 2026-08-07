"""Benign obfuscated-OUTPUT capability items (CPU).

Section 11 measured cipher-output capability on harmful prompts, where the 27B's
failure could be capability or residual refusal. This builds the benign counterpart
so the same three conditions can be run on a frontier model without asking for harmful
content: plain (ceiling), `out` (plain in, base64 out) and `inout` (base64 both ways).

Items are the first 50 of the same seed-0 WebQuestions draw `capability_qa.py` already
used, so the existing 27B plain ceiling (54%) is the ceiling for these items too.
"""

import argparse
import json
from pathlib import Path

from capability_check import build_sent
from capability_qa import load_qa

CONDITIONS = ("plain", "out", "inout")
OBFUSCATION = {"plain": (False, False), "out": (False, True), "inout": (True, True)}


def build_rows(qa, cipher):
    rows = []
    for index, item in enumerate(qa):
        for condition in CONDITIONS:
            in_obf, out_obf = OBFUSCATION[condition]
            if condition == "plain":
                sent = f"{item['question']}\n\nAnswer concisely with just the fact."
            else:
                sent = build_sent(item["question"], cipher, in_obf, out_obf)
            rows.append(
                {
                    "id": f"{index}-{cipher}-{condition}",
                    "qidx": index,
                    "cipher": cipher,
                    "cond": condition,
                    "out_obf": out_obf,
                    "data_type": f"{cipher}:{condition}",
                    "question": item["question"],
                    "golds": item["golds"],
                    "sent": sent,
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--pool", type=int, default=150, help="the seed-0 draw to subset")
    parser.add_argument("--cipher", default="base64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-q-chars", type=int, default=90)
    parser.add_argument("--out", default="data/outobf_qa_items.jsonl")
    args = parser.parse_args()

    qa = load_qa(args.pool, args.seed, args.max_q_chars)[: args.n]
    rows = build_rows(qa, args.cipher)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"[items] {len(rows)} rows = {len(qa)} questions x {len(CONDITIONS)} conditions")
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
