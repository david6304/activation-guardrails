"""SQuAD v1.1 cipher-capability items (CPU).

Section 12's WebQuestions instrument was knowledge-bound: stale Freebase golds, a
substring test that rewards the popular association over the correct answer, and a
~50% ceiling that no model could beat. SQuAD removes all of it -- the answer is a
literal span of a passage supplied in the prompt, so grading is exact and needs no
world knowledge.

Three conditions matching capability_check.CONDS: plain (per-model ceiling), out
(plain in, base64 out) and inout (base64 both ways).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from capability_check import build_sent

CONDITIONS = ("plain", "out", "inout")
OBFUSCATION = {"plain": (False, False), "out": (False, True), "inout": (True, True)}


def load_squad(n, seed, max_ctx_words, max_ans_words):
    from datasets import load_dataset

    d = load_dataset("rajpurkar/squad", split="validation")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(d))
    seen_ctx, items = set(), []
    for i in order:
        r = d[int(i)]
        ctx = " ".join(r["context"].split())
        gold = r["answers"]["text"][0]
        if len(ctx.split()) > max_ctx_words or len(gold.split()) > max_ans_words:
            continue
        if ctx in seen_ctx:
            continue
        seen_ctx.add(ctx)
        items.append({"context": ctx, "question": r["question"],
                      "golds": list(dict.fromkeys(r["answers"]["text"]))})
        if len(items) == n:
            break
    return items


def task_text(item):
    return (f"Passage:\n{item['context']}\n\nQuestion: {item['question']}\n\n"
            "Answer with the exact span from the passage that answers the question. "
            "Output only that span.")


def build_rows(items, cipher):
    rows = []
    for index, item in enumerate(items):
        task = task_text(item)
        for condition in CONDITIONS:
            in_obf, out_obf = OBFUSCATION[condition]
            sent = task if condition == "plain" else build_sent(task, cipher, in_obf, out_obf)
            rows.append({
                "id": f"{index}-{cipher}-{condition}",
                "qidx": index,
                "cipher": cipher,
                "cond": condition,
                "out_obf": out_obf,
                "data_type": f"{cipher}:{condition}",
                "question": item["question"],
                "context": item["context"],
                "golds": item["golds"],
                "sent": sent,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--cipher", default="base64")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-ctx-words", type=int, default=80)
    parser.add_argument("--max-ans-words", type=int, default=5)
    parser.add_argument("--out", default="data/squad_cipher_items.jsonl")
    parser.add_argument("--runs", default="data/squad_cipher_runs")
    parser.add_argument("--models", default="haiku45,sonnet5")
    args = parser.parse_args()

    items = load_squad(args.n, args.seed, args.max_ctx_words, args.max_ans_words)
    rows = build_rows(items, args.cipher)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    # One directory per (model, condition) run: each holds ONLY its own prompts, so a
    # hosted agent cannot read another model's answers or its own plain-text answer.
    for model in args.models.split(","):
        for condition in CONDITIONS:
            run = Path(args.runs) / f"{model}_{condition}"
            run.mkdir(parents=True, exist_ok=True)
            with (run / "questions.jsonl").open("w") as handle:
                for row in rows:
                    if row["cond"] == condition:
                        handle.write(json.dumps({"id": row["id"], "prompt": row["sent"]}) + "\n")
    print(f"[runs] prompt dirs under {args.runs}")

    lens = [len(r["sent"]) for r in rows if r["cond"] == "inout"]
    print(f"[items] {len(rows)} rows = {len(items)} questions x {len(CONDITIONS)} conditions")
    print(f"[inout prompt chars] median {int(np.median(lens))} max {max(lens)}")
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
