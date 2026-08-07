"""Grade the frontier-model obfuscated-output QA answers (CPU).

The 27B arm is graded inside `capability_qa.py`; these answers come from Claude models
driven as subagents, so they arrive as {id, response} files and are graded here with the
same WebQuestions normalisation and the same base64 decoder.
"""

import argparse
import json
from pathlib import Path

from capability_check import CIPHERS
from capability_qa import is_correct


def score(answers_path, items):
    rows = [json.loads(line) for line in Path(answers_path).open() if line.strip()]
    decode = CIPHERS["base64"]["dec"]
    graded = []
    for row in rows:
        item = items[row["id"]]
        raw = row["response"].strip()
        if item["out_obf"]:
            answer, decode_ok = decode(raw)
        else:
            answer, decode_ok = raw, True
        graded.append(
            {
                **item,
                "answer_raw": raw,
                "answer": answer,
                "decode_ok": decode_ok,
                "correct": is_correct(answer, item["golds"]),
            }
        )
    return graded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/outobf_qa_items.jsonl")
    parser.add_argument("--answers", nargs="+", required=True, help="model=path pairs")
    parser.add_argument("--out", default="data/outobf_qa_frontier.jsonl")
    args = parser.parse_args()

    items = {
        row["id"]: row
        for row in (
            json.loads(line) for line in Path(args.items).open() if line.strip()
        )
    }
    everything = []
    for pair in args.answers:
        model, path = pair.split("=", 1)
        graded = score(path, items)
        for row in graded:
            row["model"] = model
        everything.extend(graded)
        n = len(graded)
        ok = sum(row["decode_ok"] for row in graded) / n
        acc = sum(row["correct"] for row in graded) / n
        cond = graded[0]["cond"]
        print(f"  {model:22s} {cond:6s} n={n} decode_ok={ok:.0%} acc={acc:.0%}")

    output = Path(args.out)
    with output.open("w") as handle:
        for row in everything:
            handle.write(json.dumps(row) + "\n")
    print(f"[done] {output}")


if __name__ == "__main__":
    main()
