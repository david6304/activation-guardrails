"""Grade the SQuAD cipher-capability answers with the standard EM/F1 scorer (CPU).

Handles both shapes: {id, response} files written by hosted-model agents, and the
generation jsonl from capability_qa.py (which carries `answer_raw`). Output-obfuscated
rows are graded on the DECODED answer, as in RESULTS.md section 11.
"""

import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path

import base64

_PUNC = str.maketrans("", "", string.punctuation)


def normalise(text):
    text = text.lower().translate(_PUNC)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def em(pred, golds):
    return float(any(normalise(pred) == normalise(g) for g in golds))


def f1(pred, golds):
    best = 0.0
    p = normalise(pred).split()
    for g in golds:
        gt = normalise(g).split()
        shared = sum((Counter(p) & Counter(gt)).values())
        if shared:
            precision, recall = shared / len(p), shared / len(gt)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def dec_b64(text):
    """As capability_check.dec_b64 but with a 4-character floor, not 8.

    SQuAD spans are short: "6.7" encodes to `Ni43`, four characters, which the shared
    decoder's floor scores as a decode failure. Longest valid decode still wins, so a
    lower floor can only add candidates shorter than any existing winner.
    """
    best = ""
    for candidate in re.findall(r"[A-Za-z0-9+/=\s]{4,}", text):
        s = re.sub(r"\s+", "", candidate)
        for pad in range(4):
            try:
                d = base64.b64decode(s + "=" * pad).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                continue
            if len(d) > len(best):
                best = d
            break
    return best, bool(best)


def grade(raw, item):
    answer, decode_ok = dec_b64(raw) if item["out_obf"] else (raw, True)
    return {**item, "answer_raw": raw, "answer": answer, "decode_ok": decode_ok,
            "em": em(answer, item["golds"]), "f1": f1(answer, item["golds"])}


def load_answers(path):
    """-> {id: raw answer}; accepts agent {id,response} or capability_qa rows."""
    out = {}
    for line in Path(path).open():
        if line.strip():
            row = json.loads(line)
            out[row["id"]] = (row.get("response") or row.get("answer_raw") or "").strip()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/squad_cipher_items.jsonl")
    parser.add_argument("--answers", nargs="+", required=True, help="model=path pairs")
    parser.add_argument("--out", default="data/squad_cipher_graded.jsonl")
    args = parser.parse_args()

    items = {r["id"]: r for r in
             (json.loads(line) for line in Path(args.items).open() if line.strip())}

    everything = []
    for pair in args.answers:
        model, path = pair.split("=", 1)
        answers = load_answers(path)
        missing = [i for i in answers if i not in items]
        if missing:
            raise ValueError(f"{path}: unknown ids {missing[:3]}")
        for item_id, raw in answers.items():
            everything.append({**grade(raw, items[item_id]), "model": model, "src": path})

    with Path(args.out).open("w") as handle:
        for row in everything:
            handle.write(json.dumps(row) + "\n")

    print(f"{'model':16s} {'cond':6s} {'n':>3s}  decode_ok    EM     F1")
    key = lambda r: (r["model"], r["cond"])  # noqa: E731
    for (model, cond), rows in sorted(
        {k: [r for r in everything if key(r) == k] for k in map(key, everything)}.items()
    ):
        n = len(rows)
        print(f"{model:16s} {cond:6s} {n:3d}  {sum(r['decode_ok'] for r in rows)/n:8.0%} "
              f"{sum(r['em'] for r in rows)/n:6.0%} {sum(r['f1'] for r in rows)/n:6.0%}")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
