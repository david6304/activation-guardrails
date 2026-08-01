"""Cut generated responses to a shorter token horizon, for the truncation diagnostic.

P2 generates at 512 tokens, and 97-98% of plain responses hit that cap. The
latency curves are unaffected -- they are read over k<=512 and every score there
is observed -- but the *label* is not: a response judged benign-non-refusal at
token 512 may have turned harmful at token 600, and would then wrongly enter the
calibration-negative class that the operating point depends on.

So generate the same prompts at 1024 and judge twice: the full response, and the
512-token prefix this produces. If the strata agree, the 512 cap is safe to keep.

Re-tokenising the decoded text is not exactly the original id sequence -- decode
then encode is not the identity -- but the prefix boundary only has to be close
enough to reproduce the label, not to be token-exact.
"""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="data/p2_pilot_plain_1024.jsonl")
    parser.add_argument("--out", default="data/p2_pilot_plain_1024_prefix512.jsonl")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    rows = [json.loads(line) for line in Path(args.inp).open()]
    if not rows:
        raise ValueError(f"{args.inp} is empty")
    model = args.model or rows[0]["generator"]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)

    cut = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as out:
        for row in rows:
            ids = tokenizer(row["response"], add_special_tokens=False)["input_ids"]
            if len(ids) > args.max_tokens:
                row["response"] = tokenizer.decode(
                    ids[: args.max_tokens], skip_special_tokens=True
                )
                cut += 1
            row["n_response_tokens"] = min(len(ids), args.max_tokens)
            row["truncated"] = len(ids) > args.max_tokens
            out.write(json.dumps(row) + "\n")
    print(f"[done] {output_path}: {len(rows)} rows, {cut} cut to {args.max_tokens}")


if __name__ == "__main__":
    main()
