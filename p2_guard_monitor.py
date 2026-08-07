"""P2 guard comparator: Qwen3Guard-Gen-8B over the reader's response prefixes (GPU).

Qwen3Guard is the strongest evaluated open guard on plain English in C4 and the
only neural comparator P2 runs -- ShieldGemma and Llama Guard 4 add cost without
changing the contrast, and WildGuard's fixed decision cannot be calibrated to a
common operating point.

Its template moderates the **last assistant turn** when given a user+assistant
conversation, and the verdict still begins at the same `Safety:` decision
position, so the continuous score is P('Unsafe') over the three verdict branches
exactly as in `phase1/score_modern_guards.py`.

Prefixes are decoded from the protected reader's response token ids, so a horizon
k means the same thing here as it does for the probe. A generative guard cannot
produce a running score, so it is evaluated on a k grid; the probe is compared to
it on that same grid.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from p2_text_monitor import K_GRID, READER_MODEL, load_npz, prefix_texts
from phase1.score_modern_guards import DECISION_PREFIX, QWEN_MODEL, label_token_ids
from probe_prompt import file_sha256


def load_guard(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokeniser = AutoTokenizer.from_pretrained(model_path)
    tokeniser.padding_side = "left"
    ids = label_token_ids(tokeniser)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, device_map="auto")
    model.eval()
    return model, tokeniser, ids


def score_prefixes(conversations, model, tokeniser, ids, batch_size):
    import torch

    unsafe = np.empty(len(conversations), dtype=np.float32)
    order = sorted(
        range(len(conversations)),
        key=lambda index: len(conversations[index][0]) + len(conversations[index][1]),
    )
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        texts = [
            tokeniser.apply_chat_template(
                [
                    {"role": "user", "content": conversations[index][0]},
                    {"role": "assistant", "content": conversations[index][1]},
                ],
                tokenize=False,
            )
            + DECISION_PREFIX
            for index in batch
        ]
        encoded = tokeniser(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        with torch.no_grad():
            # Only the decision position is read, so keep one row of the head:
            # the full [batch, tokens, vocab] tensor is what fails on a MIG slice.
            logits = model(**encoded, logits_to_keep=1).logits[:, -1, ids]
        probabilities = torch.softmax(logits.float(), dim=-1)
        for index, probability in zip(batch, probabilities.tolist()):
            unsafe[index] = probability[1]
        done = min(start + batch_size, len(order))
        if start % (batch_size * 50) == 0 or done == len(order):
            print(f"  qwen3guard {done}/{len(order)}", flush=True)
    return unsafe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="data/p2_latency_scores.npz")
    parser.add_argument("--judged", default="data/p2_judged.jsonl")
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--tokenizer", default=READER_MODEL)
    parser.add_argument("--out", default="data/p2_guard_monitor.npz")
    parser.add_argument("--request-field", default="prompt", choices=("prompt", "prompt_sent"),
                        help="prompt = plaintext request (the original artefact); "
                             "prompt_sent = the request the reader actually received")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="first N rows (smoke)")
    args = parser.parse_args()

    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    scores = load_npz(Path(args.scores))
    judged = {
        row["id"]: row
        for row in (
            json.loads(line) for line in Path(args.judged).open() if line.strip()
        )
    }
    row_ids = [str(value) for value in scores["ids"]]
    missing = [value for value in row_ids if value not in judged]
    if missing:
        raise ValueError(f"{len(missing)} scored rows absent from {args.judged}")
    # Which request the guard sees. `prompt` (default) is the plaintext request, as the
    # judge is given; `prompt_sent` is what the protected reader actually received, which
    # is the matched comparison in ciphered conditions. They are identical in plain.
    prompts = [judged[value][args.request_field] for value in row_ids]

    token_ids = scores["response_token_ids"]
    lengths = scores["response_length"]
    if args.limit:
        token_ids = token_ids[: args.limit]
        lengths = lengths[: args.limit]
        prompts = prompts[: args.limit]
        row_ids = row_ids[: args.limit]
    grid = [k for k in K_GRID if k <= token_ids.shape[1] and k > 0]

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model, guard_tokeniser, label_ids = load_guard(args.model)
    qwen3guard = np.empty((len(lengths), len(grid)), dtype=np.float32)
    for column, k in enumerate(grid):
        texts = prefix_texts(token_ids, lengths, k, tokenizer)
        print(f"[score] k={k} rows={len(texts)}", flush=True)
        qwen3guard[:, column] = score_prefixes(
            list(zip(prompts, texts)), model, guard_tokeniser, label_ids, args.batch_size
        )

    metadata = {
        "monitor": args.model,
        "score": "softmax P('Unsafe') over the three verdict branches at 'Safety:'",
        "moderates": f"the last assistant turn, given judged['{args.request_field}']",
        "request_field": args.request_field,
        "prefixes": "decoded from the protected reader's response token ids",
        "tokenizer": args.tokenizer,
        "k_grid": grid,
        "k_zero": "not scored; an empty assistant turn is not a response",
        "scores_sha256": file_sha256(Path(args.scores)),
        "batch_size": args.batch_size,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=np.asarray(row_ids),
        k_grid=np.asarray(grid),
        qwen3guard=qwen3guard,
        guard_monitor_metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
