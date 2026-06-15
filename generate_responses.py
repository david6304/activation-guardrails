"""Generate responses to WildJailbreak prompts with the abliterated Gemma 3.

The (prompt, response) exchanges produced here are what the protected model's
probe is trained/evaluated on. The JSONL output is the reproducible source of
truth -- the activation cache (extract_activations.py) is built from it, so we
do not need byte-identical regeneration.

Same script serves train (vanilla_* / `vanilla` column) and eval
(adversarial_* / `adversarial` column) via --data-types / --prompt-col.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Heuristic only, to report whether the abliterated model is actually complying
# (low refusal rate on harmful prompts = good). Not used for labelling.
REFUSAL_PREFIXES = (
    "i can't", "i cannot", "i can not", "i'm sorry", "i am sorry", "i apologize",
    "i'm not able", "i am not able", "i won't", "i will not", "as an ai",
    "sorry, but", "i'm unable", "i am unable",
)


def load_wildjailbreak():
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files("allenai/wildjailbreak", repo_type="dataset")
             if f.endswith(".tsv") and "train" in f]
    path = hf_hub_download("allenai/wildjailbreak", files[0], repo_type="dataset")
    return pd.read_csv(path, sep="\t", dtype=str, na_filter=False)


def sample_prompts(df, data_types, prompt_col, n_per_type, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for dt in data_types:
        sub = df[(df["data_type"] == dt) & (df[prompt_col].str.len() > 0)].reset_index(drop=True)
        take = min(n_per_type, len(sub))
        idx = rng.choice(len(sub), size=take, replace=False)
        for j in idx:
            rows.append({"data_type": dt, "prompt": sub.iloc[int(j)][prompt_col]})
        print(f"  {dt}: {len(sub)} available, sampled {take}")
    return rows


def looks_like_refusal(text):
    t = text.strip().lower()
    return any(t.startswith(p) for p in REFUSAL_PREFIXES)


def generate(rows, model_id, max_new_tokens, temperature, top_p, batch_size, seed):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
        set_seed,
    )

    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model, last = None, None
    for cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            model = cls.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
            print(f"[load] {cls.__name__} -> {type(model).__name__}")
            break
        except Exception as e:  # noqa: BLE001 -- real boundary: Auto-class mapping
            last = e
    if model is None:
        raise RuntimeError(f"could not load {model_id}: {last}")
    model.eval()

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        msgs = [[{"role": "user", "content": r["prompt"]}] for r in batch]
        inputs = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=top_p, pad_token_id=tok.pad_token_id,
            )
        new = gen[:, inputs["input_ids"].shape[1]:]
        for r, ids in zip(batch, new):
            n_tok = int((ids != tok.pad_token_id).sum())
            r["response"] = tok.decode(ids, skip_special_tokens=True)
            r["n_response_tokens"] = n_tok
            r["truncated"] = n_tok >= max_new_tokens
        print(f"  generated {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return rows


def summarise(rows):
    df = pd.DataFrame(rows)
    print("\n[summary]")
    for dt, g in df.groupby("data_type"):
        refusal = g["response"].map(looks_like_refusal).mean()
        lens = g["n_response_tokens"]
        trunc = g["truncated"].mean()
        print(f"  {dt}: n={len(g)} refusal={refusal:.1%} "
              f"resp_tokens median={lens.median():.0f} p95={lens.quantile(.95):.0f} "
              f"truncated={trunc:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-types", default="vanilla_harmful,vanilla_benign")
    ap.add_argument("--prompt-col", default="vanilla", choices=["vanilla", "adversarial"])
    ap.add_argument("--n-per-type", type=int, default=500)
    ap.add_argument("--model", default="/home/s2296274/models/gemma-3-12b-it-heretic")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="sample + write placeholder responses, no model load (local test)")
    args = ap.parse_args()

    data_types = args.data_types.split(",")
    print(f"[sample] data_types={data_types} prompt_col={args.prompt_col!r} n_per_type={args.n_per_type}")
    df = load_wildjailbreak()
    rows = sample_prompts(df, data_types, args.prompt_col, args.n_per_type, args.seed)

    if args.dry_run:
        for r in rows:
            r.update(response="[DRY RUN]", n_response_tokens=0, truncated=False)
    else:
        rows = generate(rows, args.model, args.max_new_tokens, args.temperature,
                        args.top_p, args.batch_size, args.seed)
        summarise(rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"id": i, **r}) + "\n")
    print(f"\n[done] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
