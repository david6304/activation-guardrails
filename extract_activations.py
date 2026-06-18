"""Cache all-layer residual-stream activations for the response tokens of each
judged exchange, from the protected Gemma 3 12B. These are the features the
streaming probe trains/evaluates on.

Teacher-forced: prompt+response go through one forward pass (no generation); we
keep the hidden states of the *response* tokens only (the streaming probe scores
the model's own output, so prompt activations aren't needed). Per token we concat
all hidden_states entries (embeddings + every layer) -> (n_layers+1)*hidden_size.

Output: one float16 .npy per exchange (data/acts/<id>.npy, shape [n_resp, concat])
plus a manifest jsonl row (id, label, data_type, n_resp, path). Skip-if-exists, so
the job resumes after a timeout.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_model(model_id):
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    last = None
    for cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            model = cls.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
            print(f"[load] {cls.__name__} -> {type(model).__name__}")
            return model
        except Exception as e:  # noqa: BLE001 -- real boundary: Auto-class mapping
            last = e
    raise RuntimeError(f"could not load {model_id}: {last}")


def response_span(tok, prompt, response):
    """Token ids of the full exchange and the [start, end) span of the response.

    Work from the rendered text (the string prefix is guaranteed; token prefixes are
    not, due to BPE merges at the header/content boundary) and locate the response by
    character offset."""
    prefix_text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True)
    full_text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
        tokenize=False, add_generation_prompt=False)
    assert full_text.startswith(prefix_text), "prompt text is not a prefix of the full exchange"
    enc = tok(full_text, add_special_tokens=False, return_offsets_mapping=True)
    cut = len(prefix_text)
    start = next(j for j, (s, _) in enumerate(enc["offset_mapping"]) if s >= cut)
    return enc["input_ids"], start, len(enc["input_ids"])


def extract(rows, model, tok, out_dir, manifest_path, batch_size):
    out_dir.mkdir(parents=True, exist_ok=True)
    spans = [response_span(tok, r["prompt"], r["response"]) for r in rows]

    order = sorted(range(len(rows)), key=lambda i: len(spans[i][0]))
    with open(manifest_path, "a") as mf:
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            if all((out_dir / f"{rows[i]['id']}.npy").exists() for i in idx):
                continue
            batch_ids = [spans[i][0] for i in idx]
            enc = tok.pad({"input_ids": batch_ids}, padding=True, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, use_cache=False)
            hs = torch.stack(out.hidden_states, dim=0)  # [L+1, B, seq, hidden]
            for b, i in enumerate(idx):
                _, s, e = spans[i]
                vec = hs[:, b, s:e, :].permute(1, 0, 2).reshape(e - s, -1)  # [n_resp, concat]
                path = out_dir / f"{rows[i]['id']}.npy"
                np.save(path, vec.to(torch.float16).cpu().numpy())
                mf.write(json.dumps({"id": rows[i]["id"], "label": bool(rows[i]["harmful"]),
                                     "data_type": rows[i]["data_type"], "n_resp": e - s,
                                     "path": str(path)}) + "\n")
            mf.flush()
            print(f"  extracted {min(start + batch_size, len(order))}/{len(order)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="judged jsonl (with harmful labels)")
    ap.add_argument("--model-id", default="google/gemma-3-12b-it")
    ap.add_argument("--out-dir", default="data/acts")
    ap.add_argument("--manifest", default="data/acts_manifest.jsonl")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    rows = []
    with open(args.inp) as f:
        for line in f:
            r = json.loads(line)
            if r.get("harmful") is not None:
                rows.append(r)
            if args.limit and len(rows) >= args.limit:
                break
    print(f"[extract] {len(rows)} labelled exchanges -> {args.out_dir}")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = load_model(args.model_id)
    model.eval()

    extract(rows, model, tok, Path(args.out_dir), Path(args.manifest), args.batch_size)
    print("[done]")


if __name__ == "__main__":
    main()
