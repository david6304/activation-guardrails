"""Confirm Gemma 3 config, measure per-token activation cost, estimate the
training activation-cache size. No caching, no training -- this only answers
"is the all-layer cache affordable, and what are the real dimensions?".

Run on a short GPU allocation (Interactive). Needs CUDA only for the forward pass;
the dataset schema inspection runs anywhere.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from transformers import AutoConfig, AutoTokenizer

SEED = 0


def load_model(model_id):
    # Gemma 3 12B-IT is a multimodal checkpoint, so AutoModelForCausalLM may not
    # map it. Try the text-LM class, fall back to the image-text-to-text class.
    # We only need hidden states, not generation. Print whichever resolved.
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    last = None
    for cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
        try:
            model = cls.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
            print(f"[load] resolved via {cls.__name__} -> {type(model).__name__}")
            return model
        except Exception as e:  # noqa: BLE001 -- real boundary: Auto-class mapping
            last = e
    raise RuntimeError(f"could not load {model_id}: {last}")


def load_wildjailbreak():
    # WildJailbreak ships as TSV; the datasets Arrow builder mis-infers a column
    # as double and fails on long prompt strings. Read the raw TSV as all-strings.
    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files("allenai/wildjailbreak", repo_type="dataset")
             if f.endswith(".tsv") and "train" in f]
    path = hf_hub_download("allenai/wildjailbreak", files[0], repo_type="dataset")
    df = pd.read_csv(path, sep="\t", dtype=str, na_filter=False)
    print(f"\n[WildJailbreak] {len(df)} rows; columns: {df.columns.tolist()}")
    if "data_type" in df.columns:
        print("  data_type counts:", df["data_type"].value_counts().to_dict())
    for _, row in df.head(2).iterrows():
        print("  sample row:", {k: (str(v)[:120] + "...") for k, v in row.items()})
    return df


def prompt_token_lengths(df, tokenizer, prompt_field, n_sample=512):
    # Templated length of just the user turn, to ground the mean-tokens estimate.
    idx = np.random.default_rng(SEED).choice(len(df), size=min(n_sample, len(df)), replace=False)
    col = df[prompt_field].to_numpy()
    lengths = []
    for i in idx:
        text = col[i]
        if not text:
            continue
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )
        lengths.append(len(ids))
    lengths = np.array(lengths)
    print(
        f"\n[prompt tokens, field={prompt_field!r}] n={len(lengths)} "
        f"mean={lengths.mean():.0f} median={np.median(lengths):.0f} "
        f"p95={np.percentile(lengths, 95):.0f} max={lengths.max()}"
    )
    return lengths


def measure_per_token_cost(model_id):
    cfg = AutoConfig.from_pretrained(model_id)
    # Gemma 3 12B/4B is a multimodal config; the text stack is usually under .text_config.
    text_cfg = getattr(cfg, "text_config", cfg)
    n_layers = getattr(text_cfg, "num_hidden_layers")
    d_model = getattr(text_cfg, "hidden_size")
    print(f"\n[config] {model_id}: num_hidden_layers={n_layers} hidden_size={d_model}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = load_model(model_id)
    model.eval()

    # One real exchange through the model; grab all-layer residual stream.
    msgs = [
        {"role": "user", "content": "Explain how a refrigerator works."},
        {"role": "assistant", "content": "A refrigerator moves heat from inside to outside using a refrigerant cycle."},
    ]
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)

    hs = out.hidden_states  # tuple len (n_layers + 1): embeddings + each layer
    seq = inputs["input_ids"].shape[1]
    concat_dim = len(hs) * hs[0].shape[-1]
    dtype_bytes = hs[0].element_size()
    bytes_per_token = concat_dim * dtype_bytes
    print(
        f"[forward] hidden_states entries={len(hs)} (incl. embeddings), seq_len={seq}, "
        f"per-entry hidden={hs[0].shape[-1]}, dtype={hs[0].dtype}"
    )
    print(
        f"[per-token] concat_dim={concat_dim} bytes/token={bytes_per_token} "
        f"({bytes_per_token / 1024:.1f} KiB)"
    )
    return bytes_per_token, tokenizer


def estimate_cache(bytes_per_token, mean_tokens, exchange_counts):
    print(f"\n[cache estimate] mean_tokens={mean_tokens}")
    for n in exchange_counts:
        tb = n * mean_tokens * bytes_per_token / 1024**4
        print(f"  {n:>7,} exchanges -> {tb:.2f} TB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="google/gemma-3-12b-it")
    ap.add_argument("--prompt-field", default="vanilla",
                    help="WildJailbreak column holding the user prompt (verify from schema print)")
    ap.add_argument("--response-tokens", type=int, default=256,
                    help="assumed mean response length; combined with measured prompt length")
    ap.add_argument("--exchange-counts", type=int, nargs="+", default=[5000, 10000, 30000])
    ap.add_argument("--skip-data", action="store_true", help="only do config + forward pass")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    bytes_per_token, tokenizer = measure_per_token_cost(args.model_id)

    mean_tokens = args.response_tokens + 64  # fallback prompt estimate if data skipped
    if not args.skip_data:
        df = load_wildjailbreak()
        if args.prompt_field in df.columns:
            plen = prompt_token_lengths(df, tokenizer, args.prompt_field)
            mean_tokens = int(np.median(plen) + args.response_tokens)
        else:
            print(f"\n[warn] field {args.prompt_field!r} not in columns; using fallback prompt length")

    estimate_cache(bytes_per_token, mean_tokens, args.exchange_counts)


if __name__ == "__main__":
    main()
