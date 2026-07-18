"""Reverse-mechanism pilot: does harmful *intent* of a reversed prompt exist at the
prompt-final token position at all, or is reverse decode purely constructed in-generation?

Extract last-token all-layer residual activations from the PROTECTED gemma-3-27b-it on a
few hundred WJ-vanilla prompts (raw data_type labels; pilot noise is fine), conditions
plain + reverse. Train a small per-layer logistic probe on PLAIN, test on plain (in-dist
sanity) and on reverse (transfer). If plain->reverse AUROC sits near 0.5 at every layer,
the reversed intent is not present at prompt-final and reverse drops as a probe venue; if
it is clearly above chance, the input-side probe has signal to read.

Raw WJ labels only. StrongREJECT / XSTest stay held-out (untouched here).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from capability_check import build_sent
from generate_responses import load_wildjailbreak, sample_prompts


def build_prompts(n, seed):
    df = load_wildjailbreak()
    harmful = sample_prompts(df, ["vanilla_harmful"], "vanilla", n, seed)
    benign = sample_prompts(df, ["vanilla_benign"], "vanilla", n, seed)
    rows = []
    for label, group in ((1, harmful), (0, benign)):
        for i, p in enumerate(group):
            rows.append({"label": label, "orig": p["prompt"],
                         "plain": p["prompt"],
                         "reverse": build_sent(p["prompt"], "reverse", in_obf=True, out_obf=False)})
    return rows


def extract_last_token(texts, model, tok, batch_size):
    """Last-token all-layer hidden states -> array [n, L+1, hidden] (float32).

    float32, not float16: Gemma-3's residual stream has massive-activation dimensions
    whose magnitudes exceed float16's ~6.5e4 range, so a float16 cast stores inf and the
    probe's StandardScaler rejects it. bf16->float32 is lossless and cannot overflow."""
    import torch

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    feats = [None] * len(texts)
    t0 = time.time()
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        msgs = [[{"role": "user", "content": texts[i]}] for i in idx]
        enc = tok.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True,
            return_dict=True, padding=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        hs = torch.stack(out.hidden_states, dim=0)  # [L+1, B, seq, hidden]
        # left padding -> the real last token is always column -1
        last = hs[:, :, -1, :].permute(1, 0, 2)  # [B, L+1, hidden]
        for b, i in enumerate(idx):
            feats[i] = last[b].float().cpu().numpy()
        done = min(start + batch_size, len(order))
        print(f"  extracted {done}/{len(order)}  {done/(time.time()-t0):.2f}/s", flush=True)
    return np.stack(feats)  # [n, L+1, hidden]


def probe_per_layer(X_plain, y, X_rev, seed):
    """Per-layer logistic probe trained on plain (5-fold CV for the plain in-dist AUROC),
    full-plain-fit scored on reverse for transfer. Returns list of (layer, auc_plain,
    auc_rev)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    n_layers = X_plain.shape[1]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    results = []
    for L in range(n_layers):
        Xp = X_plain[:, L, :].astype(np.float32)
        Xr = X_rev[:, L, :].astype(np.float32)
        # plain in-dist: cross-validated (no train/test leak)
        oof = np.zeros(len(y))
        for tr, te in skf.split(Xp, y):
            sc = StandardScaler().fit(Xp[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(sc.transform(Xp[tr]), y[tr])
            oof[te] = clf.decision_function(sc.transform(Xp[te]))
        auc_plain = roc_auc_score(y, oof)
        # transfer: fit on ALL plain, score reverse
        sc = StandardScaler().fit(Xp)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xp), y)
        auc_rev = roc_auc_score(y, clf.decision_function(sc.transform(Xr)))
        results.append((L, auc_plain, auc_rev))
        print(f"  layer {L:2d}: plain AUROC={auc_plain:.3f}  plain->reverse AUROC={auc_rev:.3f}",
              flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-27b-it")
    ap.add_argument("--n", type=int, default=300, help="prompts per class")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/reverse_pilot.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reprobe", action="store_true",
                    help="load cached .feats.npz and re-run the probe only (no model, CPU)")
    args = ap.parse_args()

    n = args.limit or args.n
    rows = build_prompts(n, args.seed)
    y = np.array([r["label"] for r in rows])
    print(f"[prompts] {len(rows)} = {int(y.sum())} harmful + {int((1-y).sum())} benign", flush=True)

    feat_path = Path(args.out).with_suffix(".feats.npz")
    if args.reprobe and feat_path.exists():
        print(f"[reprobe] loading cached features {feat_path} (no model load)", flush=True)
        d = np.load(feat_path)
        X_plain, X_rev, y = d["X_plain"], d["X_rev"], d["y"]
    else:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForImageTextToText,
            AutoTokenizer,
            set_seed,
        )

        set_seed(args.seed)
        tok = AutoTokenizer.from_pretrained(args.model)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model, last = None, None
        for cls in (AutoModelForCausalLM, AutoModelForImageTextToText):
            try:
                model = cls.from_pretrained(args.model, dtype=dtype, device_map="auto")
                print(f"[load] {cls.__name__} -> {type(model).__name__}", flush=True)
                break
            except Exception as e:  # noqa: BLE001 -- real boundary: Auto-class mapping
                last = e
        if model is None:
            raise RuntimeError(f"could not load {args.model}: {last}")
        model.eval()

        print("[extract] plain", flush=True)
        X_plain = extract_last_token([r["plain"] for r in rows], model, tok, args.batch_size)
        print("[extract] reverse", flush=True)
        X_rev = extract_last_token([r["reverse"] for r in rows], model, tok, args.batch_size)
        np.savez(feat_path, X_plain=X_plain, X_rev=X_rev, y=y)
        print(f"[extract] cached features -> {feat_path}", flush=True)

    print("[probe] per-layer plain-train", flush=True)
    results = probe_per_layer(X_plain, y, X_rev, args.seed)
    best = max(results, key=lambda r: r[2])
    print(f"\n[best transfer] layer {best[0]}: plain->reverse AUROC={best[2]:.3f} "
          f"(plain in-dist {best[1]:.3f})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"n_per_class": int(y.sum()), "model": args.model, "seed": args.seed,
         "layers": [{"layer": L, "auc_plain": ap_, "auc_reverse": ar_}
                    for L, ap_, ar_ in results],
         "best_transfer_layer": best[0], "best_auc_reverse": best[2]}, indent=2))
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
