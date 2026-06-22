"""Score judged exchanges with a trained streaming probe, forward-only (no cache).

For eval and calibration we do not store activations (see RESEARCH_LOG 2026-06-22):
each exchange is run teacher-forced through the protected Gemma 3 12B once, the probe
is applied per response token, EMA-smoothed, and reduced to a single per-exchange
detection score (max EMA). Only the scalar is kept, so any FPR operating point can be
read off the score distribution later.

Uses the same model load and response-span logic as extract_activations.py, so the
all-layer concatenation order matches the cache the probe was trained on.
"""

import argparse
import json
from pathlib import Path

import torch

from extract_activations import load_model, response_span
from train_probe import ema_max


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="judged jsonl (eval) or wildchat jsonl (calib)")
    ap.add_argument("--probe", default="data/probe.pt")
    ap.add_argument("--model-id", default="google/gemma-3-12b-it")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.0, help="override EMA decay; 0 -> use probe's")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    probe = torch.load(args.probe, map_location="cpu")
    D, alpha = probe["D"], (args.alpha if args.alpha > 0 else probe["alpha"])

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = load_model(args.model_id)
    model.eval()
    device = model.device

    n_layerp1 = model.config.num_hidden_layers + 1
    hidden = model.config.hidden_size
    assert n_layerp1 * hidden == D, f"probe D={D} != (L+1)*hidden={n_layerp1}*{hidden}"
    W2 = probe["W"].view(n_layerp1, hidden).to(device).float()   # [L+1, hidden]
    b = float(probe["b"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            if line.strip():
                done.add(json.loads(line)["id"])
    rows = [r for r in rows if r["id"] not in done]
    print(f"[score] {len(done)} done, {len(rows)} to go, alpha={alpha:.4f}", flush=True)

    spans = [response_span(tok, r["prompt"], r["response"]) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: len(spans[i][0]))

    with out_path.open("a") as out:
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            enc = tok.pad({"input_ids": [spans[i][0] for i in idx]},
                          padding=True, return_tensors="pt").to(device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
                # per-token logit = sum_l W2[l] . hidden_l + b, accumulated in float32
                logits_all = torch.full(enc["input_ids"].shape, b, device=device, dtype=torch.float32)
                for l in range(n_layerp1):
                    logits_all += (hs[l].float() * W2[l]).sum(-1)
            # gather response-token logits per item, then batched EMA-max
            resp = [logits_all[b_, spans[i][1]:spans[i][2]] for b_, i in enumerate(idx)]
            lengths = torch.tensor([t.shape[0] for t in resp], device=device)
            T = int(lengths.max())
            padded = torch.zeros(len(idx), T, device=device)
            for b_, t in enumerate(resp):
                padded[b_, : t.shape[0]] = t
            scores = ema_max(padded, lengths, alpha).cpu()
            for j, i in enumerate(idx):
                r = rows[i]
                out.write(json.dumps({"id": r["id"], "data_type": r.get("data_type"),
                                      "label": r.get("harmful"), "n_resp": spans[i][2] - spans[i][1],
                                      "score": float(scores[j])}) + "\n")
            out.flush()
            print(f"  scored {min(start + args.batch_size, len(order))}/{len(order)}", flush=True)
    print(f"[done] -> {out_path}")


if __name__ == "__main__":
    main()
