"""Train the CC++ streaming probe online: recompute the protected Gemma 3 12B
activations each step instead of reading the cached .npy.

Reading the 1.8 TB activation cache off Lustre ran ~5.5 h/epoch; the 12B forward is
faster and writes nothing. Same SWiM loss / EMA-max scoring as train_probe.py, and
the saved checkpoint is identical (drop+standardise folded into a full-D weight), so
score_probe.py is unchanged. Input is the judged jsonl (prompt/response/harmful), the
same file extract_activations.py consumed.

The massive-activation channel (dim 2339) is kept here -- on-the-fly activations are
bf16, so it does not overflow as it did in the fp16 cache; standardisation handles its
scale, so no dim is dropped.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from extract_activations import load_model, response_span
from train_probe import ema_max, split_rows, swim_loss


def load_rows(path, limit):
    rows = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("harmful") is not None:
                r["label"] = int(bool(r["harmful"]))
                rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def make_batch(tok, spans, idx, device):
    enc = tok.pad({"input_ids": [spans[i][0] for i in idx]},
                  padding=True, return_tensors="pt").to(device)
    return enc["input_ids"], enc["attention_mask"]


def length_bucketed_batches(idx_list, spans, batch_size, bucket_mult=50):
    """Batches of similar-length sequences (less padding -> less wasted forward compute).

    Shuffle, split into megabatches, sort each by length into batches, then shuffle the
    batch order. Re-randomised each call, so batch contents and order vary across epochs
    while padding stays low. Uses the seeded global RNG."""
    order = idx_list[:]
    random.shuffle(order)
    batches, mb = [], batch_size * bucket_mult
    for s in range(0, len(order), mb):
        chunk = sorted(order[s : s + mb], key=lambda j: len(spans[j][0]))
        batches += [chunk[k : k + batch_size] for k in range(0, len(chunk), batch_size)]
    random.shuffle(batches)
    return batches


def token_logits(model, ids, attn, W, b, mean, std):
    """Per-token harmfulness logit [B, seq], grad flowing only to W, b (model frozen).

    Accumulate over layers instead of stacking all L+1 hidden states: peak memory is
    one [B, seq, H] tensor, not [L+1, B, seq, H], so larger batches fit and each step
    moves far less memory."""
    with torch.no_grad():
        hs = model(input_ids=ids, attention_mask=attn,
                   output_hidden_states=True, use_cache=False).hidden_states
    logit = b
    for li, h in enumerate(hs):
        logit = logit + (((h.float() - mean[li]) / std[li]) * W[li]).sum(-1)
    return logit


def gather_resp(logits, spans, idx, device):
    resp = [logits[bi, spans[i][1]:spans[i][2]] for bi, i in enumerate(idx)]
    lengths = torch.tensor([t.shape[0] for t in resp], device=device)
    T = int(lengths.max())
    padded = torch.zeros(len(idx), T, device=device)
    for bi, t in enumerate(resp):
        padded[bi, : t.shape[0]] = t
    return padded, lengths


def compute_stats(model, tok, spans, idx_list, batch_size, device, n_layerp1, hidden):
    """Per-(layer, dim) mean/std over response tokens of a sample of train exchanges."""
    order = sorted(idx_list, key=lambda i: len(spans[i][0]))
    s = torch.zeros(n_layerp1, hidden, dtype=torch.float64, device=device)
    ss = torch.zeros_like(s)
    n, t0 = 0, time.time()
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        ids, attn = make_batch(tok, spans, idx, device)
        with torch.no_grad():
            hs = torch.stack(model(input_ids=ids, attention_mask=attn,
                                   output_hidden_states=True, use_cache=False).hidden_states)
        for bi, i in enumerate(idx):
            v = hs[:, bi, spans[i][1]:spans[i][2], :].double()   # [L+1, n_resp, H]
            s += v.sum(1)
            ss += (v * v).sum(1)
            n += v.shape[1]
        if (start // batch_size) % 20 == 0:
            dt = time.time() - t0
            print(f"  [stats] {min(start + batch_size, len(order))}/{len(order)} exch, "
                  f"{n} tokens, {dt:.0f}s", flush=True)
    mean = (s / n).float()
    std = (ss / n - (s / n) ** 2).clamp_min(1e-12).sqrt().float()
    return mean, std


@torch.no_grad()
def evaluate(model, tok, rows, spans, idx_list, W, b, mean, std, batch_size, device, alpha):
    order = sorted(idx_list, key=lambda i: len(spans[i][0]))
    scores, ys = [], []
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        ids, attn = make_batch(tok, spans, idx, device)
        logits = token_logits(model, ids, attn, W, b, mean, std)
        padded, lengths = gather_resp(logits, spans, idx, device)
        scores.append(ema_max(padded, lengths, alpha).cpu())
        ys.append(torch.tensor([rows[i]["label"] for i in idx], dtype=torch.float32))
    scores = torch.cat(scores).numpy()
    ys = torch.cat(ys).numpy()
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(ys, scores))
    except Exception:  # noqa: BLE001 -- sklearn missing or single-class val
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/judged_train.jsonl")
    ap.add_argument("--model-id", default="google/gemma-3-12b-it")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=10, help="max epochs; early-stops on val")
    ap.add_argument("--patience", type=int, default=3, help="stop after this many evals w/o val gain")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="eval/checkpoint every N steps; 0 = once per epoch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2,
                    help="L2 on W (probe is 188k-dim over ~9k exchanges, so regularise)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--stats-rows", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/probe.pt")
    args = ap.parse_args()

    alpha = args.alpha if args.alpha > 0 else 2.0 / (args.M + 1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_id)
    tok.padding_side = "right"   # token_logits/gather slice response spans on unpadded indices
    model = load_model(args.model_id)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = model.device
    text_cfg = getattr(model.config, "text_config", model.config)   # Gemma 3 is multimodal
    n_layerp1 = text_cfg.num_hidden_layers + 1
    hidden = text_cfg.hidden_size
    D = n_layerp1 * hidden

    rows = load_rows(args.inp, args.limit)
    spans = [response_span(tok, r["prompt"], r["response"]) for r in rows]
    train_rows, val_rows = split_rows(rows, args.val_frac, args.seed)
    row_idx = {id(r): i for i, r in enumerate(rows)}   # split_rows returns the same dicts
    tr_idx = [row_idx[id(r)] for r in train_rows]
    va_idx = [row_idx[id(r)] for r in val_rows]
    n_pos = sum(r["label"] for r in train_rows)
    print(f"[data] train={len(train_rows)} (pos={n_pos}) val={len(val_rows)} D={D} "
          f"M={args.M} tau={args.tau} alpha={alpha:.4f}", flush=True)

    stats_idx = tr_idx[: args.stats_rows] if args.stats_rows else tr_idx
    mean, std = compute_stats(model, tok, spans, stats_idx,
                              args.batch_size, device, n_layerp1, hidden)
    print(f"[stats] from {len(stats_idx)} exchanges", flush=True)

    W = torch.zeros(n_layerp1, hidden, device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.AdamW([{"params": [W], "weight_decay": args.weight_decay},
                             {"params": [b], "weight_decay": 0.0}], lr=args.lr)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    best_auroc, no_improve, gstep = float("-inf"), 0, 0

    def checkpoint(tag):
        """Eval on val; save the probe if it's the best so far; True if patience hit."""
        nonlocal best_auroc, no_improve
        auroc = evaluate(model, tok, rows, spans, va_idx,
                         W, b, mean, std, args.batch_size, device, alpha)
        print(f"[{tag}] val_auroc {auroc:.4f}", flush=True)
        if auroc > best_auroc:
            best_auroc, no_improve = auroc, 0
            W_eff = (W.detach() / std)   # fold standardise into a score_probe-compatible weight
            b_eff = float(b.detach()) - float((W_eff * mean).sum())
            torch.save({"W": W_eff.reshape(-1).cpu(), "b": b_eff, "D": D, "hidden": hidden,
                        "M": args.M, "tau": args.tau, "alpha": alpha, "online": True,
                        "epoch": epoch, "step": gstep, "val_auroc": auroc, "seed": args.seed,
                        "model_id": args.model_id, "config": vars(args)}, out)
            print(f"[saved] best probe -> {out} ({tag}, val_auroc {auroc:.4f})", flush=True)
            return False
        no_improve += 1
        if no_improve >= args.patience:
            print(f"[early stop] no val gain in {args.patience} evals; "
                  f"best val_auroc {best_auroc:.4f}", flush=True)
            return True
        return False

    stop = False
    for epoch in range(args.epochs):
        batches = length_bucketed_batches(tr_idx, spans, args.batch_size)
        n_steps = len(batches)
        running, seen, t0 = 0.0, 0, time.time()
        for step, idx in enumerate(batches):
            ids, attn = make_batch(tok, spans, idx, device)
            logits = token_logits(model, ids, attn, W, b, mean, std)
            padded, lengths = gather_resp(logits, spans, idx, device)
            labels = torch.tensor([rows[i]["label"] for i in idx],
                                  dtype=torch.float32, device=device)
            loss = swim_loss(padded, lengths, labels, args.M, args.tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
            seen += len(idx)
            gstep += 1
            if step % 20 == 0:
                dt = time.time() - t0
                print(f"  epoch {epoch} step {step}/{n_steps} loss {running / seen:.4f} "
                      f"{dt:.0f}s ({seen / dt:.1f} exch/s)", flush=True)
            if args.eval_every and gstep % args.eval_every == 0:
                stop = checkpoint(f"epoch {epoch} step {gstep}")
                if stop:
                    break
        print(f"[epoch {epoch}] train_loss {running / seen:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if not args.eval_every:
            stop = checkpoint(f"epoch {epoch}")
        if stop:
            break


if __name__ == "__main__":
    main()
