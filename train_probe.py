"""Train the CC++ streaming linear probe on cached all-layer activations.

Reads the activation cache (extract_activations.py): one float16 .npy per exchange,
shape [n_resp, (L+1)*hidden], indexed by data/acts_manifest.jsonl. A single linear
layer maps each token's concatenated all-layer activation to a harmfulness logit.

Training follows CC++ (Cunningham et al. 2026, Sec. 5, Eqs. 2-4):
  - SWiM logit smoothing: average the raw logit over a sliding window of M tokens.
  - Softmax-weighted BCE: weight each position's BCE by softmax(z_bar/tau), so the
    loss focuses on the most confidently-harmful positions; positions t < M (without
    a full window) are excluded, except sequences shorter than M which get a single
    averaged prediction.
Inference / model selection uses the EMA of the logit (the paper's deployment-time
substitute for SWiM), scored per exchange as max EMA over response tokens.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def swim_loss(logits, lengths, labels, M, tau):
    """Softmax-weighted BCE over SWiM-smoothed logits (CC++ Eqs. 2-4).

    logits [B, T] raw per-token logits; lengths [B] real response lengths; labels [B].
    """
    B, T = logits.shape
    ar = torch.arange(T, device=logits.device)
    c = torch.cumsum(logits, dim=1)
    cprev = torch.zeros_like(c)
    if T > M:
        cprev[:, M:] = c[:, : T - M]
    zbar_win = (c - cprev) / M                      # window mean, valid for t >= M-1
    cm = c / (ar + 1)                               # cumulative mean, for short seqs
    len_col = lengths.unsqueeze(1)
    pos = ar.unsqueeze(0)
    is_long = len_col >= M
    mask_long = is_long & (pos >= M - 1) & (pos < len_col)
    mask_short = (~is_long) & (pos == len_col - 1)  # single prediction
    valid = mask_long | mask_short
    zbar = torch.where(mask_short, cm, zbar_win)

    w = torch.softmax((zbar / tau).masked_fill(~valid, float("-inf")), dim=1)
    y = labels.unsqueeze(1).expand(-1, T)
    bce = F.binary_cross_entropy_with_logits(zbar, y, reduction="none")
    bce = torch.where(valid, bce, torch.zeros_like(bce))
    return (w * bce).sum(dim=1).mean()


def ema_max(logits, lengths, alpha):
    """Per-exchange detection score: max over response tokens of the EMA-smoothed
    logit. logits [B, T], lengths [B] -> [B]."""
    ema = logits[:, 0].clone()
    best = ema.clone()
    T = logits.shape[1]
    for t in range(1, T):
        ema = alpha * logits[:, t] + (1 - alpha) * ema
        cand = torch.where(t < lengths, ema, torch.full_like(ema, float("-inf")))
        best = torch.maximum(best, cand)
    return best


# Residual-stream dim 2339 is Gemma 3 12B's massive-activation channel (~1e4-1e5);
# in the float16 cache it overflows to inf in the deep layers (RESEARCH_LOG 2026-06-22).
# Drop it from every layer block and standardise the rest before the probe.
HIDDEN = 3840
DROP_CHANNEL = 2339


def keep_dims(D):
    drop = np.arange(DROP_CHANNEL, D, HIDDEN)   # dim 2339 in every (L+1) block
    return np.setdiff1d(np.arange(D), drop)


class ActsDataset(Dataset):
    def __init__(self, rows, keep, mean=None, std=None):
        self.rows = rows
        self.keep = keep
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        a = np.load(r["path"])[:, self.keep].astype(np.float32)   # [n_resp, Dk]
        a = torch.from_numpy(a)
        assert torch.isfinite(a).all(), f"non-finite activation in {r['path']}"
        if self.mean is not None:
            a = (a - self.mean) / self.std
        return a, float(r["label"])


def compute_stats(rows, keep, num_workers):
    """Per-dim mean/std over response tokens of a sample of train exchanges."""
    loader = DataLoader(ActsDataset(rows, keep), batch_size=1,
                        num_workers=num_workers, collate_fn=lambda b: b[0])
    n, s, ss = 0, None, None
    for a, _ in loader:
        a = a.double()
        s = a.sum(0) if s is None else s + a.sum(0)
        ss = (a * a).sum(0) if ss is None else ss + (a * a).sum(0)
        n += a.shape[0]
    mean = s / n
    std = (ss / n - mean**2).clamp_min(1e-12).sqrt()
    return mean.float(), std.float()


def collate(batch):
    acts, labels = zip(*batch)
    lengths = torch.tensor([a.shape[0] for a in acts])
    T, D = int(lengths.max()), acts[0].shape[1]
    X = torch.zeros(len(batch), T, D, dtype=acts[0].dtype)
    for i, a in enumerate(acts):
        X[i, : a.shape[0]] = a
    return X, lengths, torch.tensor(labels, dtype=torch.float32)


def split_rows(rows, val_frac, seed):
    by_label = {0: [], 1: []}
    for r in rows:
        by_label[int(r["label"])].append(r)
    rng = random.Random(seed)
    train, val = [], []
    for lab in (0, 1):
        g = by_label[lab][:]
        rng.shuffle(g)
        n_val = int(round(len(g) * val_frac))
        val += g[:n_val]
        train += g[n_val:]
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


@torch.no_grad()
def evaluate(model, loader, device, alpha):
    model.eval()
    scores, ys = [], []
    for X, lengths, labels in loader:
        logits = model(X.to(device).float()).squeeze(-1)
        scores.append(ema_max(logits, lengths.to(device), alpha).cpu())
        ys.append(labels)
    scores = torch.cat(scores).numpy()
    ys = torch.cat(ys).numpy()
    try:
        from sklearn.metrics import roc_auc_score
        auroc = float(roc_auc_score(ys, scores))
    except Exception:  # noqa: BLE001 -- sklearn missing or single-class val
        auroc = float("nan")
    return auroc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/acts_manifest.jsonl")
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.0,
                    help="EMA decay for inference; 0 -> 2/(M+1) (window centre-of-mass match)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--stats-rows", type=int, default=2000,
                    help="train exchanges sampled to estimate standardisation stats")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="use first N manifest rows (smoke test)")
    ap.add_argument("--out", default="data/probe.pt")
    args = ap.parse_args()

    alpha = args.alpha if args.alpha > 0 else 2.0 / (args.M + 1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    rows = [json.loads(l) for l in open(args.manifest) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    train_rows, val_rows = split_rows(rows, args.val_frac, args.seed)
    D = int(np.load(train_rows[0]["path"]).shape[1])
    keep = keep_dims(D)
    n_pos = sum(int(r["label"]) for r in train_rows)
    print(f"[data] train={len(train_rows)} (pos={n_pos}) val={len(val_rows)} D={D} "
          f"keep={len(keep)} M={args.M} tau={args.tau} alpha={alpha:.4f}", flush=True)

    stats_rows = train_rows[: args.stats_rows] if args.stats_rows else train_rows
    mean, std = compute_stats(stats_rows, keep, args.num_workers)
    print(f"[stats] from {len(stats_rows)} exchanges", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.nn.Linear(len(keep), 1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(ActsDataset(train_rows, keep, mean, std), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers, collate_fn=collate)
    val_loader = DataLoader(ActsDataset(val_rows, keep, mean, std), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, collate_fn=collate)

    for epoch in range(args.epochs):
        model.train()
        running, seen = 0.0, 0
        for step, (X, lengths, labels) in enumerate(train_loader):
            logits = model(X.to(device).float()).squeeze(-1)
            loss = swim_loss(logits, lengths.to(device), labels.to(device), args.M, args.tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * len(labels)
            seen += len(labels)
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step} loss {running / seen:.4f}", flush=True)
        auroc = evaluate(model, val_loader, device, alpha)
        print(f"[epoch {epoch}] train_loss {running / seen:.4f} val_auroc {auroc:.4f}", flush=True)

    # Fold drop+standardise into a full-D weight so score_probe needs no change:
    # W.(x-mean)/std + b == (W/std).x + (b - sum(W*mean/std)), zero on dropped dims.
    W_kept = model.weight.detach().cpu().squeeze(0).float() / std
    b_eff = float(model.bias.detach().cpu()) - float((W_kept * mean).sum())
    W_full = torch.zeros(D)
    W_full[torch.from_numpy(keep)] = W_kept

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"W": W_full, "b": b_eff, "D": D, "M": args.M, "tau": args.tau,
                "alpha": alpha, "drop_channel": DROP_CHANNEL, "hidden": HIDDEN,
                "val_auroc": auroc, "seed": args.seed}, out)
    print(f"[done] saved probe -> {out}")


if __name__ == "__main__":
    main()
