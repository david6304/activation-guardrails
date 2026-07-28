"""The one place P2 reimplements a frozen detector, so the one place it is tested.

p2_extract_latency.score_positions rescores the all-layer probe and the
layer-averaged centroid from per-layer dot products, because materialising
[positions, layers, hidden] for a 512-token response is not affordable. A silent
error there -- a transposed reshape, an off-by-one in which layers the centroid
covers -- would invalidate every latency number without failing anything else.

Run: conda run -n msc-diss python test_p2_scoring.py
"""

import numpy as np
import torch

from p2_extract_latency import score_positions
from phase1.phase1_activation import centroid_scores

rng = np.random.default_rng(0)
L, H, N = 7, 13, 5           # layers+1 = L, hidden = H, examples = N
feats = rng.normal(size=(N, L, H)).astype(np.float32)
w = rng.normal(size=(L * H,)).astype(np.float32)
b = 0.37
hc = rng.normal(size=(L - 1, H)).astype(np.float32)
lc = rng.normal(size=(L - 1, H)).astype(np.float32)

ref_log = feats.reshape(N, -1) @ w + b
ref_cen = centroid_scores(feats, hc, lc)

# Present the same features as one "sequence" per row so score_positions sees the
# identical vectors the frozen path saw.
hidden = [torch.as_tensor(feats[:, layer, :]).unsqueeze(1) for layer in range(L)]
rows = torch.arange(N)
pos = torch.zeros(N, dtype=torch.long)
got_log, got_cen = score_positions(
    hidden, rows, pos, torch.as_tensor(w.reshape(L, H)), b,
    torch.as_tensor(hc), torch.as_tensor(lc))
print("logistic max|delta| =", float(np.abs(got_log.numpy() - ref_log).max()))
print("centroid max|delta| =", float(np.abs(got_cen.numpy() - ref_cen).max()))
assert np.allclose(got_log.numpy(), ref_log, atol=1e-4), "logistic path diverges"
assert np.allclose(got_cen.numpy(), ref_cen, atol=1e-5), "centroid path diverges"
print("OK: per-layer accumulation reproduces both frozen detectors")
