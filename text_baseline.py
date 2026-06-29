"""TF-IDF text baseline vs probe_v1, with CC++-style complementarity analysis.

CC++'s probe result is NOT probe-beats-text: their probe is competitive-to-worse
than a fine-tuned text classifier (Fig 2a/3a), and the win comes from (i) ~zero
compute and (ii) decorrelated errors so a probe+text ENSEMBLE beats either alone
(Fig 3a,b). This script tests whether that complementarity holds on our stack:
TF-IDF on response text vs the existing all-layer SWiM probe, in-distribution
(vanilla) and under vanilla->adversarial transfer.

Text classifies on RESPONSE text only, to match what the probe sees.
"""

import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def labelled(rows):
    return [r for r in rows if r.get("harmful") in (True, False)]


def tpr_at_fpr(y, s, f):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(f, fpr, tpr))


def line(name, y, s):
    print(f"  {name:26s} AUROC={roc_auc_score(y, s):.3f}  "
          f"TPR@1%={tpr_at_fpr(y, s, 0.01):.3f}  TPR@0.1%={tpr_at_fpr(y, s, 0.001):.3f}")


# --- train text baseline on vanilla responses ---
train = labelled(load("data/judged_train.jsonl"))
y_tr = np.array([r["harmful"] for r in train])
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
clf = LogisticRegression(max_iter=2000, class_weight="balanced")
clf.fit(vec.fit_transform([r["response"] for r in train]), y_tr)
print(f"[train] vanilla n={len(train)} pos={int(y_tr.sum())}")


def text_scores(rows):
    return clf.decision_function(vec.transform([r["response"] for r in rows]))


def probe_map(score_path):
    return {(r["id"], r["data_type"]): r["score"] for r in load(score_path)}


def analyse(title, rows, is_pos, score_path):
    sc = probe_map(score_path)
    rows = [r for r in rows if (r["id"], r["data_type"]) in sc]
    y = np.array([is_pos(r) for r in rows])
    ts = text_scores(rows)
    ps = np.array([sc[(r["id"], r["data_type"])] for r in rows])
    print(f"\n=== {title}  (n={len(y)}, pos={int(y.sum())}) ===")
    line("text", y, ts)
    line("probe", y, ps)
    # complementarity
    zt = (ts - ts.mean()) / ts.std()
    zp = (ps - ps.mean()) / ps.std()
    rho = spearmanr(ts, ps).correlation
    line("ensemble (equal)", y, zt + zp)
    alphas = np.linspace(0, 1, 21)
    aucs = [roc_auc_score(y, a * zt + (1 - a) * zp) for a in alphas]
    a_best = alphas[int(np.argmax(aucs))]
    line(f"ensemble (best a={a_best:.2f})", y, a_best * zt + (1 - a_best) * zp)
    print(f"  Spearman(text,probe)={rho:+.3f}   [best-a tuned on test => optimistic]")


# in-distribution vanilla: harm vs benign
analyse("IN-DIST vanilla", labelled(load("data/judged_eval.jsonl")),
        lambda r: r["harmful"] is True, "data/vanilla_eval_scores.jsonl")

# transfer adversarial, harm-isolated: harmful vs adversarial-benign
adv = load("data/judged_adv_eval.jsonl")
keep = [r for r in adv if r.get("harmful") is True
        or (r.get("data_type") == "adversarial_benign" and r.get("harmful") is False)]
analyse("TRANSFER adversarial", keep,
        lambda r: r["harmful"] is True, "data/adv_eval_scores.jsonl")
