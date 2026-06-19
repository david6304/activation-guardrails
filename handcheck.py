"""Hand-check the judge against David's labels to validate the rubric before scaling.

Two steps:
  1. sample: draw a stratified sample of judged exchanges into a CSV with an empty
     `my_label` column.
  2. score: after the column is filled (1=harmful, 0=benign), report agreement and a
     confusion matrix vs the judge, and list the disagreements to eyeball.

Strata target the decision boundary: harmful-prompt responses the judge called harmful
(over-flagging?), harmful-prompt responses it called benign (missed / truly benign?),
and benign-prompt responses (clean negatives).
"""

import argparse

import pandas as pd


def sample(args):
    rows = pd.read_json(args.inp, lines=True)
    rows = rows[rows["harmful"].notna()].copy()
    rows["harmful"] = rows["harmful"].astype(bool)

    harmful_dt = rows[rows["data_type"].str.endswith("harmful")]
    benign_dt = rows[rows["data_type"].str.endswith("benign")]
    strata = {
        "harmful_prompt_judged_harmful": harmful_dt[harmful_dt["harmful"]],
        "harmful_prompt_judged_benign": harmful_dt[~harmful_dt["harmful"]],
        "benign_prompt": benign_dt,
    }
    ns = {"harmful_prompt_judged_harmful": args.n_pos,
          "harmful_prompt_judged_benign": args.n_neg,
          "benign_prompt": args.n_benign}

    picks = []
    for name, sub in strata.items():
        take = min(ns[name], len(sub))
        picks.append(sub.sample(n=take, random_state=args.seed).assign(stratum=name))
        print(f"  {name}: {len(sub)} available, sampled {take}")
    out = pd.concat(picks)

    out["my_label"] = ""
    cols = ["id", "stratum", "data_type", "truncated", "harmful", "category",
            "my_label", "judge_reason", "prompt", "response"]
    out = out[[c for c in cols if c in out.columns]].rename(columns={"harmful": "judge_harmful"})
    out.to_csv(args.out, index=False)
    print(f"\n[done] wrote {len(out)} rows -> {args.out}  (fill my_label: 1=harmful, 0=benign)")


def label(args):
    df = pd.read_csv(args.csv, dtype={"my_label": str})
    blank = df["my_label"].isna() | (df["my_label"].astype(str).str.strip() == "")
    todo = list(df[blank].index)
    print(f"{len(todo)} unlabelled of {len(df)}. 1=harmful, 0=benign, s=skip, q=save+quit.")
    print("(judge verdict hidden to avoid anchoring; saved after every label)\n")
    for i in todo:
        r = df.loc[i]
        print("=" * 80)
        print(f"id={r['id']}  truncated={r.get('truncated')}")
        print("-- PROMPT --\n" + str(r["prompt"]))
        print("-- RESPONSE --\n" + str(r["response"]))
        x = ""
        while x not in {"1", "0", "s", "q"}:
            x = input("label [1/0/s/q]: ").strip().lower()
        if x == "q":
            break
        if x == "s":
            continue
        df.loc[i, "my_label"] = x
        df.to_csv(args.csv, index=False)
    df.to_csv(args.csv, index=False)
    print("saved.")


def recheck(args):
    """Re-examine only the over-flag rows (you=harmful, judge=benign) against the
    'did the RESPONSE deliver material harm?' criterion. Shows the judge's reasoning
    and lets you re-label in place."""
    df = pd.read_csv(args.csv, dtype={"my_label": str})
    mine = df["my_label"].map(_to_bool)
    judge = df["judge_harmful"].map(_to_bool)
    todo = list(df[(mine == True) & (judge == False)].index)
    print(f"{len(todo)} over-flag rows (you=harmful, judge=benign). "
          "Criterion: does the RESPONSE deliver material harm? 1=harmful, 0=benign, s=keep, q=quit.\n")
    for i in todo:
        r = df.loc[i]
        print("=" * 80)
        print(f"id={r['id']}  cat={r.get('category')}")
        print("-- PROMPT --\n" + str(r["prompt"]))
        print("-- RESPONSE --\n" + str(r["response"]))
        print("-- JUDGE (said benign) --\n" + str(r.get("judge_reason")))
        x = ""
        while x not in {"1", "0", "s", "q"}:
            x = input("label [1/0/s/q]: ").strip().lower()
        if x == "q":
            break
        if x == "s":
            continue
        df.loc[i, "my_label"] = x
        df.to_csv(args.csv, index=False)
    df.to_csv(args.csv, index=False)
    print("saved.")


def _to_bool(x):
    s = str(x).strip().lower()
    if s in {"1", "1.0", "true", "harmful", "y", "yes", "h"}:
        return True
    if s in {"0", "0.0", "false", "benign", "n", "no", "b"}:
        return False
    return None


def score(args):
    df = pd.read_csv(args.csv)
    df["mine"] = df["my_label"].map(_to_bool)
    df["judge"] = df["judge_harmful"].map(_to_bool)
    n_blank = df["mine"].isna().sum()
    d = df[df["mine"].notna()].copy()
    if len(d) == 0:
        print("no labels filled in yet")
        return

    agree = (d["mine"] == d["judge"]).mean()
    print(f"[agreement] {agree:.1%} on {len(d)} labelled ({n_blank} blank)")
    print("\nconfusion (rows=judge, cols=mine):")
    print(pd.crosstab(d["judge"], d["mine"], rownames=["judge"], colnames=["mine"]))

    # Per-stratum agreement: where does the judge disagree with David?
    if "stratum" in d.columns:
        print("\nagreement by stratum:")
        for name, g in d.groupby("stratum"):
            print(f"  {name}: {(g['mine'] == g['judge']).mean():.1%} (n={len(g)})")

    dis = d[d["mine"] != d["judge"]]
    if len(dis):
        print(f"\ndisagreements ({len(dis)}): ids {list(dis['id'])}")


def disagreements(args):
    df = pd.read_csv(args.csv)
    df["mine"] = df["my_label"].map(_to_bool)
    df["judge"] = df["judge_harmful"].map(_to_bool)
    dis = df[df["mine"].notna() & (df["mine"] != df["judge"])]
    print(f"{len(dis)} disagreements. truncated breakdown:")
    print(dis["truncated"].value_counts())
    for _, r in dis.iterrows():
        print("=" * 80)
        print(f"id={r['id']} | judge={r['judge']} mine={r['mine']} | "
              f"trunc={r.get('truncated')} | cat={r.get('category')}")
        print("JUDGE: " + str(r.get("judge_reason")))
        print("RESP: " + str(r["response"])[:args.chars])


def blind_prep(args):
    """Strip the judge verdict and David's labels so Claude can label blind."""
    df = pd.read_csv(args.csv)
    out = df[["id", "prompt", "response"]].copy()
    out["claude_label"] = ""
    out.to_csv(args.out, index=False)
    print(f"[done] wrote {len(out)} rows -> {args.out}  (id/prompt/response only)")


def triangulate(args):
    """Merge David, Claude, and judge labels by id; report pairwise + 3-way agreement."""
    david = pd.read_csv(args.csv)
    claude = pd.read_csv(args.claude)
    m = david[["id", "my_label", "judge_harmful"]].merge(
        claude[["id", "claude_label"]], on="id")
    m["david"] = m["my_label"].map(_to_bool)
    m["claude"] = m["claude_label"].map(_to_bool)
    m["judge"] = m["judge_harmful"].map(_to_bool)
    m = m.dropna(subset=["david", "claude", "judge"])
    print(f"[triangulate] {len(m)} rows with all three labels\n")
    for a, b in [("david", "judge"), ("claude", "judge"), ("david", "claude")]:
        print(f"  {a} vs {b}: {(m[a] == m[b]).mean():.1%}")
    unan = (m["david"] == m["claude"]) & (m["claude"] == m["judge"])
    print(f"\n  all three agree: {unan.mean():.1%} ({unan.sum()}/{len(m)})")
    dis = m[~unan]
    if len(dis):
        print(f"\n  splits ({len(dis)}): ids {list(dis['id'])}")
        print(dis[["id", "david", "claude", "judge"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample")
    s.add_argument("--in", dest="inp", required=True)
    s.add_argument("--out", default="data/handcheck.csv")
    s.add_argument("--n-pos", type=int, default=30)
    s.add_argument("--n-neg", type=int, default=30)
    s.add_argument("--n-benign", type=int, default=15)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=sample)

    l = sub.add_parser("label")
    l.add_argument("--csv", default="data/handcheck.csv")
    l.set_defaults(func=label)

    c = sub.add_parser("score")
    c.add_argument("--csv", default="data/handcheck.csv")
    c.set_defaults(func=score)

    g = sub.add_parser("disagreements")
    g.add_argument("--csv", default="data/handcheck.csv")
    g.add_argument("--chars", type=int, default=600)
    g.set_defaults(func=disagreements)

    bp = sub.add_parser("blind-prep")
    bp.add_argument("--csv", default="data/handcheck_v2.csv")
    bp.add_argument("--out", default="data/handcheck_blind.csv")
    bp.set_defaults(func=blind_prep)

    rc = sub.add_parser("recheck")
    rc.add_argument("--csv", default="data/handcheck_v2.csv")
    rc.set_defaults(func=recheck)

    t = sub.add_parser("triangulate")
    t.add_argument("--csv", default="data/handcheck_v2.csv")
    t.add_argument("--claude", default="data/handcheck_blind.csv")
    t.set_defaults(func=triangulate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
