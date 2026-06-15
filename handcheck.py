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
    rows = rows[rows["harmful"].notna()]

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

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
