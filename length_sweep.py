"""Compare judge harm labels across response-length clips (512/384/256).

Decides whether 512 generator tokens can drop: pick the shortest clip that retains
>=95% of the harmful labels found at full length (512 = baseline, no truncation).
"""

import pandas as pd

files = {512: "data/judged_v2_512.jsonl",
         384: "data/judged_v2_384.jsonl",
         256: "data/judged_v2_256.jsonl"}

dfs = {n: pd.read_json(p, lines=True).set_index("id") for n, p in files.items()}
base = dfs[512]["harmful"]
common = base.index
base = base.loc[common].dropna().astype(bool)
base_harm = base[base].index  # ids judged harmful at full length

print(f"{len(common)} exchanges; {len(base_harm)} harmful at 512 (baseline)\n")
print(f"{'N':>4}  {'harmful':>7}  {'retain':>7}  {'agree':>6}  {'flip H->B':>9}  {'flip B->H':>9}")
for n in (512, 384, 256):
    h = dfs[n]["harmful"].reindex(common).dropna().astype(bool)
    both = base.index.intersection(h.index)
    b, hh = base.loc[both], h.loc[both]
    retain = hh.reindex(base_harm).fillna(False).mean()
    agree = (b == hh).mean()
    flip_hb = ((b) & (~hh)).sum()
    flip_bh = ((~b) & (hh)).sum()
    print(f"{n:>4}  {int(hh.sum()):>7}  {retain:>6.1%}  {agree:>6.1%}  {flip_hb:>9}  {flip_bh:>9}")
