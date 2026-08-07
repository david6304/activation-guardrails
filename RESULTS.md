# Consolidated results (living table)

Update as runs land. All on protected `gemma-3-it`. Raw data paths noted per block.
Freeze target ~end July 2026. See `RESEARCH_LOG.md` for method/decisions.

## 1. Capability — decode retention on WebQuestions (acc / plain-acc), N=150

| transform     | 12B ret | 27B ret | mechanism |
|---------------|--------:|--------:|-----------|
| plain (base)  | 100% (acc 50.7%) | 100% (acc 54.0%) | — |
| french        | 96% | 100% | NLLB translation |
| swahili       | 87% | 99% | NLLB translation |
| hindi         | 85% | 88% | NLLB translation |
| zulu          | 72% | 89% | NLLB translation |
| reverse       | 36% | 51% | char reversal + decode scaffold |
| vowel-removal | —   | 61% | fallback cipher |
| piglatin      | —   | 2%  | fallback (model-blind → control) |
| nato          | 0%  | 1%  | serial-CoT decode → predicted-fail control |
| morse         | 0%  | 0%  | serial-CoT decode → predicted-fail control |

27B ≥ 12B on every transform (left-shift, as claim 2 predicts). Caveat: retention is a
WebQuestions trivia proxy; `capability_qa.py:42` substring-matches short golds — hand-validation
with paired CIs still pending. Raw: Eddie `data/cap_qa_langs_{12b,27b}.jsonl`;
MLP `data/cap_qa_{27b,12b}.jsonl` (jobs 3543515/3543526) + `data/cap_qa_27b.jsonl` (3557511, fallbacks).
The Eddie 12B/27B language files have identical `(qidx, condition, sent)` inputs (0/600
differences; per-condition SHA-256s match). NLLB revision:
`f8d333a098d19b4fd9a8b18f94170487ad3f821d`.

## 2. Public-guard recall / FPR — WJ-harmful vs XSTest-safe

Languages n=150 (job 3562722); ciphers n=50 (jobs 3541577, 3557500). WildGuard = dedicated
prompt-harm classifier; ShieldGemma-9b = composite-guideline P(Yes).

| transform | WildGuard recall | WG FPR | ShieldGemma recall | SG FPR |
|-----------|-----------------:|-------:|-------------------:|-------:|
| plain     | 96.7% | 0.7% | 62.0% | 12.7% |
| french    | 88.0% | 0.0% | 56.7% | 11.3% |
| hindi     | 16.0% | 1.3% | 50.7% | 12.0% |
| swahili   |  0.0% | 0.0% | 38.7% |  6.7% |
| zulu      |  0.0% | 0.7% | 13.3% |  5.3% |
| reverse   |  0.0% | 0.0% |  0.0% |  0.0% |
| nato      |  0.0% | 0.0% |  0.0% |  0.0% |
| morse     |  0.0% | 0.0% |  0.0% |  0.0% |
| vowel     |  4.0% | 2.0% | 22.0% |  2.0% |
| piglatin  |  4.0% | 0.0% | 10.0% |  4.0% |

WildGuard shows graded blindness by language resource level (plain → french → hindi → swahili/zulu);
both guards ≈0% on ciphers. **Swahili = capability/detectability dissociation**: 99% model decode
(27B) but 0% WildGuard recall. Raw: MLP `data/guard_screen_langs.jsonl`, `data/guard_screen.jsonl`.

## 3. Probe — plain-trained all-layer concat, transfer to test conditions

Threshold = 0.1% WildChat alert rate. Trained on **operational-harm** positives only: the 960
`hate & harassment targeting protected groups` positives and one off-taxonomy `harassment`
positive are **excluded** by default (outside the locked operational definition; the seven
protected-group-category rows judged benign remain hard negatives). Operational harm explicitly
includes violent and non-violent crime, sex crimes and child exploitation, weapons/CBRN, cyber,
self-harm, privacy/PII, and fraud/deception. Re-include all judge-harmful categories with
`--keep-protected-group` for a broad-harm sensitivity variant if time allows.
Test conditions now cover the full coupling axis: plain, reverse, french, hindi, swahili, zulu
(`probe_prompt.py`, commit f0d7c55).

| transform | 12B AUROC | 12B TPR | 27B AUROC | 27B TPR |
|-----------|-----------|---------|-----------|---------|
| plain     | _pending_ | | _pending_ | |
| reverse   | _pending_ | | _pending_ | |
| french    | _pending_ | | _pending_ | |
| hindi     | _pending_ | | _pending_ | |
| swahili   | _pending_ | | _pending_ | |
| zulu      | _pending_ | | _pending_ | |

Definitive re-run pending on f0d7c55 (exclude protected-group + fra/hin/swh conditions). Prior
jobs Eddie 56990400 (12B, running) / Wintermute 3563857 (27B, pending) are on superseded code
(include-protected-group, plain/reverse/zulu only). Reverse pilot (per-layer, n=300, raw WJ
labels): plain-val AUROC ~0.997; plain→reverse transfer peaks **0.955 @ layer 34/62**.
Raw (when done): `data/probe_prompt_{model}_report.json` + `_scores.npz`.

## 4. Phase 1 development matrix — all detectors × all six conditions (27B, `t_inst`)

Test n=1781 (568 harmful / 1213 benign); tune n=1781 used for calibration only. Cells are
**AUROC / TPR% / FPR%** at the 1%-FPR operating point. **Strict** = threshold from *plain* tune
negatives applied to every condition; **matched** = condition-specific tune negatives (uses no
shifted harmful examples). Complete matrix — no pending cells.

Strict transfer:

| detector | plain | french | hindi | swahili | zulu | reverse |
|----------|-------|--------|-------|---------|------|---------|
| all-layer logistic | **0.988 / 75.4 / 1.1** | **0.980 / 63.0 / 0.9** | **0.982 / 61.8 / 0.8** | **0.967 / 37.3 / 0.2** | **0.962 / 34.5 / 0.4** | 0.556 / 0.0 / 0.0 |
| centroid (Zhao) | 0.939 / 56.5 / 2.2 | 0.932 / 30.8 / 1.0 | 0.923 / 19.5 / 0.1 | 0.895 / 9.7 / 0.2 | 0.886 / 6.9 / 0.2 | **0.767 / 0.0 / 0.0** |
| ShieldGemma-9b | 0.940 / 45.2 / 1.4 | 0.925 / 36.8 / 0.6 | 0.902 / 34.3 / 1.7 | 0.886 / 30.1 / 1.1 | 0.833 / 8.8 / 0.2 | 0.659 / 0.0 / 0.0 |
| Qwen3Guard-Gen-8B | 0.968 / 54.9 / 1.9 | 0.951 / 31.7 / 0.8 | 0.950 / 15.5 / 0.1 | 0.882 / 0.2 / 0.0 | 0.795 / 0.0 / 0.0 | 0.595 / 0.0 / 0.0 |
| Llama-Guard-4-12B | 0.960 / 40.8 / 0.7 | 0.913 / 24.5 / 0.8 | 0.889 / 21.8 / 0.9 | 0.696 / 0.0 / 0.0 | 0.717 / 0.0 / 0.0 | 0.480 / 0.0 / 0.0 |
| HaloGuard-1.0-4B | 0.924 / 8.8 / 0.8 | 0.908 / 27.1 / 4.0 | 0.832 / 4.4 / 0.5 | 0.817 / 3.2 / 0.2 | 0.745 / 0.4 / 0.1 | 0.625 / 0.0 / 0.0 |
| multilingual-e5-base | 0.967 / 56.3 / 0.9 | 0.947 / 51.2 / 1.6 | 0.906 / 18.3 / 0.7 | 0.837 / 2.1 / 0.1 | 0.813 / 0.9 / 0.0 | 0.616 / 0.0 / 0.0 |
| DeBERTa-v3-small guard | 0.981 / 63.9 / 1.1 | 0.940 / 15.3 / 0.1 | 0.862 / 0.0 / 0.0 | 0.833 / 0.2 / 0.0 | 0.805 / 0.0 / 0.0 | 0.472 / 0.0 / 0.0 |
| char TF–IDF | 0.962 / 45.2 / 1.0 | 0.731 / 0.5 / 0.1 | 0.531 / 0.0 / 0.0 | 0.596 / 0.0 / 0.0 | 0.580 / 0.0 / 0.0 | 0.543 / 0.0 / 0.0 |

Condition-matched calibration (TPR% / FPR%; AUROC is calibration-invariant, as above):

| detector | plain | french | hindi | swahili | zulu | reverse |
|----------|-------|--------|-------|---------|------|---------|
| all-layer logistic | **75.4 / 1.1** | **72.5 / 1.3** | **69.5 / 1.2** | **59.3 / 1.3** | **51.2 / 1.2** | 2.1 / 0.7 |
| centroid (Zhao) | 56.5 / 2.2 | 42.8 / 1.8 | 38.9 / 1.2 | 33.6 / 1.5 | 20.8 / 0.7 | **13.6 / 1.6** |
| ShieldGemma-9b | 45.2 / 1.4 | 36.8 / 0.6 | 30.1 / 1.3 | 30.1 / 1.1 | 15.3 / 0.8 | 1.9 / 0.2 |
| Qwen3Guard-Gen-8B | 54.9 / 1.9 | 49.1 / 1.7 | 41.5 / 1.2 | 16.4 / 1.0 | 9.2 / 1.2 | 1.6 / 1.2 |
| Llama-Guard-4-12B | 40.8 / 0.7 | 30.5 / 1.2 | 0.0 / 0.0 † | 6.0 / 1.2 | 5.5 / 1.4 | 1.8 / 2.1 |
| HaloGuard-1.0-4B | 8.8 / 0.8 | 8.5 / 1.6 | 10.4 / 1.6 | 8.5 / 1.6 | 6.0 / 0.9 | 9.7 / 1.2 |
| multilingual-e5-base | 56.3 / 0.9 | 41.2 / 0.7 | 31.9 / 1.7 | 16.7 / 1.6 | 14.1 / 1.4 | 5.1 / 2.1 |
| DeBERTa-v3-small guard | 63.9 / 1.1 | 40.0 / 1.3 | 13.6 / 0.6 | 11.8 / 1.4 | 6.2 / 0.8 | 2.1 / 1.8 |
| char TF–IDF | 45.2 / 1.0 | 8.8 / 0.8 | 5.8 / 0.9 | 3.9 / 0.8 | 3.3 / 1.2 | 3.7 / 1.5 |

WildGuard at its **native** decision point (no threshold to calibrate, so shown separately):

| transform | plain | french | hindi | swahili | zulu | reverse |
|-----------|-------|--------|-------|---------|------|---------|
| TPR | 97.4% | 89.6% | 22.9% | 1.4% | 0.4% | 0.0% |
| FPR | 17.4% | 13.8% | 1.5% | 0.0% | 0.0% | 0.0% |

Probe TPR retention vs plain: strict 84% / 82% / 50% / 46% / 0%; matched 96% / 92% / 79% / 68% / 3%
(french / hindi / swahili / zulu / reverse).

**Reading.** The all-layer activation probe leads every text baseline on AUROC *and* TPR in all
five language conditions under both calibrations, including against multilingual-e5, a text
encoder built for multilingual semantics, and against both current-generation guards —
Qwen3Guard-Gen-8B (purpose-built multilingual, 2025-09), Llama-Guard-4-12B, and
HaloGuard-1.0-4B (2026-07, the newest guard evaluated here and the only one declaring Swahili
and isiZulu coverage by design). Llama Guard 4 is
the *weakest* of the three guards at the low-resource end despite being the newest and largest
(matched swahili 6.0%, zulu 5.5%), and is below chance on reverse (AUROC 0.480). Qwen3Guard is
the strongest text comparator at the
mid-resource end (matched french 49.1%, hindi 41.5%, both well above ShieldGemma's 36.8% / 30.1%)
but *not* at the low-resource end, where it falls below ShieldGemma (swahili 16.4% vs 30.1%, zulu
9.2% vs 15.3%) — multilingual guard training moves french and hindi and leaves swahili and zulu
roughly where they were. The probe's matched margin over the best of the three open guards in each
condition is +20.5 / +23.4 / +28.0 / +29.2 / +35.9 points (plain → zulu). Adding HaloGuard does not change
those margins: it ranks acceptably (AUROC 0.924 plain, 0.817 swahili) but **never produces a
usable operating point**, returning 6--10% matched TPR in every condition including plain
English. Its scores pile up at 0 and 1 — the 1%-FPR quantile lands inside a mass of ties — so
this is a *calibration* failure, not an absence of rank signal, and it should be reported as
such. Its batch-composition audit also fails (12/48 cells over the 1e-3 tolerance, max 0.0156),
the padding sensitivity expected of a score that is undecided on most rows. Treat the HaloGuard
row as one current-generation datapoint, not as evidence about multilingual guards in general:
the model is three weeks old at time of scoring, with no independent replication of its
published claims. Degradation is monotone in language
resource level
(plain → french → hindi → swahili → zulu) for every detector, but far steeper for text: WildGuard
falls 97.4% → 0.4%, ShieldGemma 45.2% → 15.3% (matched), the probe 75.4% → 51.2%. ShieldGemma is
now the strongest *single* text comparator on the low-resource end (swahili 30.1%), so it is the
right fusion partner for Step 4 — not WildGuard. Reverse defeats every detector, and the centroid
is the only one retaining signal there (0.767 AUROC, 13.6% matched TPR) while the all-layer
logistic collapses to 0.556 — a probe-geometry failure rather than absent signal.

**Caveat.** The ShieldGemma cross-environment audit failed (max abs probability difference 0.0312
vs 1e-3 tolerance) on 2 of 48 frozen audit cells; it changed no reported metric (see
`RESEARCH_LOG.md` 2026-07-24). WildGuard's audit passed exactly. Qwen3Guard's batch-composition
audit (the same 48 cells rescored at batch size 1) also failed the 1e-3 tolerance, at 5 of 48 cells
and max 0.0220, but the deviation is confined to unsaturated scores: ≤4.8e-4 on the 40 cells with
P(Unsafe) outside [0.01, 0.99] and up to 0.0220 on the 8 mid-range cells. The plain/french/hindi
thresholds sit in the saturated region (0.995–0.997) and are unaffected; the swahili, zulu and
reverse thresholds (0.61, 0.071, 0.0034) are not, so a few alerts there are within score noise.
All reported Qwen3Guard numbers come from one internally consistent batched pass.

**† Llama Guard 4 hindi is degenerate, not blind.** Its P(unsafe) saturates hard: the matched
hindi threshold is float32 **1.0** exactly, so the 99th percentile of hindi tune negatives sits on
the saturation ceiling and no harmful prompt can exceed it, giving 0.0 / 0.0. AUROC is a
respectable 0.889, so the rank signal is there — what fails is thresholding a score with a large
tied mass at 1.0. Read the cell as a calibration failure of the guard's score distribution, never
as "Llama Guard 4 detects no Hindi harm". The plain and hindi thresholds are both 1.0 and french
0.999999.

The same saturation drives Llama Guard 4's batch-composition audit failure: max 0.0567 over 7 of
48 cells, and **every failing cell is swahili or reverse** — none are plain, french or hindi.
Where the model commits, its score is stable to padding; where it is undecided, it is not. On zulu
82% of prompts fall in the undecided band (0.01–0.99) at mean score 0.494, i.e. near coin-flip, so
its low-resource rows carry more score noise than the other two guards.

Llama Guard 4 **does not lead any condition**, so the strongest-guard-per-condition split used
downstream is unchanged: Qwen3Guard at plain/french/hindi, ShieldGemma at swahili/zulu.

Raw: `data/phase1_baselines_multilingual.npz` (Eddie job 57134364), `data/c4_modern_guards.npz`
(Eddie job 57140956), `data/c4_modern_guards_lg4.npz` (Eddie job 57164545, carries the Qwen3Guard
arrays through), `data/phase1_activation_multilingual_27b.npz`, `data/phase1_small_guard.npz`,
`data/phase1_multilingual_e5.npz`; analysis
`data/phase1_text_encoder_multilingual_results.json`, `data/c4_modern_guard_results.json` and
`data/c4_lg4_results.json`. Commits `59649dd` (rows through TF–IDF), `00149bb` (Qwen3Guard),
`374c491` (Llama Guard 4) and `c2ae933` (analysis), seed 0. Revisions: ShieldGemma-9b
`b8b63601…`, Qwen3Guard-Gen-8B `4505cb1a…`, Llama-Guard-4-12B `87acb4b9…`. Known-test status:
exploratory development evidence — no model, layer, threshold or condition was selected on it.

## 5. C1 — threshold transport from *unlabelled* shifted traffic

The matched operating point of §4 uses labelled tune negatives. This section replaces that
labelling requirement: the threshold is the 99th percentile of `k` same-condition prompts drawn
without using labels, contaminated at harmful prevalence `pi`, which is what shifted *benign
traffic* looks like in deployment. 400 draws per cell, seed 0, applied to the frozen test split.
Strict and oracle reference rows are the §4 strict and matched columns.

Headline cell `k=300, pi=0.01` — mean TPR% / mean realised FPR% (recovery = TPR as a fraction of
the §4 matched oracle):

| detector | french | hindi | swahili | zulu | reverse |
|----------|--------|-------|---------|------|---------|
| all-layer logistic | **62.1 / 1.00** (86%) | **61.4 / 0.80** (88%) | **55.0 / 1.05** (93%) | **47.7 / 1.00** (93%) | 2.8 / 0.95 (130%) |
| centroid (Zhao) | 41.9 / 1.88 (98%) | 37.5 / 1.09 (96%) | 33.2 / 1.60 (99%) | 20.5 / 0.82 (99%) | **12.6 / 1.49** (93%) |
| ShieldGemma-9b | 36.6 / 0.86 (99%) | 29.1 / 1.26 (97%) | 29.5 / 1.08 (98%) | 15.3 / 0.85 (100%) | 2.8 / 0.61 (147%) |
| Qwen3Guard-Gen-8B | 43.4 / 1.44 (88%) | 40.1 / 1.05 (96%) | 16.1 / 0.99 (99%) | 9.6 / 1.32 (104%) | 2.3 / 1.36 (147%) |
| multilingual-e5-base | 37.6 / 0.69 (91%) | 28.1 / 1.56 (88%) | 16.7 / 1.68 (100%) | 15.6 / 1.61 (111%) | 5.7 / 2.33 (111%) |
| DeBERTa-v3-small guard | 38.7 / 1.30 (97%) | 14.4 / 0.87 (106%) | 13.0 / 1.51 (110%) | 7.1 / 1.01 (114%) | 2.2 / 2.10 (106%) |
| char TF–IDF | 9.0 / 1.10 (102%) | 5.6 / 0.85 (96%) | 4.3 / 1.08 (111%) | 3.6 / 1.28 (108%) | 3.9 / 2.04 (104%) |

Probe TPR versus `k` and `pi` (mean TPR%; realised FPR% in the swahili row beneath):

| condition | k | pi=0.00 | 0.01 | 0.02 | 0.05 | 0.10 |
|---|---:|---|---|---|---|---|
| french | 300 | 73.5 | 62.1 | 46.9 | 24.4 | 12.9 |
| hindi | 300 | 70.9 | 61.4 | 49.5 | 26.6 | 13.7 |
| swahili | 100 | 67.0 | 60.3 | 50.8 | 34.4 | 21.1 |
| swahili | 300 | 62.0 | 55.0 | 45.8 | 25.3 | 13.4 |
| swahili | 1000 | 60.3 | 54.9 | 44.5 | 22.3 | 11.6 |
| swahili | 3000 | 59.7 | 54.6 | 43.6 | 19.9 | 10.9 |
| swahili FPR | 300 | 1.44 | 1.05 | 0.67 | 0.16 | 0.02 |
| zulu | 300 | 53.6 | 47.7 | 42.6 | 25.3 | 13.5 |
| reverse | 300 | 2.8 | 2.8 | 2.7 | 2.7 | 2.5 |

**Reading.** The acceptance rule (probe recovers ≥85% of oracle TPR at `k=300, pi=0.01` in ≥3 of
the four language conditions) is met in all four: 86 / 88 / 93 / 93%. The condition-matched
operating point therefore does not need labelled shifted data — a few hundred unlabelled
same-condition prompts recover most of it, which is a deployment claim rather than an adaptation
result. The detector ranking of §4 is unchanged, and the probe's advantage over the better of the
two open guards at this operating point is +18.7 / +21.3 / +25.5 / +32.4 points
(french → zulu).

Three qualifications, all reportable rather than repairable. First, the probe is the *only*
detector for which unlabelled calibration is worth much: it gains +17.7 (swahili) and +13.2 (zulu)
TPR points over its strict threshold, whereas ShieldGemma gains −0.6 and +6.5 — though Qwen3Guard,
whose strict thresholds transport worst of all, also gains substantially (+15.9 swahili, +24.6
hindi). Second, recovery is bounded above by ~90–100% and never exceeds the oracle in the language
conditions; the >100% cells belong to detectors whose oracle TPR is near zero, where the ratio is
uninformative. Third, contamination costs TPR steeply — at `pi=0.05` the probe retains only
25.3% swahili TPR — but it does so by pushing the threshold *up*: realised FPR falls to 0.16%, so
a contaminated estimate makes the detector conservative, not unsafe. Larger `k` slightly *lowers*
TPR (`k=100` overshoots both TPR and FPR because the interpolated 99th percentile of 100 points is
anti-conservative); `k=300` is already close to the `k=3000` asymptote.

Reverse is unaffected, as expected: there is no operating-point signal for calibration to recover.

Raw: `data/c1_unlabelled_calibration.json` (SHA-256
`f5855064cca4b4c2373ebb18cdb8a55ed72a8aa7aabbab16f15989af999fbf63`),
`conda run -n msc-diss python -m phase1.analyse_unlabelled_calibration`, seed 0, 400 draws per
cell, CPU only. Inputs are the frozen score artefacts of §4. Reference rows use the repo's
`threshold_at_one_percent`; the simulated draws use the plain interpolated 99th percentile, since
the order-statistic version is markedly conservative at `k=100`.

## 6. C2 — layer aggregation destroys shift-robust signal

Single-layer readouts from `data/phase1_layerwise_27b.npz`, with the layer chosen once by argmax
AUROC on **plain tune** scores only (logistic → L34, centroid → L54; no transformed and no test
data enter the rule), against the two aggregated readouts. Test AUROC:

| detector (plain-tune selection) | plain | swahili | reverse | vowel |
|---|---|---|---|---|
| all-layer logistic | 0.988 | 0.967 | 0.556 | — |
| layer-averaged centroid | 0.939 | 0.895 | 0.767 | 0.825 |
| L34 logistic | **0.989** | **0.973** | 0.668 | 0.832 |
| L54 centroid | 0.971 | 0.925 | **0.829** | **0.879** |

Paired bootstrap (10,000 repeats, resampling calibration negatives and test examples jointly)
against the all-layer logistic, on reverse: L54 centroid **+0.273 AUROC** [+0.246, +0.299], L34
logistic +0.112 [+0.087, +0.136], layer-averaged centroid +0.211 [+0.184, +0.238]. On plain and
swahili the single layer is level with the all-layer probe on AUROC (+0.001 [−0.000, +0.003] and
+0.006 [+0.002, +0.010] for L34) — so the reverse gain is not bought with in-distribution loss.

**The gain is rank-level and does not survive at the operating point.** Condition-matched TPR% /
FPR% at 1%:

| detector | plain | swahili | reverse | vowel |
|---|---|---|---|---|
| all-layer logistic | 75.4 / 1.1 | 59.3 / 1.3 | 2.1 / 0.7 | — |
| layer-averaged centroid | 56.5 / 2.2 | 33.6 / 1.5 | **13.6 / 1.6** | **20.8 / 0.6** |
| L34 logistic | 78.0 / 1.2 | 57.2 / 1.0 | 3.3 / 0.9 | 13.0 / 1.0 |
| L54 centroid | 62.0 / 1.2 | 43.3 / 1.4 | 5.5 / 1.1 | 10.2 / 0.7 |

On reverse the selected layer beats the all-layer probe by only +3.3 TPR points [+0.2, +6.9],
while the *simpler* layer-averaged centroid beats it by +10.2 [+3.5, +15.3]. A plain-tune-selected
layer is therefore not a cipher detector, and this is not an argument for a layer sweep — Gate B
(2026-07-24) already closed that route. The result is diagnostic: concatenating all 62 layers
maximises in-distribution fit and destroys out-of-distribution rank transfer, and the information
reverse-condition prompts still carry sits in a narrow band near L54 that the concatenated probe's
weights do not read.

Selection leaves little on the table. The best test layer for reverse is L54 for the centroid
(0.829, identical to the plain-tune choice) and L55 for the logistic (0.755 against 0.668),
so the honest plain-only rule costs the centroid nothing and the logistic 0.09 AUROC.

Length interaction at L54 centroid (test AUROC by prompt-length quartile): plain 0.981 / 0.992 /
0.963 / 0.933; swahili 0.949 / 0.957 / 0.902 / 0.884; reverse 0.910 / 0.905 / 0.771 / 0.722;
vowel 0.913 / 0.927 / 0.843 / 0.805. The drop is largest on reverse (0.19) against plain (0.05),
which is the interaction motivating C6 — suggestive but confounded, since plain falls too.

Raw: `data/c2_layerwise_selection.json` (SHA-256
`8b24df60bc9b3c27f384013248b1ed89d748340861b08aacfb5c78e9b350f6b0`, includes the full 62-layer
AUROC curves for the figure), `conda run -n msc-diss python -m phase1.analyse_layerwise_selection`,
seed 0, CPU only. Layer ids are 1-indexed as stored in `layer_indices`. Vowel removal exists only
in the layerwise artefact, so its layer-averaged centroid row is recomputed as the mean over
layers (agrees with the frozen centroid to ~1e-4 where both exist).

## 7. C3 — operating points on a 50,000-prompt WildChat background pool

The §4 thresholds come from 1,213 labelled tune negatives, so the 1% point rests on about
twelve prompts and 0.1% is not estimable at all. This sets thresholds on 50,000 unlabelled
first-user-turn WildChat-1M prompts instead (seed 0, deduplicated, **no overlap with any Phase 1
prompt**, manifest SHA-256 `f7fcc84c2b46beb10acef951551b3ec896db53ab6a761803160bcb28c039dcc5`).
The pool is real traffic and unfiltered by language: 47% English, 24% Chinese, 12% Russian,
median 83 characters with a heavy tail (p99 11,910). Prompts are left-truncated to 2,048 tokens,
the budget the probe already applies.

WildChat is unlabelled, so the quantity the threshold fixes is a **background alert rate**, never
an FPR. The labelled FPR on the Phase 1 test negatives is reported beside each TPR. This also
makes the numbers only loosely comparable to CC++ and Segment-Level Coherence, which report at a
labelled-negative 0.1% FPR.

TPR% at a **1% background alert rate** (labelled test FPR% in brackets):

| detector | plain | french | hindi | swahili | zulu | reverse |
|----------|-------|--------|-------|---------|------|---------|
| all-layer logistic | **80.6** (1.73) | **70.1** (1.32) | **68.7** (1.15) | **47.2** (0.58) | **46.1** (0.82) | 0.0 (0.00) |
| centroid (Zhao) | 23.1 (0.49) | 5.6 (0.00) | 0.2 (0.00) | 0.0 (0.00) | 0.0 (0.00) | 0.0 (0.00) |

TPR% at a **0.1% background alert rate** — the operating point §4 could not reach:

| detector | plain | french | hindi | swahili | zulu | reverse |
|----------|-------|--------|-------|---------|------|---------|
| all-layer logistic | **66.5** (0.49) | **53.0** (0.41) | **52.1** (0.25) | **26.9** (0.16) | **23.1** (0.00) | 0.0 (0.00) |
| centroid (Zhao) | 9.7 (0.08) | 1.4 (0.00) | 0.0 (0.00) | 0.0 (0.00) | 0.0 (0.00) | 0.0 (0.00) |

Intervals are 2,000-repeat bootstrap resamples of the pool, propagating uncertainty in the pool
quantile: probe plain 0.1% [63.2, 67.6], swahili 0.1% [24.8, 28.9], zulu 0.1% [20.6, 26.1].

**Reading.** The probe holds a usable operating point an order of magnitude stricter than the one
§4 reports: at one alert per thousand real prompts it still recovers two-thirds of plain harmful
prompts and about a quarter of Swahili ones. The tenfold tightening costs 14 points on plain
(80.6→66.5) and 20 on swahili (47.2→26.9), so the low-FPR tail is where language shift bites
hardest — consistent with C2's finding that rank information survives shift better than the
operating point does.

Two observations about the pool itself. First, the frozen §4 tune-negative threshold (3.956)
raises alerts on only **0.41%** of real WildChat traffic, so the deployed threshold is
*conservative* relative to traffic: WildJailbreak's benign stratum scores higher than real user
prompts, and calibrating on it overstates the alert volume a deployment would actually see.
Second, the centroid all but collapses under pool calibration (plain 23.1% at 1%, 9.7% at 0.1%)
while it reached 56.5% on tune negatives — its scores separate WildJailbreak benign prompts far
better than they separate real traffic, which is a caution against reading §4's centroid row as a
deployment number.

**Incomplete.** ShieldGemma and Qwen3Guard have **not** been scored on the pool, so this section
carries no cross-detector comparison at 0.1%; that requires roughly 17 GPU-hours per guard at the
measured rate and is the outstanding half of C3. The Swahili background was cut (see
`DISSERTATION_FINISH_PLAN.md` C3), so all rows above use the plain pool threshold.

Raw: `data/c3_pool_probe_plain.npz` (SHA-256
`8e07d9b9676e20b56918d9e14b8d205ae15918c9d36071349d3c70c2f468d081`),
`data/c3_pool_results.json` (SHA-256
`3a19f53d9548b38a4a4f9da1d7b220f10d4ad9b980ed9ec47b28ca366ed6803a`),
`data/c3_wildchat_prompts.jsonl`, `data/c3_wildchat_manifest.json`. MLP job 3567213 (2× RTX A6000,
bf16, batch 4, 3h07m39s, exit 0), `google/gemma-3-27b-it` revision `005ad3404e59d6023443cb575daa05336842228a`,
`t_inst`, frozen probe and centroid parameters from `data/phase1_activation_27b.npz`, seed 0,
source commit `c04587c`. Analysis: `conda run -n msc-diss python -m phase1.analyse_wildchat_pool
--pool data/c3_pool_probe_plain.npz`.

## 8. C7 — external-source confirmation on NVIDIA Aegis 2.0

Source transfer, not language transfer: train and test in §§3–4 are both WildJailbreak. The
frozen probe, centroid and ShieldGemma are scored on 22,305 Aegis 2.0 prompts relabelled under
the frozen operational rubric. **No retraining, no layer reselection, no threshold fitted on
external positives.** Thresholds come only from the 7,480 external *tune* negatives; TPR and
realised FPR are reported on the disjoint external *test* partition (7,345 positives /
7,480 negatives). Paired 10,000-repeat bootstrap resamples calibration negatives, test positives
and test negatives jointly and re-estimates every threshold inside each replicate.

TPR / realised FPR, with the probe−ShieldGemma TPR difference and its 95% interval:

| condition | calibration | probe | centroid | ShieldGemma | probe − guard |
|---|---|---|---|---|---|
| plain | frozen WJ threshold | 55.5 / 1.79 | 20.4 / 3.98 | 74.5 / 6.76 | −19.1 [−20.3, −17.8] |
| plain | external English | 50.0 / 1.19 | 8.8 / 0.90 | 46.4 / 0.88 | +4.1 [−0.6, +9.1] |
| swahili | frozen WJ threshold | 23.2 / 0.76 | 4.2 / 0.28 | 34.2 / 2.90 | −11.0 [−12.1, −9.9] |
| swahili | external English | 18.9 / 0.44 | 0.1 / 0.01 | 5.8 / 0.16 | +13.4 [+10.4, +16.6] |
| swahili | external Swahili (matched) | 27.7 / 1.00 | 8.4 / 0.95 | 18.9 / 0.88 | +8.8 [+3.7, +14.3] |

AUROC: probe 0.934 plain / 0.837 swahili; centroid 0.580 / 0.556; ShieldGemma 0.937 / 0.862.

**Qwen3Guard collapses out of source, and this reorders the guards.** Qwen3Guard was the
*strongest* guard on plain English in §4 (matched 54.9 against ShieldGemma's 45.2). On the
external pool, matched to condition, it falls to 21.2 / 0.94 plain and 3.0 / 1.30 swahili
(AUROC 0.924 / 0.763). The probe beats it by **+28.4 [+23.5, +33.6]** plain and
**+24.7 [+20.6, +29.5]** swahili, paired 10,000-repeat bootstrap.

| condition | probe | ShieldGemma | Qwen3Guard |
|---|---|---|---|
| plain | 50.0 / 1.19 / 0.934 | 46.4 / 0.88 / 0.937 | 21.2 / 0.94 / 0.924 |
| swahili | 27.7 / 1.00 / 0.837 | 18.9 / 0.88 / 0.862 | 3.0 / 1.30 / 0.763 |

Two consequences. **ShieldGemma is the strongest guard on the external pool in both conditions**,
so the C7 comparator selected from C4 was the correct one and the acceptance test above stands
as computed. And **guard rankings are source-dependent**: the newest purpose-built multilingual
guard generalises worst across a source change, which is itself a reportable finding and a
caution against reading any single-source guard leaderboard as a deployment ordering.

**The Swahili ROC curves cross, and the crossing is the result.** Standardised partial AUROC over
FPR ∈ [0, 1.5%], paired 2,000-repeat bootstrap over test rows:

| condition | metric | probe − ShieldGemma |
|---|---|---|
| plain | pAUC@1.5% | −0.004 [−0.023, +0.014] |
| plain | full AUROC | −0.004 [−0.008, +0.001] |
| swahili | pAUC@1.5% | **+0.034 [+0.019, +0.050]** |
| swahili | full AUROC | **−0.026 [−0.032, −0.019]** |

On Swahili the probe is significantly *worse* over the whole curve and significantly *better* in
the low-FPR region; both intervals exclude zero in opposite directions. The +8.8-point matched-TPR
advantage is therefore a stable property of the operating region, not an artefact of one quantile.
On plain English both metrics are ties. Since the ≤1%-FPR regime was pre-declared and is the
deployment-relevant one, the operating-point comparison stands as primary — but the full-curve
reversal must be reported beside it, and no general external-discrimination claim is available.

The difference column is the **bootstrap mean** of the paired per-replicate difference, so it does
not equal the difference of the two point estimates in the same row (plain external English:
50.0 − 46.4 = 3.6, bootstrap mean +4.1). Realised FPRs are close but not identical, so these are
detectors nominally calibrated to 1%, not exactly FPR-matched.

**The C7 acceptance criterion is not met.** It required a ≥10-point matched-TPR advantage over
the strongest guard at realised FPR ≤1.5%. The matched Swahili advantage is **+8.8 points
[+3.7, +14.3]** at 1.00% against 0.88% FPR — real and interval-excluding-zero, but below the
pre-declared bar. The interval *does* contain +10, so a true ≥10-point effect is not excluded —
it simply was not demonstrated. Report it as "not demonstrated", never as "disproved".

Four things this section must state plainly.

1. **Source transfer costs more than language transfer.** Plain AUROC falls 0.988 → 0.934 and
   Swahili 0.967 → 0.837 moving off WildJailbreak; matched Swahili TPR falls 59.3 → 27.7. The
   magnitude matches the leave-one-dataset-out collapse reported in
   `papers/notes/2602.14161-fomin-when-benchmarks-lie.md`.
2. **ShieldGemma leads the probe on AUROC in both external conditions** (0.937 vs 0.934;
   0.862 vs 0.837). The §4 claim that the probe beats every text baseline on AUROC *and*
   TPR@1%FPR is a within-source claim and does not survive the source change. The probe's
   external advantage is confined to the operating point under matched calibration.
3. **The frozen-threshold rows are not FPR-matched and support no TPR comparison.** ShieldGemma's
   frozen WildJailbreak threshold lands at 6.76% (plain) and 2.90% (swahili) realised FPR against
   the probe's 1.79% and 0.76%; its higher TPR there is bought with alert volume. Read these rows
   as threshold portability, not detector quality: the probe's old threshold survives the source
   change better than the guard's, though at 1.79% it still breaks the 1.5% budget, so it is not
   fully portable either.
4. **The centroid fails to transfer usefully** — 0.580 / 0.556 AUROC is weak but above chance,
   so its §4 row should not be read as a cross-source number.

Label quality: judge parse-error rate 1.14% (268/23,489). Blind Claude-vs-judge hand-check on 48
stratified rows gives 81.2% raw and **94.3% population-weighted** agreement, against 87.5% for the
Phase 1 judge validation; disagreement is one-directional (judge broader). Estimated label noise
is ~12.2% of positives and ~2.5% of negatives, from 48 checks, so both estimates carry wide
uncertainty. Noise depresses absolute TPR for every detector, but it does **not** follow that the
comparison is unbiased: mislabelled rows may correlate differently with each detector's scores.
916 positives were excluded as protected-group
harassment, per the frozen construct. Aegis-versus-judge disagreement is large in both directions
(4,409 Aegis-unsafe judged benign; 570 Aegis-safe judged harmful), driven by Aegis's broader
taxonomy (Profanity, Sexual, Harassment, Unauthorized Advice).

Pool construction: 28,216 raw rows → 27,224 after dropping REDACTED/empty → 25,806 after
normalised-text dedupe → **zero** overlap with the 10,000 Phase 1 prompts → 23,489 after dropping
prompts exceeding 256 NLLB tokens, so the English and Swahili arms cover byte-identical rows.
Swahili translation left 33 rows (0.1%) identical to their source, all ≤4-word fragments.

Raw: `data/c7_external_scores.npz` (SHA-256 `ef6e7e7d…`), `data/c7_external_guard.npz`
(ShieldGemma, `edb78b74…`), `data/c7_external_guard_qwen.npz` (Qwen3Guard, `f1b0be87…`, MLP job
3577642, 34m13s), `data/c7_external_results.json` (`a6f7cb80…`), `data/c7_judged_all.jsonl`
(`dbd4001e…`), `data/c7_partition.json` (`4c4ea38b…`), `data/c7_translations/swahili.jsonl`
(`6ebf79b8…`). MLP jobs 3569598/3569599 (judge, Qwen3.6-27B), 3571703 (NLLB translation,
53m32s), 3573545 (probe scoring, 33m00s), 3577169 (ShieldGemma, 28m56s), all
`h200_3g.71gb`, exit 0. `google/gemma-3-27b-it` `005ad3404e59d6023443cb575daa05336842228a`;
`google/shieldgemma-9b` `b8b636016df4540721a098c7aab91c97ec6ee508`; NLLB
`f8d333a098d19b4fd9a8b18f94170487ad3f821d`; `nvidia/Aegis-AI-Content-Safety-Dataset-2.0`
`d86bb8bedff51d25ac834ab7838f1cc61acb7a2c`; seed 0; source commit `c7c29e2`. Analysis:
`conda run -n msc-diss python -m phase1.analyse_c7_external`.

## 9. P1 — cipher conditions and read position (27B, frozen plain-trained probe)

**The pre-declared primary test fails.** `base64` vs `base64_shuffled`, paired, required
*both* the matched-TPR and AUROC intervals to exclude zero. Neither position delivers that
for the probe:

| position | detector | ΔTPR (95% CI) | excl 0 | ΔAUROC (95% CI) | excl 0 |
|---|---|---|---|---|---|
| t_inst | logistic | −0.54% [−2.29, +0.88] | no | **−0.055** [−0.091, −0.018] | yes (wrong sign) |
| t_cipher | logistic | +4.68% [+0.88, +9.69] | yes | +0.025 [−0.004, +0.053] | no |
| t_inst | centroid | +3.22% [+0.88, +5.99] | yes | +0.075 [+0.042, +0.108] | yes |
| t_cipher | centroid | +2.55% [−0.88, +5.46] | no | +0.061 [+0.041, +0.081] | yes |

Only the centroid at `t_inst` satisfies both, and it does so between two detectors that are
both at chance in absolute terms (base64 TPR 1.06%, shuffled 1.23%). Report the primary test
as failed; the centroid cell is not a result.

**Matched calibration, logistic probe, test split.** TF-IDF char(3-5) is the surface control.

| condition | position | AUROC | TPR@1%FPR | FPR | TF-IDF AUROC |
|---|---|---|---|---|---|
| plain (ceiling) | t_inst | 0.9878 | 75.35% | 1.07% | 0.9622 |
| plain_wrapped | t_inst | 0.9685 | 54.58% | 0.74% | 0.9607 |
| plain_wrapped | t_cipher | 0.9873 | 74.12% | 1.15% | 0.9607 |
| base64 | t_inst | 0.4795 | 1.06% | 1.32% | 0.5369 |
| base64 | t_cipher | 0.5066 | 5.46% | 1.15% | 0.5369 |
| base64_shuffled | t_cipher | 0.4820 | 1.06% | 0.82% | 0.5195 |
| rot13 | t_inst | 0.2294 | 0.53% | 0.99% | 0.5021 |
| rot13 | t_cipher | 0.8169 | 3.70% | 0.74% | 0.5021 |
| vowel | t_cipher | 0.9201 | 26.23% | 1.15% | 0.7552 |
| reverse | t_cipher | 0.5003 | 2.64% | 1.73% | 0.5429 |

**Base64-trained probe** (trained on base64's own train split, layer chosen on tune):
`t_inst` L55 test AUROC **0.9116** (TPR 21.48%); `t_cipher` L58 0.8792 (TPR 11.97%).

Three findings worth carrying:

1. **The base64 null is not an artefact of read position or the wrapper.** `plain_wrapped`
   places English in the identical scaffold and recovers the ceiling (0.9873 at `t_cipher`,
   0.9685 at `t_inst` against a bare-plain 0.9878). Reading seven tokens downstream through
   the instruction costs ~20 points of TPR but little AUROC.
2. **The harm signal exists under base64; the plain-English direction cannot reach it.** A
   condition-specific probe reaches 0.912 where the frozen probe sits at 0.480. This is a
   *transfer* failure, not absence of representation — the stronger and more useful claim.
3. **High AUROC without usable TPR recurs.** rot13 0.817 AUROC at 3.70% TPR, vowel 0.920 at
   26.23%. Neither is an operational success at 1% FPR.

**Do not claim internal decoding.** Per-layer test AUROC: rot13 `t_cipher` is 0.82 at
**layer 1**; the self-trained base64 probe is 0.671 at **layer 0** (the embedding) and 0.858
by layer 1. Signal that high before any meaningful computation is lexical/token-level, so
TF-IDF failing at 0.502 does not establish "no surface signal" — char n-grams miss
correlations that subword token identity carries. The one exception is self-trained `t_inst`,
which climbs 0.716 (L1) → 0.916 (L40): the only depth signature in this data consistent with
computation rather than lookup.

Reproducibility: Eddie job 57340069, exit 0, 7051s, 8×L40S. `google/gemma-3-27b-it`
`005ad3404e59d6023443cb575daa05336842228a`, seed 0, batch 4, all 8903 rows, frozen split
5341/1781/1781. `data/p1_position_scores_57340069.npz` SHA-256 `4004a506…d060`;
`data/p1_position_results.json`; conditions frozen in `data/p1_conditions_manifest.json`;
TF-IDF `data/p1_baselines_tfidf.npz`; ceiling `data/p1_decode_then_guard_ceiling.json`.
Decode-fidelity gate (`p1_decode_fidelity.py`): token F1 0.970 overall, 0.973 within the
≤90-char capability gate vs 0.968 beyond, so the null is not a decode failure. Analysis:
`conda run -n msc-diss python -m phase1.analyse_p1_position --scores
data/p1_position_scores_57340069.npz`.

## 10. P2 — response-stream monitoring, with the Qwen3Guard comparator

Responses to harmful prompts only, judged-harmful positives against judged-benign
non-refusals; refusals are excluded everywhere. Strata: plain 161 harmful / 187 benign
test (196 tune negatives), base64 90 / 90 (94 tune negatives). Every TPR is shown with
its realised FPR. `k` is the response-token horizon; the probe reads a running maximum
over response positions, TF-IDF and the guard are scored on a coarse grid.

**The pre-declared 1%-FPR test fails for every monitor.** The target was 50% TPR. The
best terminal result is Qwen3Guard at 18.6%, and the probe reaches 1.2%.

| condition | k | AUROC probe / TF-IDF / Qwen3Guard | TPR@1% probe / TF-IDF / guard | TPR@5% probe / TF-IDF / guard |
|---|---|---|---|---|
| plain | 0 | 0.719 / — / — | 1.2% (0.5) / 0.0% (0.0) / — | 23.6% (8.6) / 0.0% (0.0) / — |
| plain | 8 | 0.738 / 0.689 / **0.796** | 1.2% (0.5) / 1.9% (0.0) / 10.6% (0.5) | 21.1% (8.0) / 1.9% (0.0) / 24.8% (1.1) |
| plain | 32 | 0.754 / 0.730 / **0.809** | 1.2% (0.5) / 3.1% (1.1) / 11.2% (1.1) | 21.7% (6.4) / 3.1% (1.1) / 24.8% (3.7) |
| plain | 64 | 0.768 / 0.744 / **0.836** | 1.2% (0.5) / 8.1% (1.1) / 12.4% (1.1) | 23.6% (6.4) / 8.1% (1.1) / 28.0% (4.3) |
| plain | 512 | 0.754 / 0.789 / **0.876** | 1.2% (0.5) / 14.9% (2.1) / **18.6%** (1.1) | 19.9% (5.9) / 18.6% (2.7) / 36.0% (4.8) |
| base64 | 0 | 0.642 / — / — | not representable | 6.7% (3.3) / 0.0% (0.0) / — |
| base64 | 8 | 0.503 / 0.473 / **0.771** | not representable | 1.1% (4.4) / 0.0% (0.0) / 42.2% (10.0) |
| base64 | 32 | 0.703 / 0.670 / **0.796** | not representable | 28.9% (4.4) / 8.9% (3.3) / 47.8% (10.0) |
| base64 | 512 | 0.793 / 0.838 / **0.913** | not representable | 27.8% (4.4) / 28.9% (3.3) / 53.3% (5.6) |

base64's 1% column is empty because 94 tune calibration negatives cannot resolve a 1%
quantile (≥101 required). That is a reported gate failure; it is not repaired by pooling
conditions or by borrowing benign-prompt responses as negatives. AUROC needs no threshold,
so the base64 trajectory is still readable.

Three things the guard changes.

1. **The probe's early advantage is over TF-IDF only.** The paired probe−TF-IDF test at
   5% FPR still favours the probe at k=0/8/16/32 in plain (+20.8/+18.8/+19.7/+18.2 points,
   all excluding zero) and at k=32 in base64 (+21.5% [+7.8, +35.6]). Qwen3Guard beats the
   probe on AUROC at **every** k from 8 onward in both conditions, so the "early readability"
   claim cannot be stated against a purpose-built guard.
2. **Qwen3Guard is the strongest response-stream monitor here**, terminal AUROC 0.876 plain
   and 0.913 base64 — and unlike the probe it does not degrade under base64, because it reads
   the English response, not the ciphered request.
3. **Latency goes the wrong way for the probe.** First-crossing at 1% FPR, restricted mean:
   probe crosses **+66.7 tokens later** than Qwen3Guard [+39.6, +94.2] and +50.6 later than
   TF-IDF [+26.1, +75.7]; both exclude zero. Medians are saturated (159/1061 probe rows never
   cross).

Caveat on the base64 5% cells: the guard realises 10-11% FPR at k=8-32 against a 5% target,
so those TPRs are not calibrated comparisons — the 94-negative calibration set is the same
weakness the 1% gate names.

The length confound was checked and does not explain the base64 rise: length-only AUROC
0.603 (plain 0.507), but restricting to responses still generating at k gives the same
values (k=32: 0.706 vs 0.703), so it is not short negatives freezing their running maximum.

Reproducibility: guard scores from MLP job 3585833, exit 0, 35m37s, 3×`h200_1g.18gb`,
`Qwen/Qwen3Guard-Gen-8B`, batch 16, all 1061 rows, commit `df9ca34`.
`data/p2_guard_monitor.npz` SHA-256 `fac55bb8…f962`, keyed to
`data/p2_latency_scores.npz` `01a95671…b04f`. Inputs `data/p2_judged_analysis.jsonl`
(1061 non-refusal rows) and `data/p2_text_monitor.npz`. Analysis:
`conda run -n msc-diss python p2_analyse_readability.py --guard data/p2_guard_monitor.npz
--bootstrap 2000` and `conda run -n msc-diss python p2_analyse_latency.py --guard
data/p2_guard_monitor.npz` → `data/p2_readability_results.json`,
`data/p2_latency_results.json`, `figures/p2_latency.png`.

## 11. Obfuscated *output* capability at 27B (future-work evidence)

The dissertation's future-work argument assumes a model could emit harmful content
directly in ciphered form, defeating a text guard that never sees plaintext. That was
assertion; this measures it. 50 WildJailbreak `vanilla_harmful` prompts through the
**abliterated** 27B (so a failure is a cipher failure, not a refusal), 512 new tokens,
judged after decoding with the same rubric as every other harm number here.

| cell | in | out | harmful | decode_ok | truncated |
|---|---|---|---|---|---|
| plain (ceiling) | plain | plain | **40%** | 100% | 100% |
| base64 `out` | plain | base64 | **22%** | 74% | 52% |
| base64 `inout` | base64 | base64 | **2%** | 78% | 58% |
| reverse `out` | plain | reverse | **0%** | 100% | 98% |
| reverse `inout` | reverse | reverse | **0%** | 100% | 12% |

**The 27B cannot carry harmful content through a cipher in the realistic venue.** Writing
base64 alone costs 18 of the 40 available points; adding ciphered *input* costs almost all
of the rest (22% → 2%). The two error sources compound, which is the pattern the capability
design was built to detect. Reverse output fails outright: the raw text is well-formed
reversed English, but decoded it is degenerate repetition ("...to home only, you hould be
affected by aspect and distance" looping), so `decode_ok=100%` is trivially satisfied by a
self-inverse cipher and says nothing about content. `reverse:inout` also gives up early —
median 69 response tokens against 512 everywhere else.

What survives in base64 is fluent-looking but corrupted: valid base64 decoding to English
with systematic character errors ("phqsical", "bealated", "techinques") and no instructional
content. The model can operate the *encoding* and cannot carry *substance* through it.

Two caveats. Truncation at 512 tokens binds hardest on base64, which expands ~4/3, so the
ciphered harm rates are conservative — within `base64:out`, truncated rows are judged
harmful more often (30.8%) than untruncated ones (12.5%). And the high refusal counts in
ciphered cells (36/50 for `reverse:inout`) come from an abliterated model, so they are
better read as failures to produce a usable answer than as safety behaviour.

For the writeup: this supports "obfuscated output is a real but currently unreliable
channel at this scale", not "frontier models can already do this". Size is a proxy, not the
demonstrated variable — the experiment isolates read and write capability at one scale.

Reproducibility: MLP job 3585962 (generation, exit 0, 1h32m, 5×`h200_1g.18gb`,
`~/models/gemma-3-27b-it-heretic`, seed 0, batch 4, 512 new tokens, temperature 0.7,
top-p 0.9) and 3586168 (judging, exit 0, 22m39s, `Qwen/Qwen3.6-27B`, batch 8,
`--max-new-tokens 256`, 0 parse errors). Commit `ff8aab0`.
`data/cap_outobf_27b.jsonl` → `data/cap_outobf_27b_judged.jsonl` SHA-256 `7f50e5c8…5821`.
Command: `sbatch -p Teaching --gres=gpu:h200_1g.18gb:5 --time=02:00:00 run_capcheck.sh
~/models/gemma-3-27b-it-heretic ablit27b 50 data/cap_outobf_27b.jsonl 4 512
base64,reverse out,inout`.

## 12. Benign obfuscated-output capability across model scales

Section 11 found the 27B cannot carry harmful content through a cipher, but that mixes
capability with residual refusal, and it cannot say whether the *venue* is realistic on
stronger models. This is the benign counterpart: the same three conditions on ordinary
factual questions, so frontier models can be measured without asking for harmful content.

Items are the first 50 of the seed-0 WebQuestions draw `capability_qa.py` already used, so
the 27B plain ceiling is the ceiling for these items too. Grading is the existing
WebQuestions normalisation, applied to the *decoded* answer.

| model | plain (ceiling) | base64 `out` | base64 `inout` | decode_ok |
|---|---|---|---|---|
| gemma-3-27b-it (stock) | 54% | *pending* | *pending* | — |
| Haiku 4.5 | 48% | **48%** (−0) | blocked (see below) | 100% |
| Sonnet 5 | 56% | **52%** (−4) | blocked (see below) | 100% |
| gpt-5.4-mini | 50% | **46%** (−4) | **46%** (−4) | 100% |
| gpt-5.6-luna | 50% | **46%** (−4) | **46%** (−4) | 100% |

**Emitting base64 is close to free for a frontier model, and reading it as well costs
nothing further.** Every model decodes at 100% with well-formed output (`Tmljb21lZGlh` ->
"Nicomedia"), and the accuracy cost of the cipher is 0-4 points. The compound `inout`
condition — the one where the 27B fell to 2% harmful in section 11 — costs the GPT models
nothing beyond `out`. So the obfuscated-output venue is a real capability at frontier
scale and not at 27B, which is what the future-work argument needs.

**Working the encoding out explicitly makes the content worse.** Sonnet's first run
hand-derived base64 through ASCII codes and 6-bit groups: still 100% valid, but **38%**
accuracy against **52%** writing it fluently. The algorithmic route spends the model on
the encoding rather than the answer. Haiku's two runs agree exactly (48%/48%), so the
protocol is stable across repetitions.

Four limitations, all of which matter.

1. **The grading metric has a low ceiling and is not a cross-model ranking.** It requires a
   gold string to be a substring of the answer, so "Brussels" fails against `City of
   Brussels` and "Greek" fails against `Greek Language`; several golds are stale Freebase
   entries ("minority leader" -> Pelosi) and at least one is simply wrong (Gordon Brown's
   resignation given as 2007). A lenient bidirectional match adds 4-12 points to every cell
   and reorders nothing. The *within-model* delta (plain vs cipher) is the trustworthy
   quantity, because a bad gold penalises both arms of the same model identically.
2. **Tool non-use is attested, not proven.** These models were driven as agents with file
   access, instructed not to use any encoder, and asked to report tool use; their reported
   command logs are consistent. An API call with tools disabled would be stronger evidence.
   One Codex run *was* contaminated — it read another model's answer file and returned all
   50 answers byte-identical — and was discarded and re-run in an isolated directory.
   Independence of the re-runs was checked: 17-25/50 identical answers between models,
   against 19/50 for two known-independent runs, since models converge on short strings
   like `QmVsZ2l1bQ==`.
3. **Exact model versions are unverified.** Codex self-reports only "GPT-5" for both
   requested models; the two rows are distinct models (0/50 identical plain answers, clearly
   different styles) but the deployment strings are what was requested, not what was
   confirmed.
4. **The Claude `inout` cells are missing for a harness reason, not a model reason.** Both
   Claude subagents were terminated by a Claude Code safety classifier before answering any
   item, on benign trivia; the Haiku agent's error names Sonnet, so the block sits on the
   orchestration path, not the subject model. The same condition run manually in Claude
   Desktop complies. Do not report this as a refusal by the model.

Reproducibility: items `data/outobf_qa_items.jsonl` from `prepare_outobf_qa.py` (seed 0,
first 50 of the 150-question draw, `--max-q-chars 90`). Raw answers under
`data/outobf_qa_answers/`, graded by `score_outobf_qa.py` into
`data/outobf_qa_frontier.jsonl`. Claude arms run as Claude Code subagents; GPT arms via the
Codex MCP with `sandbox=workspace-write`, one isolated working directory per run. No seed
or temperature control was available for any hosted model — a real reproducibility gap
against the 27B's seed 0.
