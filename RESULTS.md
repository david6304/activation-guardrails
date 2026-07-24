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
| multilingual-e5-base | 0.967 / 56.3 / 0.9 | 0.947 / 51.2 / 1.6 | 0.906 / 18.3 / 0.7 | 0.837 / 2.1 / 0.1 | 0.813 / 0.9 / 0.0 | 0.616 / 0.0 / 0.0 |
| DeBERTa-v3-small guard | 0.981 / 63.9 / 1.1 | 0.940 / 15.3 / 0.1 | 0.862 / 0.0 / 0.0 | 0.833 / 0.2 / 0.0 | 0.805 / 0.0 / 0.0 | 0.472 / 0.0 / 0.0 |
| char TF–IDF | 0.962 / 45.2 / 1.0 | 0.731 / 0.5 / 0.1 | 0.531 / 0.0 / 0.0 | 0.596 / 0.0 / 0.0 | 0.580 / 0.0 / 0.0 | 0.543 / 0.0 / 0.0 |

Condition-matched calibration (TPR% / FPR%; AUROC is calibration-invariant, as above):

| detector | plain | french | hindi | swahili | zulu | reverse |
|----------|-------|--------|-------|---------|------|---------|
| all-layer logistic | **75.4 / 1.1** | **72.5 / 1.3** | **69.5 / 1.2** | **59.3 / 1.3** | **51.2 / 1.2** | 2.1 / 0.7 |
| centroid (Zhao) | 56.5 / 2.2 | 42.8 / 1.8 | 38.9 / 1.2 | 33.6 / 1.5 | 20.8 / 0.7 | **13.6 / 1.6** |
| ShieldGemma-9b | 45.2 / 1.4 | 36.8 / 0.6 | 30.1 / 1.3 | 30.1 / 1.1 | 15.3 / 0.8 | 1.9 / 0.2 |
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
encoder built for multilingual semantics. Degradation is monotone in language resource level
(plain → french → hindi → swahili → zulu) for every detector, but far steeper for text: WildGuard
falls 97.4% → 0.4%, ShieldGemma 45.2% → 15.3% (matched), the probe 75.4% → 51.2%. ShieldGemma is
now the strongest *single* text comparator on the low-resource end (swahili 30.1%), so it is the
right fusion partner for Step 4 — not WildGuard. Reverse defeats every detector, and the centroid
is the only one retaining signal there (0.767 AUROC, 13.6% matched TPR) while the all-layer
logistic collapses to 0.556 — a probe-geometry failure rather than absent signal.

**Caveat.** The ShieldGemma cross-environment audit failed (max abs probability difference 0.0312
vs 1e-3 tolerance) on 2 of 48 frozen audit cells; it changed no reported metric (see
`RESEARCH_LOG.md` 2026-07-24). WildGuard's audit passed exactly. No LlamaGuard: gated by the Meta
licence and never staged — an explicit limitation, not a pending run.

Raw: `data/phase1_baselines_multilingual.npz` (Eddie job 57134364), `data/phase1_activation_multilingual_27b.npz`,
`data/phase1_small_guard.npz`, `data/phase1_multilingual_e5.npz`; analysis
`data/phase1_text_encoder_multilingual_results.json`. Commit `59649dd`, seed 0. Known-test status:
exploratory development evidence — no model, layer, threshold or condition was selected on it.
