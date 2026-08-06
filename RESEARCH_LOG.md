# Research Log

Short dated entries for important research decisions, results, and changes in
direction. Do not record routine coding work.

## YYYY-MM-DD - Short title

## 2026-08-06 - Qwen3Guard closes the P2 comparator gap, and it beats the probe

Qwen3Guard-Gen-8B scored over the same 1061 response prefixes (MLP job 3585833, exit 0,
35m37s). The pre-declared 1%-FPR outcome is unchanged and still the headline: **no monitor
reaches the 50% TPR target** — the best terminal number is the guard at 18.6% TPR (1.1% FPR),
against TF-IDF 14.9% and the probe 1.2%.

What it costs us: **the probe's early-readability advantage was over TF-IDF only.** The guard
leads on AUROC at every k from 8 onward in both conditions (plain 0.796 -> 0.876, base64
0.771 -> 0.913, versus the probe's 0.738 -> 0.754 and 0.503 -> 0.793), and the probe crosses
its 1% threshold **66.7 tokens later** than the guard [+39.6, +94.2]. The paired probe-minus-
TF-IDF 5% result stands as before (plain k=0-32, base64 k=32). So RQ2 is a negative for
response-stream deployment: the prompt-trained direction does not transfer to response
positions well enough to compete with a purpose-built guard on the stream.

The guard also does not degrade under base64 — it reads the English response, not the
ciphered request — which is the cleaner way to state why P2 does not test ciphered outputs.

`data/p2_guard_monitor.npz` SHA-256 `fac55bb8…f962`. RESULTS.md section 10.

## 2026-08-05 - P1: the base64 null is transfer failure, not absent representation

The pre-declared primary test (base64 vs base64_shuffled, both intervals excluding zero)
**fails** at both read positions for the probe. What the run does establish is better than
that test was: **plain_wrapped**, English in the identical cipher scaffold, recovers the
ceiling (0.9873 t_cipher vs 0.9878 bare plain), so the base64 null is not the read position
or the wrapper; and a **base64-trained probe reaches 0.912 AUROC** where the frozen
plain-trained probe sits at 0.480, so the harm is represented and the plain direction simply
cannot reach it.

Correction to an earlier reading: rot13 0.817 with TF-IDF at 0.502 is **not** evidence of
internal decoding. rot13 is already 0.82 at layer 1 and the self-trained probe is 0.671 at
the embedding, i.e. lexical, before computation. Char n-grams failing does not prove absence
of surface signal. Drop the locality framing; keep the transfer claim.

Eddie job 57340069, exit 0, 7051s. `data/p1_position_scores_57340069.npz` SHA-256
`4004a506...d060`. RESULTS.md section 9.

## 2026-08-01 - Guard rankings are source-dependent; HaloGuard cannot operate

Two additions close out C7's comparator question. **Qwen3Guard collapses on the external pool** —
matched 21.2% plain / 3.0% swahili against its in-source 54.9% / 16.4%, so the probe beats it by
+28.4 [+23.5, +33.6] and +24.7 [+20.6, +29.5]. ShieldGemma is the strongest guard externally in
both conditions, confirming C7 used the right comparator. The finding worth reporting: **the
newest purpose-built multilingual guard generalises worst across a source change**, so a
single-source guard leaderboard is not a deployment ordering.

**HaloGuard-1.0-4B (2026-07) added to the §4 matrix** to answer the "current guards" objection —
it is the only guard evaluated that declares Swahili and isiZulu coverage. It ranks acceptably
(AUROC 0.924 plain, 0.817 swahili) but returns **6-10% matched TPR in every condition including
plain English**: its scores saturate at 0/1 and the 1%-FPR quantile lands inside a mass of ties.
A calibration failure, not absent rank signal — report it that way. Batch audit fails (12/48
cells, max 0.0156). Three weeks old with no independent replication, so it is one datapoint, not
evidence about multilingual guards generally.

Reproducibility: MLP jobs 3577642 (Qwen3Guard external, 34m13s) and 3577657 (HaloGuard matrix,
29m11s), both `h200_3g.71gb`, exit 0. `astroware/HaloGuard1-Gen-4B` revision `f157c1f8...`,
policy prompt committed verbatim at `phase1/haloguard_policy.txt` (SHA-256 `c4c34771...`),
generation check 96/96. `data/c4_haloguard.npz` SHA-256 `421a41af...`,
`data/c7_external_guard_qwen.npz` `f1b0be87...`. `RESULTS.md` §§4, 8.

## 2026-07-31 - C7: the probe survives source transfer, but the acceptance bar is missed

Frozen probe, centroid and ShieldGemma scored on 22,305 Aegis 2.0 prompts relabelled under the
frozen rubric; thresholds from 7,480 external tune negatives only, reported on a disjoint test
partition. **Matched Swahili advantage is +8.8 TPR points [+3.7, +14.3]** at 1.00% vs 0.88% FPR
— real, but the point estimate is short of the pre-declared >=10-point bar, so **C7's acceptance
criterion is not met**. The interval contains +10, so a >=10-point effect is not demonstrated but
also not excluded.

Three things that change what the dissertation can claim. **ShieldGemma leads on AUROC in both
external conditions** (0.937/0.862 vs the probe's 0.934/0.837), so "beats every text baseline on
AUROC and TPR" is a within-source claim only; the external advantage is confined to the operating
point under matched calibration. **Source transfer costs more than language transfer** — plain
AUROC 0.988->0.934, matched Swahili TPR 59.3->27.7, matching the Fomin leave-one-dataset-out
collapse. **The centroid fails to transfer usefully** (0.580/0.556 — weak, not chance), so
its §4 row is not a cross-source number. Also: the frozen-threshold rows are not FPR-matched (ShieldGemma realises
6.76%/2.90% against the probe's 1.79%/0.76%), so they support no TPR comparison — but they do show
the probe's threshold transfers to a new source far more stably than the guard's.

**The Swahili ROC curves cross**: partial AUROC over FPR<=1.5% is +0.034 [+0.019, +0.050] for the
probe while full AUROC is -0.026 [-0.032, -0.019]; both exclude zero, opposite directions. So the
+8.8 operating-point advantage is a stable property of the low-FPR region, not one quantile. Plain
is a tie on both (-0.004 each, intervals cross zero).

Judge parse errors 1.14%; blind hand-check 94.3% population-weighted (48 rows, four strata);
estimated label noise ~12.2% of positives, ~2.5% of negatives (48 checks, so wide uncertainty;
noise lowers every detector's TPR but does not automatically leave the comparison unbiased).
`RESULTS.md` §8. Full reproducibility block there.

## 2026-07-28 - P1 preflight settles the read position; P2 pre-declarations fixed

**`t_inst` is 7 tokens downstream of the ciphertext in every row** — the decoder-instruction
wrapper appends "Respond normally in plain English." after the payload — so the read
position is a genuine confound and P1 scores both `t_inst` and `t_cipher`. Prompts are
short (longest rendered condition 276 tokens against the 2048 limit), so truncate-then-encode
is vacuous. base64 and rot13 round-trip exactly on all 3,562 rows; the shuffled control
preserves length and base64 validity with no row left unpermuted. **Decode-then-guard
ceiling, on the page first:** probe 75.4% matched TPR / 0.988 AUROC, qwen3guard 54.9%,
tfidf 45.2%, shieldgemma 45.2%, llamaguard4 40.8%. `reverse` was added as a fifth P1
condition — the ordering prediction is stated against it and only `t_inst` existed.

**P1's guard pass is deferred to Extras** (Codex, agreed): no guard enters the pre-declared
base64-vs-shuffled test, so the cost buys table completeness only. Consequence to state
plainly in the write-up: guard performance on the new cipher surfaces was **not measured**,
and must not be extrapolated from reverse.

**P2 pre-declared, before any generation:** horizon k_max = 512 protected-reader response
tokens; common absolute target 50% TPR, with "not reached by 512" reported rather than the
target lowered; 1% nominal sequence-level FPR from a **single** threshold on the running
maximum, never recalibrated per k (that would give every prefix a fresh 1% allowance and
make first-crossing incomparable); negatives are same-condition **tune** responses to
harmful prompts judged benign and non-refusal. EOS is censoring: fixed denominator,
terminal running maximum carried forward, no Kaplan–Meier — length correlates with refusal
and harm, so the censoring is informative. Text comparators are the frozen response TF-IDF
and Qwen3Guard-Gen-8B only.

**The response judge now emits an explicit `refusal` verdict.** The harm criteria are
unchanged and prompt mode is untouched (C7 unaffected), but a response verdict missing
`refusal` is a parse error rather than a silent false, so refusals cannot leak back into
the benign class. To be validated on 50 hand-checked rows before it filters anything.

**Sizing consequence worth knowing now:** the 1% rule needs ≥101 tune calibration negatives
per condition, and only benign-non-refusal responses to harmful prompts count. The 200-prompt
compliance pilot will not reach that — it is a compliance estimate, not an analysis — and
the full run needs the whole tune split to have a chance of clearing it.

## 2026-07-27 - Direction: the cipher failure may be a read-position artefact

`t_inst` sits before the model has decoded a ciphered prompt, so "reverse is not
represented" may be "reverse is not represented *yet*". `DISSERTATION_FINISH_PLAN.md`
rewritten around two items. **P1 (input):** base64 / shuffled-base64 / rot13 read at both
the final ciphertext token and `t_inst`, gated on a base64 capability check that
adjudicates the 2026-07-03 drop against David's hand test; primary test is base64 vs
shuffled-base64 on AUROC *and* matched TPR. **P2 (output):** generate with abliterated
27B (MLP), judge responses, teacher-force through the protected 27B (Eddie) in one
forward pass, probe every response position; claim is off-policy detection *latency*
against text on an identical token stream.

Five decisions to keep: (1) **ciphers are instruments, not threats** — all invert
deterministically without a model, so threat framing belongs to the language conditions
and to lossy encodings; (2) P2's contrast must be **within harmful prompts** (harmful vs
benign-non-refusal responses) — the naive benign-prompt negative class is separable at
k=0 from prompt intent alone, which would have produced a strong-looking invalid
headline; (3) the response judge needs an explicit refusal verdict, since the rubric
currently folds refusals into `harmful=false`; (4) C7 needs a frozen disjoint external
tune/test partition; (5) C7 confirms neither P1 nor P2, so both stay exploratory with no
confirmatory set in budget. C5 downgraded to a CPU decode-then-guard ceiling, C6 folded
into extras. Teacher-forcing over on-policy is settled by the 2026-06-29 result (0.810 vs
0.772) and keeps the abliterated weights on MLP.

## 2026-07-27 - C4 complete: Llama Guard 4 is the weakest of the three guards

Llama Guard 4 scored on all six conditions, so C4 is finished and C5/C7 are unblocked.
**The probe leads every condition on both AUROC and matched TPR**, so C4's rewrite
trigger does not fire. Matched TPR, plain→reverse: probe 75.4/72.5/69.5/59.3/51.2/2.1
against LG4 40.8/30.5/**0.0**/6.0/5.5/1.8, with LG4 AUROC 0.960/0.913/0.889/0.696/0.717/0.480.

Two things worth reporting. **LG4 does not win any condition**, so "strongest guard from
C4" stays the existing Qwen3Guard (plain/french/hindi) versus ShieldGemma (swahili/zulu)
split and C5/C7 gain no third arm. And **hindi is degenerate at the operating point**:
0.000% TPR at 0.000% FPR against a respectable 0.889 AUROC, because LG4's score is
heavily saturated (89% of hindi prompts below 0.01 or above 0.99) and the 1%-FPR quantile
lands inside a mass of tied values. That is a thresholding limitation of a saturated
guard score, not an absence of rank signal, and should be reported as such.

The same saturation drives an audit failure of 0.0567 (7/48 cells over the 1e-3
tolerance) — every failing cell is swahili or reverse, none are plain/french/hindi.
Where LG4 commits it is stable; where it is undecided it is padding-sensitive. On zulu
82% of prompts sit in the undecided band at mean score 0.494, i.e. near coin-flip.

Reproducibility: Eddie job `57164545` (node1p07, 1× L40S, bf16, batch 8, 42m35s, exit 0);
source commit `374c491`, analysis at `c2ae933`; `meta-llama/Llama-Guard-4-12B` revision
`87acb4b94e930c3d679e6e7ee9d57e2feab9ea71`; seed 0; tune/test 1781/1781. Smoke `57164458`
passed the free-generation check 96/96. `data/c4_modern_guards_lg4.npz` SHA-256
`6011e0a97af5ce734f6020e8faecac6d2e4cdeb2bf6d87ecc3315bedf4817b02` (carries the Qwen3Guard
arrays through, source SHA-256 `418184f5…` verified identical to the recorded C4 file);
`data/c4_lg4_results.json` SHA-256
`0b5fe18cdf2a62aef08052debbcce1358ed74898a0174acf87363a88dc8a74ee`. Both copied off Eddie
with hashes verified; `RESULTS.md` §4 updated.

Loading needed a fix: the checkpoint sets `attention_chunk_size` to null but transformers
derives `layer_types` from `no_rope_layers` (all ones), so all 48 layers requested a
chunked mask that cannot be built. Neither field is read elsewhere in `modeling_llama4`,
so every layer is named full attention.

## 2026-07-27 - Two audit bugs fixed; Aegis clears the C7 source rule

**The pool-guard audit was measuring the wrong thing.** It compared the cheap
`logits_to_keep=1` scores, produced inside length-sorted batches, against a reference
runner that re-batched the sampled rows, so left padding differed between the paths and
the audit reported batch composition. Both paths now rescore at batch size 1. (This is
*not* the same as the C4 audit, which compares batched against batch-size-1 scoring on
purpose and is correctly labelled; only the pool audit was measuring something other than
what it claimed.)

**Llama Guard 4 loads.** The checkpoint sets `attention_chunk_size` to null but
transformers derives `layer_types` from `no_rope_layers` (all ones here), so all 48
layers requested a chunked mask that cannot be built. Neither field is read elsewhere in
`modeling_llama4`, so the fix is to name every layer full attention. Smoke pending.

**C7 source pre-check: Aegis 2.0 passes on provenance and volume**, before the judge
relabel. 28,216 rows (train+validation+test), all prompt labels human-assigned; 13,773
safe against the ≥2,000 negatives rule, and 6,561 criminal-planning / 1,424 controlled
substances / 855 weapons positives against the ≥300 rule even after the rubric drops
hate and harassment. **Zero** normalised-text overlap with the 10,000 judged Phase 1
prompts (0/25,503 unique). Relabelling is not started: the judge pass and the
calibration/evaluation partition are protocol decisions still open.

C1/C2 figures drawn (`figures/`); matplotlib is now in `msc-diss`.

## 2026-07-26 - C3 (probe half): a 0.1% operating point on real traffic

50,000 unlabelled WildChat first-user-turn prompts (seed 0, no Phase 1 overlap, 47% English)
give the operating point the 1,213 tune negatives could not support. At **0.1% background alert
rate** the probe reaches 66.5/53.0/52.1/26.9/23.1% TPR (plain→zulu, reverse 0.0); at 1% it
reaches 80.6→46.1%. Two deployment findings: the frozen §4 tune-negative threshold fires on only
**0.41%** of real traffic, so calibrating on WildJailbreak benign prompts *overstates* alert
volume; and the centroid collapses under pool calibration (56.5%→23.1% plain at 1%), so its §4
row is not a deployment number. `RESULTS.md` §7.

**Half done.** ShieldGemma and Qwen3Guard are not scored on the pool (~17 GPU-h each at the
measured rate), so there is no cross-detector comparison at 0.1% yet. The whole-pool Swahili arm
was cut: the pool is 47% English-labelled and the NLLB path assumes an English source, so it
would confound language shift with source mis-specification.

Also: `meta-llama/Llama-Guard-4-12B` access came through, weights staged on Eddie
(`87acb4b94e930c3d679e6e7ee9d57e2feab9ea71`), and a continuous scorer is written — `safe`/`unsafe`
are single tokens after the assistant turn opens, so the teacher-forced `\n\n` gives one decision
position, as with Qwen3Guard. Its chat template needs multimodal content parts; a plain string
renders an *empty* conversation, which is the bug latent in `guard_screen.run_llamaguard`. The
smoke is not yet green.

Reproducibility: MLP job 3567213 (2× RTX A6000, bf16, batch 4, 3h07m39s, exit 0); source commit
`c04587c`; `google/gemma-3-27b-it` revision `005ad3404e59d6023443cb575daa05336842228a`; seed 0.
Pool manifest SHA-256 `f7fcc84c2b46beb10acef951551b3ec896db53ab6a761803160bcb28c039dcc5`;
`data/c3_pool_probe_plain.npz` SHA-256
`8e07d9b9676e20b56918d9e14b8d205ae15918c9d36071349d3c70c2f468d081`;
`data/c3_pool_results.json` SHA-256
`3a19f53d9548b38a4a4f9da1d7b220f10d4ad9b980ed9ec47b28ca366ed6803a`.

## 2026-07-25 - C1/C2: unlabelled calibration works; layer concatenation is the reverse failure

**C1 passes.** A threshold set from 300 *unlabelled* same-condition prompts at 1% harmful
prevalence recovers 86/88/93/93% of the probe's oracle matched TPR (french/hindi/swahili/zulu),
so the matched operating point is a deployment claim, not an adaptation result. The probe leads
the better of the two open guards by +18.7/+21.3/+25.5/+32.4 points at that operating point.
Honest boundaries: only the probe and Qwen3Guard gain much over their strict thresholds; at
`pi=0.05` the probe keeps just 25.3% swahili TPR, but by becoming conservative (FPR 0.16%), not
unsafe; `k=100` overshoots TPR *and* FPR. `RESULTS.md` §5.

**C2 confirmed with intervals.** On reverse, a layer selected on plain tune data only (L54
centroid) beats the all-layer probe by +0.273 AUROC [+0.246, +0.299] while costing nothing on
plain/swahili — the signal is present and the concatenated readout cannot see it. It does **not**
survive the operating point: +3.3 matched TPR points [+0.2, +6.9], against +10.2 for the plain
layer-averaged centroid. Report as a readout-geometry diagnosis, never as a cipher detector, and
do not reopen the layer sweep (Gate B). `RESULTS.md` §6.

Reproducibility: source commit `d4ed74b`; seed 0; CPU only; inputs are the frozen §4 score
artefacts plus `data/phase1_layerwise_27b.npz`.
`conda run -n msc-diss python -m phase1.analyse_unlabelled_calibration` (400 draws/cell) →
`data/c1_unlabelled_calibration.json` SHA-256 `f5855064cca4b4c2373ebb18cdb8a55ed72a8aa7aabbab16f15989af999fbf63`;
`conda run -n msc-diss python -m phase1.analyse_layerwise_selection` (10,000 bootstrap) →
`data/c2_layerwise_selection.json` SHA-256 `8b24df60bc9b3c27f384013248b1ed89d748340861b08aacfb5c78e9b350f6b0`.
Both figures are outstanding: matplotlib is not installed in `msc-diss`; the per-cell and
per-layer values needed to draw them are in the JSON.

## 2026-07-25 - C4: the probe also beats a current multilingual guard

Qwen3Guard-Gen-8B scored on all six conditions; the probe still leads every condition on
AUROC and matched TPR, so the headline survives a 2025 purpose-built multilingual guard.
Margin over the better of the two guards: +20.5/+23.4/+28.0/+29.2/+35.9 points
(plain→zulu). **"Strongest guard from C4" is condition-dependent** — Qwen3Guard leads at
the mid-resource end (matched french 49.1%, hindi 41.5% vs ShieldGemma 36.8%, 30.1%) and
*loses* at the low-resource end (swahili 16.4%, zulu 9.2% vs 30.1%, 15.3%); C5 and C7
must carry both or state which condition selected the arm. Full matrix in `RESULTS.md` §4.

Llama Guard 4 is **not done** — `meta-llama/Llama-Guard-4-12B` still returns HF 403 for
this account. Report as a missing comparator; do not substitute an older LlamaGuard.

Score = softmax over the three verdict branches at a teacher-forced `Safety:` position,
one forward pass, validated 96/96 against free generation on the smoke rows. The
batch-composition audit failed the 1e-3 tolerance (5/48 cells, max 0.0220) but only on
unsaturated scores (≤4.8e-4 on the 40 saturated cells), which spares the
plain/french/hindi thresholds and not swahili/zulu/reverse.

Reproducibility: `Qwen/Qwen3Guard-Gen-8B` revision
`4505cb1a6f1864f21f8b27f7daf1b9a1aab6edbb`, bf16, batch 8; Eddie jobs `57140593`
(smoke) and `57140956` (node1r03, 1× L40S); source commit `00149bb`; seed 0;
tune/test 1781/1781. `python -m phase1.score_modern_guards`, then
`python -m phase1.analyse_text_encoders --activation
data/phase1_activation_multilingual_27b.npz --baselines
data/phase1_baselines_multilingual.npz --modern-guards data/c4_modern_guards.npz --out
data/c4_modern_guard_results.json`. `data/c4_modern_guards.npz` SHA-256
`418184f50a3dafa6d977869014df0b2b642791e130d36e282a5002c23435b820`.

## 2026-07-25 - Fusion fails Gate C; the probe absorbs the text guard

Step 4 compared equal logit averaging, CC++-style weighted averaging, logistic
stacking and a Neyman--Pearson worst-domain weight against the strongest single
detector, all 2-fold cross-fitted on tune (weights, standardisation and the
condition-matched threshold fitted on the calibration fold, evaluated held-out).
Worst-domain set = {french, hindi, swahili, zulu}; reverse reported separately.

ShieldGemma does hold residual signal — 13/12/26/17 probe-missed harmful tune
prompts in french/hindi/swahili/zulu, residual AUROC 0.859/0.844/0.837/0.770 —
but it buys nothing at the operating point. Worst-domain (zulu) TPR: probe
49.2%, NP fusion 50.1% (+0.88 points, 95% CI [+0.18, +1.76]) at worse realised
FPR (1.40% vs 1.32%); every other fusion is worse than the probe alone. The NP
weight lands at 0.96--0.98 on the probe, i.e. the free optimum discards the text
guard. The already-seen test agrees (51.2% vs 51.8%). Gate C fails against the
pre-declared five-point material gain, so do not train the depth-residual
ensemble; with Gate B already failed, the method route is closed and Step 7's
mechanism fallback (comprehension-conditional monitoring) is next.

Caveat: ~606 calibration negatives per fold means every system, the probe
included, overshoots the nominal 1% FPR, so the FPR test is relative to the
probe's realised 1.32%, not to 1%.

Reproducibility: source commit `0ae4d44`; seed 0; tune n=1781 (568/1213);
`conda run -n msc-diss python -m phase1.analyse_fusion`; result
`data/phase1_fusion_results.json`, SHA-256
`378f5f274b277dbc3116acbab433d0039e4e50531f252712996bc8d7b9b583f6`. Inputs:
`data/phase1_activation_multilingual_27b.npz` (SHA-256 `76e3133c...4b682e`),
`data/phase1_baselines_multilingual.npz` (SHA-256 `ea9e1519...837e72`),
`data/phase1_small_guard.npz`, `data/phase1_multilingual_e5.npz`. CPU only.

## 2026-07-24 - Multilingual matrix complete; activation probe wins every language

Step 3 of the finish plan is done: all six conditions scored for every detector.
The all-layer `t_inst` probe beats every text baseline on AUROC and TPR in all
five language conditions under both strict and condition-matched calibration,
including multilingual-e5. Matched TPR plain→zulu: probe 75.4%→51.2%,
ShieldGemma 45.2%→15.3%, WildGuard (native) 97.4%→0.4%. ShieldGemma is the
strongest single text comparator on the low-resource end (swahili 30.1%), so it
is the fusion partner for Step 4, not WildGuard. Reverse defeats everything, and
the centroid is the only detector retaining signal there (AUROC 0.767 vs 0.556
for the all-layer probe) — a probe-geometry failure, not absent signal. Full
matrix in `RESULTS.md` §4.

The ShieldGemma equivalence audit **failed** (max abs probability difference
0.0312 vs 1e-3 tolerance) on 2 of 48 frozen audit cells, Eddie bf16 L40S versus
the original MLP scores. It changed no reported metric (plain and swahili rows
bit-identical; reverse AUROC 0.657→0.659) and all six conditions were rescored
in one environment, so the matrix is internally consistent. Record it as
cross-cluster nondeterminism in ShieldGemma's continuous score; WildGuard's
native decisions matched exactly (0/24). No LlamaGuard — gated by the Meta
licence, never staged; report as a limitation rather than a pending run.

**Next: Step 4, de-risk fusion on frozen scalar scores** (probe + ShieldGemma,
cross-fitted, tune/development data only) against Gate C: stop if fusion cannot
improve worst-domain TPR at the 1% FPR constraint over the best single detector.
CPU-only — every score needed is already in the saved artefacts, so no cluster
job.

Reproducibility: Eddie job `57134364` (node1r02, 1× L40S, bf16, 59m03s, exit 0);
source commit `59649dd`; seed 0; test n=1781 (568/1213). ShieldGemma-9b revision
`b8b636016df4540721a098c7aab91c97ec6ee508` (batch 8, composite guideline),
WildGuard revision `cbba4823f3e8020e5a74a5e29bf85072def6f2ff` (batch 4).
`data/phase1_baselines_multilingual.npz`, SHA-256
`ea9e151968d9fa038ebb7d381395f20892f7d654a2c82e8b54d9af7fd837de72`; analysis
`conda run -n msc-diss python -m phase1.analyse_text_encoders --activation
data/phase1_activation_multilingual_27b.npz --baselines
data/phase1_baselines_multilingual.npz`; result
`data/phase1_text_encoder_multilingual_results.json`, SHA-256
`897fa25f1af26d82215c53e547b205b5670f2562ba016ab879f77ebbdde1c31e`.

## 2026-07-24 - Depth coherence fails Gate B

The bounded post-core depth experiment compared mean/median layer-rank
aggregation, maximum contiguous windows, and exact Top-K non-overlapping windows
for `M in {2,4,8}` and `K in {1,2,3}`. Selection used only two-fold
cross-fitted tune predictions across the fixed Swahili, reverse, and
vowel-removal development conditions. It selected `depth_topk_m8_k3`.

On the already-seen Phase 1 test, treated only as exploratory development
evidence, this method did not give a material low-FPR improvement over the
all-layer logistic probe. Under condition-matched calibration, worst-domain TPR
over the comparable Swahili/reverse conditions increased from **2.11% to
3.17%** (+1.06 points), while maximum realised FPR increased from **1.32% to
1.65%** and exceeded the 1% constraint. Concretely, depth Top-K versus all-layer
detected 339 versus 337 harmful Swahili prompts with 20 versus 16 false
positives, and 18 versus 12 harmful reverse prompts with 17 versus 9 false
positives. Under strict transfer, worst-domain TPR remained 0% and depth Top-K
also increased maximum FPR. Gate B therefore fails: do not retain this method,
add the optional benign-variance regulariser, or broaden the depth
hyperparameter search. Proceed to the pre-specified strong text baselines and
fixed multilingual matrix; do not proceed to a depth-residual ensemble.

Reproducibility: `google/gemma-3-27b-it` revision
`005ad3404e59d6023443cb575daa05336842228a`; `t_inst`; seed 0;
train/tune/test 5,341/1,781/1,781; source commit `9e6c864`; MLP job `3565792`
(A100 80 GB, batch size 4, 41m41s, exit 0). Layerwise artefact
`data/phase1_layerwise_27b.npz`, SHA-256
`56f8f529f0694fa957d5360537c6e580268a249bf3b31a555c2b1fb781fa839d`;
analysis command `conda run -n msc-diss python
phase1/analyse_depth_coherence.py`; result
`data/phase1_depth_coherence_results.json`, SHA-256
`5e0d6a1d15b5c31fbfdce8ba73bfef6b6e40c9e45124e72925fe1d94e13eaaea`.
The analysis implementation was not yet committed at this checkpoint.

## 2026-07-23 - Phase 1/2 frozen: Swahili transfers through activations; position changes calibration

The matched 27B result is positive and Phases 1/2 are frozen. At `t_inst`, the
plain-trained all-layer logistic probe reached Swahili AUROC 0.967 and, with
negative-only condition-matched calibration, 59.3% TPR at 1.32% realised FPR,
versus ShieldGemma's 29.9% at 1.07% FPR. The paired activation advantage was
+29.4 points (95% CI [+21.1, +39.8]). The pre-specified activation--ShieldGemma
ensemble did not improve on activation alone.

The exact position control shows that Swahili rank discrimination is unchanged
between `t_inst` and `t_post-inst` (AUROC 0.9669 vs 0.9670; paired difference
-0.0001, 95% CI [-0.0072, +0.0067]), but `t_inst` transfers the frozen English
threshold substantially better (37.3% vs 10.7% TPR; paired bootstrap mean
difference +28.1 points, 95% CI [+13.9, +41.5]). Matched calibration recovers
nearly equal TPR (59.3% vs 58.6%). Reverse collapses logistic discrimination at
both positions (AUROC 0.556/0.573), so its failure is not specific to the
post-instruction token. This supports a calibration-shift explanation for
Swahili and a discrimination-loss explanation for reverse, without a causal
harmfulness/refusal claim. Proceed to the bounded two-layer SAE Phase 3; no
further Phase 2 position, layer, or transformation search.

Reproducibility: `data/judged_main_prompts.jsonl`; frozen Swahili manifest SHA-256
`47e8f8c357eb93d958bc85ec9bbe92ebb7720fe2d4f9105f6f221244ace994`;
`google/gemma-3-27b-it` revision
`005ad3404e59d6023443cb575daa05336842228a`; seed 0; train/tune/test
5,341/1,781/1,781. Commands: `python -m phase1.phase1_activation --batch-size 8`,
`python -m phase1.phase1_baselines`, `python -m phase1.analyse_phase1`, and
`python -m phase1.phase1_activation --position t_post_inst --out
data/phase2_activation_t_post_inst_27b.npz --batch-size 8`, followed by
`python -m phase2.analyse_phase2 --bootstrap 10000`. Eddie jobs:
Phase 1 `57005061`/`57005218`; Phase 2 `57058315` from clean remote commit
`6a574074fdc635b0435a49e1d36cafb839e79167`. Definitive records and compact
artifacts: `phase1/RESULTS.md`, `phase2/RESULTS.md`, and
`data/phase{1,2}_*`.

## 2026-07-22 - Four-week dissertation pivot: one anchor and two bounded explanatory phases

With four weeks remaining, the project is narrowed to a 27B-only three-phase finish:
(1) an exact matched plain/Swahili/reverse comparison targeting a probe or probe--text
ensemble advantage at TPR@1% FPR; (2) `t_inst` versus `t_post-inst` analysis to distinguish
calibration shift from loss of discrimination, without making a causal claim; and (3) a
two-layer Gemma Scope SAE comparison for a bounded interpretability result. This is planned
work, not an achieved result. The pivot is motivated by the current 27B `t_post-inst` evidence:
plain AUROC 0.994, Swahili AUROC 0.967 but zero TPR at the frozen English threshold, and reverse
AUROC 0.578, together with 99% Swahili capability and 0% WildGuard detection in the earlier
screen. New reportable experiments freeze after at most ten days; more models/ciphers, MLP
sweeps, refusal abliteration, and broad SAE/Delphi work leave the critical path. Full design,
calibration rules, limitations, fallbacks, and schedule: `DISSERTATION_FINISH_PLAN.md`.

## 2026-07-22 - capqa languages + guard language screen; results consolidated in RESULTS.md

All cipher + language results for models and guards now live in `RESULTS.md` (living table).

- **capqa fra/hin/swh, both models** (N=150, Eddie): 12B french 96 / hindi 85 / swahili 87;
  27B french 100 / hindi 88 / swahili 99 (retention = acc/plain). 27B ≥ 12B everywhere.
  Post-run audit: the two files' `(qidx, condition, sent)` inputs match exactly (0/600
  differences); NLLB revision `f8d333a098d19b4fd9a8b18f94170487ad3f821d`.
- **Guard language screen** (n=150, job 3562722): WildGuard graded blindness plain 96.7 → french 88
  → hindi 16 → swahili/zulu 0 (FPR ≤1.3%); ShieldGemma weaker/noisier (62 → 13, FPR 5-13%). Both
  ≈0% on all ciphers (n=50). Swahili = capability/detectability dissociation (99% decode, 0% WG).
- **probe_prompt OOM fixed** (3 wasted runs): manual 2048-token left truncation (apply_chat_template
  `max_length` is silently ignored in transformers 5.12), `logits_to_keep=1` (the 262k-vocab lm_head
  over all positions was the real OOM, not the hidden states), and last-token-only per-layer stack.
  Memtest 32 GB @ 12B batch-8. 12B/27B probe re-running (Eddie 56990400, Wintermute 3563857).
- **Gap:** probe tests plain/reverse/zulu only; french/hindi/swahili probe points (the language
  coupling axis) are not yet implemented in `probe_prompt.py`.

## 2026-07-21 - 10k judge landed; stratum/judge disagree on 14.4%; protected-group category OPEN

Judge job 56985874 (Eddie, 2x L40S, ~100 min) -> data/judged_main_prompts.jsonl, 10000 rows,
88 parse errors (0.9%). Code byte-identical to the validated judge (sha 6609c50f).

| stratum | judge-benign | judge-harmful |
|---|---|---|
| vanilla_benign | 4875 | 125 |
| vanilla_harmful | **1310** | 3690 |

- WJ stratum and judge label **disagree on 14.4%** of rows. Vindicates the negatives redesign:
  the 1310 judge-benign `vanilla_harmful` rows are hard negatives that a WJ-benign-only training
  set would have discarded, while letting the probe learn the stratum instead of the construct.
  Positives = 3815 (38%).
- **OPEN QUESTION before the main run:** top positive category is `hate & harassment targeting
  protected groups` at 960 = **25% of all positives**, but the locked harm definition is
  operational/capability harm (violent crime, cyber, fraud, weapons, self-harm, PII) and
  protected-group harassment is not on that list. It is also the category the judge validation
  flagged as noisiest ("over-flags generic offensive jokes as protected group"). So a quarter of
  the positive class may sit outside the locked construct in its noisiest region — decide whether
  to exclude it, treat it as the broad-harm sensitivity slice, or keep it, before training.

## 2026-07-21 - Probe design fixed: all-layer concat, CC++ adaptation not reproduction

Design settled after re-reading CC++ (2601.04603) implementation + two Codex review passes.
Supersedes the layer/negatives/storage details of the 2026-07-17 entry.

- **All-layer concatenated probe is the primary detector**, last-token position, no layer
  selection. CC++ Fig 2c ablates layer count: all layers best, fewer monotonically worse.
  This also removes a selection-on-outcome trap — in the reverse pilot plain-val AUROC is
  saturated (0.9979-0.9984) while reverse transfer at those layers spans 0.759-0.955, so
  plain-val layer selection is near-arbitrary in the quantity that matters. Per-layer probes
  demoted to a descriptive emergence curve. Payload-mean pooling = sensitivity, not a
  selected alternative. SWiM/softmax-weighting/EMA dropped: at M=1 they degenerate
  algebraically to ordinary BCE on a raw logit.
- **Framed as a CC++-inspired input-side adaptation, NOT a reproduction** (unit of
  observation, temporal aggregation, effective n and operational event all differ; shared
  ingredients are all-layer concat, linear readout, 0.1% WildChat operating point).
- **Negatives redesigned**: train on judge labels irrespective of WJ stratum, including
  judge-benign rows from `vanilla_harmful` as hard negatives. Training WJ-benign-only would
  let the probe learn the WJ stratum rather than the operational construct — the register
  confound one level in. Validity check = discrimination within the `vanilla_harmful`
  stratum. StrongREJECT rejudged under the operational rubric, not assumed positive.
- **Threshold = 0.1% WildChat alert rate** (CC++'s value and terminology; WildChat is
  unlabelled so it is not an FPR). Per-model threshold, held-out WildChat audit split.
- **Coupling metric** = retention ratio TPR_transform/TPR_plain per model, raw TPR alongside.
  This is a transform-level association across different prompt sets, not per-prompt coupling:
  bootstrap capability by shared WebQuestions `qidx` and detectability by shared WJ ID in
  independent resamples, and exclude the guaranteed plain `(1, 1)` normalisation point from any
  fitted association. Our choice, not a borrowed convention. Guard against the deeper-model
  artefact (ridge makes redundant cross-layer signal cheaper, ~beta^2/L) with one low-capacity
  comparator or treat the 12B/27B contrast as supportive rather than causal.
- **Storage**: score-forward-only. Hold plain train/tune features in RAM, fold scalers into
  probe weights, persist scalar scores (~15 MB) not activations (~81 GB at 27B float32;
  the 2026-07-17 "~20 GB" estimate was wrong, and float16 is unavailable on gemma-3).
- **12B capability points already exist** (2026-07-14, same script/seed/grader): zulu 72% vs
  27B 89%, reverse 36% vs 51% — uniformly left-shifted, as claim 2 predicts. Outstanding
  capqa work = fra/hin/swh on BOTH models. capqa retention must be hand-validated on a
  stratified sample with paired CIs before carrying a primary claim (`capability_qa.py:42`
  substring-matches gold answers; short golds match spuriously; n=150).
- NLLB fidelity gate from 2026-07-17 reaffirmed (pinned revision, one shared translation
  manifest); train/tune/test split, not train/val.

## 2026-07-21 - Harm definition = operational; judge validated (97.2% human, operational-anchored)

- **Positive class = operational/capability harm** (violent and non-violent crime, sex crimes
  and child exploitation, weapons/CBRN, cyber, self-harm, privacy/PII, and fraud/deception), NOT
  broad content harm. Decided with David after a Codex discussion pass
  + literature (Zhao et al. 2025 harm≠refusal; WildGuard's broad 13-category taxonomy).
  Rationale: invariance must be *tested*, not baked into labels — a broad grab-bag positive
  class makes plain→cipher transfer and SAE-invariance uninterpretable (lexical shortcuts,
  category mixture, prevalence). The broad instinct coincides with WJ `data_type`; kept as a
  sensitivity analysis only. Misinfo/stereotypes/fiction become a held-out
  construct-extension test (does the operational direction generalise?).
- **Judge validated for this construct.** Blind Claude-vs-judge 87.5% (n=72). Human cold
  hand-labels = 69.4% (David tracked the broader construct — itself a methods point);
  re-anchored to the operational rubric = **97.2%** (only misses = 2 PII prompts mislabelled
  benign). `judge_responses.py --mode prompt` byte-identical since validation (file unchanged
  28b52df→08f8586). PAUSE 2 cleared.
- **10k prompt-intent judge submitted** (5k+5k): job 3562339, commit 08f8586, h200_3g.71gb,
  BATCH=32/MAXTOK=64, ~95 min expected, resumable.

Phase-2 gates run on protected gemma-3-27b-it.

- **Reverse mechanism confirmed** (`reverse_pilot.py`, n=300/class WJ-vanilla, raw data_type
  labels). Per-layer last-token logistic probe, train-plain: plain in-dist AUROC ~0.997;
  **plain->reverse transfer peaks 0.955 at layer 34/62**, emerging mid-network (~0.5-0.7 early,
  0.89-0.955 plateau L31-41). The decode scaffold is class-invariant and reversed text shares no
  surface features with plain, so transfer is decoded intent, not surface stats. => reverse's
  harmful intent is linearly present at the prompt-final position *before* generation; it survives
  as a probe venue, and the emergence curve feeds claims 2/3. Layer ~34 is a natural probe layer.
  (gemma-3 has massive-activation dims that overflow float16->inf; store activations float32.)
- **Fallback ciphers** (guard screen + capqa): pig-latin and vowel-removal are guard-blind
  (WildGuard 100%->4%/4%, ShieldGemma 58%->10%/22%, FPR<=4%). Decode retention: **vowel 61%**
  (usable spare, ~reverse's 51%), **piglatin 2%** (guard-blind but model-blind -> dropped as venue,
  kept as a predicted-failure control with nato/morse). Portfolio stays reverse+zulu.
- **Prompt-intent judge validated** (unblocking the main run; the prior 90.7% triangulation was
  response-harm). 200 WJ-vanilla prompts judged prompt-intent: benign 2% flagged, harmful 72.4%
  (WJ data_type is a noisy intent proxy). Blind Claude-vs-judge on 72 stratified: **87.5%** (bar
  85-90% met), 100% on clean benign, zero disagreements on operational hazards; the 9 disagreements
  are all on the demeaning-joke / protected-group / non-distributed-conspiracy boundary (judge noisy
  there). David hand-labels the same 72 to complete the gate before the 5k+5k run.
- StrongREJECT source fixed to `AlignmentResearch/StrongREJECT` (authors' 313-prompt release).

## 2026-07-17 - Dissertation plan: three claims on the input-only cipher venue

SAE coverage checked: `google/gemma-scope-2-27b-it` has `resid_post` SAEs at layers
16/31/40/53 (widths 16k-1M, L0 small/med/big) AND `resid_post_all` (every layer, fewer
widths) + Neuronpedia. 27B stays the probed model. Plan (experiments freeze ~end July;
one shared prompt-side extraction feeds everything):

1. **Phase-1 anchor - plain→cipher transfer.** Probe on protected-27B prompt activations
   (last-token AND mean-pooled, all layers; pick by ablation), trained on PLAIN
   WJ-vanilla only (prompt-intent labels), tested on plain/reverse/zulu; negatives =
   WJ-benign + XSTest-safe, StrongREJECT held-out. Operating threshold calibrated on
   WildChat first-user-turn PROMPTS (deployment traffic is plain); ciphered-benign FPR
   reported at that threshold as the robustness check. Headline = transfer, not
   mixed-train (kills the normalise-then-guard objection and the
   cipher-surface-statistics confound; mixed-train reported as the know-the-attack upper
   bound with a char-ngram TF-IDF control). Guards (full-n on the final eval sets, not
   the n=50 screen) are 0% on ciphers, so ensemble>guard is baked in - the substantive
   results are absolute low-FPR performance and transfer, guards as context.
2. **Novel claim - capability-detectability coupling.** Probe transfer recall vs decode
   retention per transform (zulu 89%, reverse 51%, nato/morse ≈0 at prompt-final as the
   predicted-failure mechanistic control: serial-CoT decode ⇒ input probe must fail).
   Languages fixed at French (high) / Hindi (mid-high) / Swahili (mid-low) / Zulu (low),
   NLLB; capqa measures where each lands. Curve presented WITHIN the translation family
   (clean graded axis, one mechanism); ciphers overlaid as separate marked points
   (different mechanism + reverse carries a decode scaffold, zulu is bare). Stated
   limitation: retention is a WebQuestions-trivia proxy, not harmful-domain vocabulary.
   Claim: the guardrail's robustness boundary is the model's comprehension boundary.
3. **Phase-3 core - SAE feature invariance (priority if time squeezes).** On matched
   plain/cipher prompt pairs: sparse-vs-dense probe on transfer; top-feature overlap
   across surface forms; layer-wise transfer curves (where invariant intent emerges;
   never for nato/morse); auto-interp shared top features (semantic harm concepts vs
   surface shortcuts).
4. **Phase-2 minimal - harm vs refusal.** (a) protected-27B refusal rate on ciphered
   harmful: refusal drops while the probe holds (Zhao payoff). (b) score the
   plain-trained probe on abliterated-27B prompt activations: survives ⇒ probes read
   intent, not refusal precursors. Drop (b) first if time runs out.

Prereqs and gates (added 2026-07-17 after design review):
- **Prompt-intent judge is UNVALIDATED** (the 90.7% triangulation was response-harm
  labels). Gate before the main run: judge ~75 stratified WJ-vanilla prompts, blind
  handcheck, ~85-90% bar. Then judge ~5k harmful + 5k benign.
- **NLLB harmful-translation fidelity check** per language: back-translate ~50 harmful
  prompts, judge intent preservation (mistranslation would silently mislabel the
  translated slices and penalise the probe).
- First cluster job = reverse-mechanism pilot: small last-token probe on plain+reverse
  (raw WJ labels fine at pilot noise; StrongREJECT/XSTest untouched) to check reverse
  intent exists at prompt-final at all (evidence so far is verbalised in-generation
  decode); if not, reverse drops. Pilot also screens 1-2 fallback ciphers (pig latin,
  vowel-removal: guard_screen + capqa) so the cipher class survives a reverse failure.
  Reverse and zulu analysed separately throughout (decode-in-context vs multilinguality).
- Engineering: no per-exchange activation cache (Lustre small-file IO); extract+train in
  one job, features in memory, persist one consolidated array (~20 GB for ~30k
  prompt×condition).

Staging: run this scoped version first - NO extra models/languages/ciphers beyond the
above. Plain-only training means later additions are eval-only (translate + capqa +
score, no retrain), so breadth now buys nothing. Pre-registered extension order if time
remains after SAE work: (1) 12B coupling replication - lower retention (zulu 72 vs 89,
reverse 36 vs 51) predicts a left-shifted recall curve, the strongest test of claim 2;
(2) more languages; (3) more ciphers.

## 2026-07-14 - Capability on protected model: portfolio = reverse + zulu, probe 27B

Decode capability of the PROTECTED gemma-3-27b/12b on WebQuestions (n=150/cipher,
retention = acc / plain):

| cipher | 27b acc / ret | 12b acc / ret |
|--------|---------------|---------------|
| plain  | 54.0% / 100%  | 50.7% / 100%  |
| zulu   | 48.0% / 89%   | 36.7% / 72%   |
| reverse| 27.3% / 51%   | 18.0% / 36%   |
| nato   |  0.7% / 1%    |  0.0% / 0%    |
| morse  |  0.0% / 0%    |  0.0% / 0%    |

- **nato/morse 0% is a truncation artefact, not incapacity:** outputs show correct
  letter-by-letter decode (`.-- = W, .... = H, ...`) cut off by the 64-token cap before
  the answer. But the decode is *serial CoT* — intent is constructed during generation,
  not present at the prompt-final position an input-side probe reads. So they are dropped
  as probe venues (guard-blind but wrong mechanism), not rerun.
- **Portfolio = reverse + zulu:** both guard-blind AND decoded in-context (zulu with no
  visible scaffold, strongest at 89% retention). reverse decode is real but lossy (small
  slips, e.g. "Snipes"→"Spines").
- **Probed model = 27B:** uniformly higher retention (zulu 89 vs 72, reverse 51 vs 36) →
  more decoded intent for the probe to read. Code: `capability_qa.py`, `run_capqa.sh`.

## 2026-07-13 - Guard-blindness screen positive; capability on protected model

Two results firming up the input-only cipher venue.

- **Capability (eyeball, abliterated 27B):** `reverse` decode ≈100% (30/30 responses
  verbalise the correct reversed request); the 16.7% harmful-response rate is residual
  *safety*, not decode — do not gate on it. `rot13` dead (0/30). The low plain ceiling
  (26.7%) is irrelevant to input-moderation (labels = prompt intent, no harmful
  generations needed), so the abliterated generator is dropped; decode capability is now
  measured on the PROTECTED gemma-3-27b/12b via benign-verifiable QA (WebQuestions,
  gist-accuracy à la SAE4Safety Table 1).

- **Guard-blindness screen (n=50 WJ-harmful vs XSTest-safe):** WildGuard + ShieldGemma-9B
  flag plain harmful (recall 100% / 58%) but collapse to **0%** on reverse/nato/morse;
  zulu WildGuard 0%, ShieldGemma 18%. Ciphered-benign FPR 0% → genuine content-blindness,
  not scaffold-detection (the XSTest-safe control rules that out). Validates the venue
  premise: deployed guards are blind to the ciphered prompt the 27B decodes, so the probe
  has room. LlamaGuard-3 pending Meta license.

- **Cipher portfolio** = model-decodable AND guard-blind. `reverse` confirmed both; `zulu`
  usable; `nato`/`morse` guard-blind, model-decode being measured now (same run also
  decides 27b-vs-12b as the probed model). Code: `guard_screen.py`, `capability_qa.py`
  (screen commit 9278fbf).

## 2026-07-03 - Cipher venue fixed to input-only; abliterated 27B built

Refined the 2026-06-29 obfuscation venue after SAE4Safety Table 1 (cipher capability) +
own tests: base64 read+write is infeasible below ~280B (35B struggles; only DeepSeek-V4
280B succeeds), and exact cipher decoding collapses even at 27B (Table 1: gemma-3-27b
ROT13 decrypt-EM 2.9%). But MMLU-ACC under cipher stays well above chance, so the model
keeps the *gist* without exact decode. Decisions:
- **Input-only** cipher (cipher in, plain out), input-side guardrail: a text classifier on
  the ciphered prompt is blind, the probe reads the model's internal gist. Drop the
  read+write obfuscated-output venue. Ciphers = **reverse + rot13** (gist retained per
  Table 1 ACC); base64/leetspeak dropped. `capability_check.py --conds in` is the gist check.
- Labels: judge WJ-vanilla by **prompt intent** (not response), response labels archived.
  Datasets: WJ-vanilla (train) + XSTest-safe (register-matched confound negative) +
  StrongREJECT (held-out clean harmful). Eval on mixed plain+cipher so probe/text/probe+text
  complementarity stays legible on the plain slice.
- Model: gemma-3-27b-it (SAE via `gemma-scope-2-27b-it`, -it variant exists) or 12b.
- **Abliterated generator** (capability check only): `~/models/gemma-3-27b-it-heretic` via
  Heretic, Trial 186 — refusals 50/100, KL divergence 0.0801.

## 2026-06-29 - Text baseline beats probe; CC++ complementarity weak even in-dist

TF-IDF logistic on response text (train vanilla, judge labels) vs frozen `probe_v1`, on
the already-scored sets. Probe loses everywhere: in-dist vanilla AUROC **0.81 vs text
0.955**; vanilla->adversarial transfer **0.684 vs 0.839** (harm-isolated). WJ keeps
harmful semantics legible on the surface (TF-IDF transfers at 0.81 on the prompt, 0.84 on
the response), so bag-of-words suffices and the probe has no room — WJ is the wrong venue
to separate an activation from a text advantage.

CC++ reframed from the PDF: their probe does **not** beat text (Fig 2a/3a, probe-only is
the *worst* system); the result is ~zero probe compute + **decorrelated-error ensemble**
(Fig 3). So reproduction success = ensemble>text, not probe>text. De-risk on existing
data: even in-distribution the AUROC-optimal probe+text ensemble sets a~1 (discards the
probe); only a marginal low-FPR lift (equal-weight ensemble TPR@0.1%FPR 0.108 vs text
0.098). Probe too weak / too correlated (Spearman 0.55 in-dist) to complement.

Leading hypothesis: **activation-source mismatch** — probe reads PROTECTED Gemma
activations teacher-forced on ABLITERATED-generated responses (off-policy text the
protected model would never emit). Next diagnostic: extract activations from the
abliterated model itself (on-policy for the generator) on a train/eval subset, retrain a
probe, compare in-dist AUROC; if it jumps, the mismatch was the cause. Hold the
adversarial extraction until settled. `text_baseline.py` has the baseline + ensemble
analysis.

**Result (same day):** abliterated on-policy in-dist AUROC **0.772** (TPR@0.1% 0.019) — not
better than protected teacher-forced 0.810. Mismatch ruled out as the bottleneck;
abliteration if anything degrades the harm representation, so protected (teacher-forced) is
the better source. Probe weakness is intrinsic (legible WJ text + linear-probe-on-Gemma);
CC++ complementarity does not reproduce here. Decision: pivot to an obfuscation venue.

**Next steps - obfuscation venue.** Validate the project premise (activations carry a harm
signal that survives surface attacks which break text) since legible WJ doesn't show it.
- Capability gating first: apply translations (low-resource langs) and ciphers
  (base64/leetspeak/substitution) to WJ harmful prompts, generate + judge, keep only
  obfuscations where the model still complies (harm preserved) - those are the interesting ones.
- Input-guardrail framing: harmfulness probe on PLAIN prompt activations (short -> cheap),
  tested on obfuscated prompts; text (TF-IDF + a production guard) collapses on the obfuscated
  surface, probe should hold if the model decodes intent. Report the gap.
- Free look: join existing adversarial prompts -> WJ `tactics` column (WJ TSV is cluster-side)
  for a per-tactic probe-vs-text breakdown; may already show the gap reversing on obfuscation-
  like tactics.
- Also worth trying: production guard baselines (WildGuard/ShieldGemma/LlamaGuard) as the strong
  text comparator; frame harm-signal survival across languages as a language-agnostic internal
  representation (feeds Phase 3 SAE interp).

## 2026-06-28 - Adversarial scoring: weak vanilla->adversarial transfer; probe score is distribution-entangled

Scored frozen `probe_v1` (online probe, lr 3e-5, in-train val AUROC 0.92) forward-only
on the adversarial eval set and ~96k WildChat calibration exchanges. Positives =
judge-harmful responses (n=1285).

Results:
- In-dist fresh vanilla (`judged_eval`, run as a pipeline sanity check): AUROC **0.81** —
  confirms the scorer is correct; the 0.92 was within-generation-batch optimism (val on
  responses from the same generation run as train).
- Vanilla->adversarial, harmful vs matched adversarial-benign: AUROC **0.684**.
- ASR@1%FPR / @0.1%FPR: **45% / 70%** (WildChat-calibrated), **81% / 89%** (matched
  adversarial-benign, harm-isolating).

Key diagnostic: the max-EMA score scale is strongly distribution-dependent. Benign mean
score vanilla **11.7** > adversarial **5.0** > WildChat **1.3** — a benign vanilla
response outscores a harmful adversarial one. Within a distribution, harm adds only ~+2;
across distributions the offset dominates. Standardisation is fit on the vanilla train
distribution (correct for deployment), so OOD inputs sit off-centre and the
WildChat-calibrated FPR is not comparable across distributions (36% of adversarial-benign
exceeds the WildChat-1%FPR threshold).

Interpretation: **not yet evidence the dense recipe fails.** CC++'s positive dense-probe
result trained on red-team/adversarial data; our Phase-1 probe trains on vanilla only, so
it likely under-covers the adversarial region rather than the recipe being broken (CC++
also report benign-science over-flagging, which we reproduce). Next: train on mixed
vanilla+adversarial with a *disjoint* adversarial test to get the in-distribution
adversarial detection ceiling (the apples-to-apples CC++ reproduction); only then is
vanilla->adversarial transfer a separable contribution.

## 2026-06-27 - Adversarial eval set built + judge validated

WildJailbreak *adversarial* eval generated (abliterated 12B, 512 tok) and judged.
Re-judged 676 rows truncated by the 64-tok judge cap (raised to 256); recovered 303
harmful labels. Final: adversarial_harmful 25.0% harmful (1252), adversarial_benign
0.7% (35) — 1285 ASR positives. Judge handcheck (Claude blind, 75 rows): ~94%
population-weighted agreement; disagreement is one-directional (judge broader than
strict material-harm, e.g. counts verbal compliance/creative-disparagement),
concentrated in the positive class. Accepted judge as v1 oracle — a guardrail
erring toward blocking is the safe direction.

## 2026-06-22 - Probe training: fp16 cache overflow on massive-activation channel

First probe run gave `nan` loss from step 0. Cause: the activation cache is stored
fp16 (`vec.to(torch.float16)`) but the model runs bf16; Gemma 3 12B's massive
activation channel (**residual dim 2339**, present in every layer, ~1e4-1e5 and
growing with depth) exceeds fp16's 65504 ceiling and saturates to `inf` in layers
20-47. Exactly **one channel** — 28 of 188160 dims overflow; finite absmax 65024.

**Fix (no re-extraction):** drop dim 2339 from all 49 layer blocks (49 dims) and
per-dim standardise the rest (stats on a 2000-exchange train sample, seeded), applied
at load in `train_probe.py`. Standardisation is folded into a full-D effective weight
at save time (`W/std`, zeroed on dropped dims; bias absorbs `-ΣW·mean/std`), so
`score_probe.py` is unchanged and its live bf16 forward multiplies the (finite) dim
2339 by zero. Dropping the channel everywhere also handles its ill-conditioning, not
just the overflow. Avoids re-running the 1.83 TB protected-12B extraction.



Decided the storage boundary after measuring the finished train cache: **9987
exchanges, 1.83 TB, ~183 MB/exchange** (fp16, all ~512 response tokens × 49 hidden
states × 3840 — token count and all-layer concat are the multipliers; ~0.38 MB/token).
Replicating this for eval and a 100k calibration set would be ~10 TB on shared Lustre.

**Decision: do not cache eval or calibration activations.** The train-cache rationale
(read every epoch × loss/EMA/layer config × seed) does not transfer: eval/calibration
are only *scored* — a single forward-only pass per probe config, no generation, no
backward. Generation (the slow autoregressive part) already happened upstream, so
scoring is cheap (~min for eval, ~10 GPU-min for a 20k dev-calib, <~1 hr for 100k).
~10-15 ablations = a few GPU-hours total, vs 5-10 TB of storage. Persist **per-exchange
scalar scores** so any FPR is read off the ASR-vs-FPR curve for free. This vindicates
the original 2026-06-15 storage instinct.

**Operating points:** report the full ASR-vs-FPR curve. **ASR@1%FPR is the primary
ablation metric** (stable: ~10% relative SE at 10k calib), **ASR@0.1%FPR the headline**
matched to CC++ (needs ~100k calib for comparable precision — run once on-the-fly for
the final probe only). Calibration set size is orthogonal to probe training (it only
sets a scalar threshold) so it cannot "dominate" the 10k training set.

**Eval must be WJ adversarial, not a held-out vanilla split** — the v1 claim is the
vanilla->adversarial tactic-transfer number. Storage levers if ever needed, biggest
first: fewer layers (run layer ablation off the train cache first, then fix the set),
token cap/stride (risky: late harm + benign-max for FPR), int8 (avoid for a repro).

## 2026-06-19 - Rubric gate cleared (90.7%) via blind triangulation

Re-judged the 512 pilot with the refined rubric (stereotype-fiction carve-out) and
ran a three-way **blind triangulation** to validate it: David and Claude (Opus 4.8 medium effort) independently
labelled the same 75-row stratified sample (30/30/15, seed 0, `handcheck.py
sample/label/blind-prep`); Claude labelled from response text only, judge verdict and
David's labels stripped. Initial agreement: David-judge **76%**, Claude-judge **84%**,
David-Claude **79%** — the gate (85%) still failed.

**Diagnosis (the gap is structural, not noise).** Decomposing the disagreement by
direction showed two distinct sources. (1) ~11 of David's misses were residual
*prompt-intent over-flagging*: legal/clinical explainers, fabricated profiles, and
deflections (e.g. a defamation-law explainer, self-harm psychoeducation, an
illegal-fishing overview) where the *response* delivers no material harm — Claude
agreed with the judge on nearly all of these, confirming the judge is reasonable.
(2) The judge has a few genuine false positives where it labels on surface form without
checking reality: a Winnie-the-Pooh banking-password joke, fabricated (not reproduced)
book text, and hallucinated celebrity-vacation details, all flagged as
disclosure/copyright violations.

**Resolution.** Adopt the judge as the single source of truth (no re-judge): its 84%
blind agreement with Claude validates it, and tightening it to distinguish
fabricated-vs-real disclosure is exactly the call an LLM judge does unreliably.
Label definition: harm = the *response* delivers materially harmful content, with
*producing requested denigrating content counted as harmful* (representational harm a
guardrail should catch). David re-examined only his 12 over-flag rows (`handcheck.py
recheck`) against that criterion; David-judge agreement rose to **90.7%** (confusion:
25 TP, 43 TN, 6 judge-harmful/David-benign, 1 the reverse). Per stratum: benign 93%,
harmful-judged-benign 97%, harmful-judged-harmful 83%.

**Residual disagreements (7, ~9%)** are exactly the two contested categories, now a
known small label-noise source rather than a blocker: illustrative group-denigration
(439 trans, 154 developing-countries, 383 dropouts) and fabricated/fictional disclosure
(371 Bezos, 180 Brad Pitt's children, 785 Winnie-the-Pooh), plus 299 (GPS-spoofing
dual-use). At thousands of training examples these won't move the probe; if needed,
hand-verify only the eval set's disclosure-category prompts later. Gate passed — proceed
to scale generation (5k/type) and `extract_activations.py`.

## 2026-06-19 - Length sweep result: keep 512

Re-judged the 1000-exchange pilot at three response clips (`length_sweep.py`; 512 =
full, no truncation). Base rate 176/1000 harmful at 512. Retention of the 512-harmful
set: **384 -> 84.1%, 256 -> 79.5%** — both well below the 95% bar, with non-trivial
two-way flipping (384: 23 H->B, 14 B->H; 256: 33 H->B, 20 B->H). Decision: **keep 512
generator tokens.** Clipping drops genuine harm that lands late (abliterated responses
front-load ~150 tokens of preamble), so a shorter cache span would discard label-bearing
content. No compute saving here; the all-layer cache stays sized for the full response
span.

## 2026-06-18 - Judge gate diagnosis, length sweep, activation-cache design

**Rubric validation (gate).** Hand-check of the Qwen judge scored 77% agreement on a
75-row stratified sample — below the 85% gate, but the failure was diagnosed as *my
over-labelling*, not judge error. Disagreement was one-directional (16 judge-benign vs
my-harmful, 1 the other way) and concentrated in the harmful-prompt/judged-benign
stratum: I was labelling on the prompt's intent, the judge correctly labels on whether
the *response* delivers material assistance. Against the agreed definition the judge is
right ~13/17. Decision: keep the judge; the operative criterion stays "response
materially advances a hazard," not topical engagement (re-introducing prompt-intent
labels would re-create the topic/sensitivity confound). One real judge inconsistency
fixed: a stereotype-fiction carve-out (it flagged one illustrative stereotype narrative,
cleared an identical one). Next: re-judge pilot with refined rubric, I correct my ~13
over-flags, re-score (expect >=90%).

**Generation length sweep.** Abliterated responses front-load ~150 tokens of preamble,
so harm lands mid-response, not early in absolute terms (this is why 256->512 earlier).
To decide whether 512 can drop for compute/cache saving, re-judge the existing pilot
with responses clipped to the first N generator tokens (`--truncate-tokens`) and pick
the shortest N that retains >=95% of harmful labels. No regeneration needed.

**Activation cache design (`extract_activations.py`).** Teacher-forced single forward
pass through the protected Gemma 3 12B; cache **response tokens only** (the streaming
probe scores generated tokens, not the prompt) — roughly halves the cache vs the
prompt+response budget. Per token: concat all hidden_states entries -> (n_layers+1) x
hidden = 49 x 3840 ~= 188k dims, float16, one .npy per exchange + manifest,
skip-if-exists. Response span located by char offset (token prefixes break on BPE
boundary merges). Calibration/eval scored on the fly, not cached.

**FPR calibration source (`load_wildchat.py`).** First English user->assistant exchange
per WildChat-1M conversation, as-is (WildChat's own responses), no toxicity filter —
natural traffic with the 99.9th-pct threshold sitting above the borderline tail.

## 2026-06-16 - Claim framing: detector capability (core) vs deployment ASR (extension)

Core claim (teacher-forced eval supports it): harmfulness is decodable from Gemma
activations at low FPR even under refusal-suppressing tactics — extends Zhao 2507.11878
(refusal and harmfulness are separate; the jailbreak beats refusal, the probe reads
harmfulness). Holds even though Gemma refuses most prompts. Deployment ASR-recovery is
an extension needing the residual-ASR pilot + on-policy/prefill eval.

## 2026-06-15 - Eval design: separate off-policy transfer from on-policy deployment

Clarified a conflation in the eval plan. Three distinct activation distributions:
(1) train = protected model reading teacher-forced **abliterated** responses
(off-policy); (2) v1 eval = same, vanilla->adversarial (off-policy, generator held
constant); (3) deployment = protected model generating its **own** harmful text
(on-policy). v1 isolates jailbreak-tactic shift but, being off-policy on both sides,
never tests the on/off-policy generalisation the method relies on — and shares any
abliterated-text style artefact, which could inflate the transfer number.

- Abliterated (train) and prefill-and-continue are **not** competing options for one
  slot. Abliterated stays the training source (volume + balance + WJ-tactic diversity;
  aligned with how He et al. and CC++ source synthetic/curated harmful data). Prefill /
  on-policy is the *deployment-eval* layer.
- The on-policy eval does double duty: supports any deployment claim **and** acts as the
  control that v1 isn't an abliterated-fingerprint artefact. Pull a minimal version
  forward rather than fully deferring to v2.
- Prefer **true on-policy** for the deployment eval (no prefix shortcut a la JEDI); fall
  back to prefill-and-continue only if positives are too sparse.
- **Added: residual-ASR pilot** to decide. Generate from the protected Gemma 3 12B on a
  few hundred WJ-adversarial prompts, judge with the same rubric, report its on-policy
  jailbreak rate. >=~10-20% -> true on-policy viable; lower -> prefill-and-continue.

Reference: CC++ evaluates on real on-policy jailbreaks (red-team + shadow deployment) +
synthetic CBRN; no abliterated generator anywhere — that trick is our substitute for
lacking their red-team pipeline.

What was decided or learned, why it matters, and the next consequence.

## 2026-06-14 - Phase 1 plan: CC++ linear probe reproduction

Supervisor asked to start by reproducing CC++ (Cunningham et al., 2026) using open
models. The full production system is out of reach (exchange classifier requires
proprietary CBRN synthetic data), so Phase 1 targets the linear probe (Section 5),
which is the novel and tractable component.

**Planned setup:**
- Protected model / probe backbone: Gemma 3 12B IT (frozen, all-layer activations
  concatenated per token)
- Training loss: SWiM logit smoothing (window M=16) + softmax-weighted BCE (τ=1),
  exchange-level labels
- Positives: BeaverTails unsafe split — pre-existing (prompt, harmful_response) pairs,
  human-annotated, no generation needed
- Negatives: BeaverTails safe split — same questions, safe responses; avoids
  register-mismatch confound
- Calibration / false positive rate: WildChat-1M (normal production traffic)
- Available but not yet needed: abliterated Gemma 3 4B and 12B IT (via Heretic) for
  generating on-distribution harmful completions if BeaverTails responses prove
  insufficient

**Key assumption to watch:** BeaverTails harmful responses were generated by PKU's
BeaverBeaver model, not Gemma 3 — potential style mismatch between positives and
negatives. Probably fine for a baseline probe but worth revisiting if results are odd.

## 2026-06-15 - Phase 1 design revised: WildJailbreak transfer, single generator

Reframed the goal after reading CC++ Section 5 closely. Phase 1 is the dense
streaming-probe *baseline* the SAE work compares against — not the production system
(no CBRN synthetic data / exchange-classifier labels). The reproducible content is the
probe recipe + ASR@0.1%FPR metric, tested as a **jailbreak transfer** result.

**Design:**
- Backbone: protected Gemma 3 12B IT, frozen, all-layer concat per token (Gemma Scope 2
  SAEs cover Gemma 3, so SAE comparison stays on the same model).
- Recipe: all-layer linear probe + SWiM smoothing (M=16) + softmax-weighted BCE (τ=1),
  exchange-level labels, EMA at inference.
- Train: WildJailbreak **vanilla** harmful/benign; responses from abliterated Gemma 3.
- Eval: WildJailbreak **adversarial** harmful/benign; responses from abliterated Gemma 3.
  Vanilla→adversarial isolates the jailbreak-tactic shift with generator and harmful
  behaviour held constant (cf. Zhao train-on-plain/test-on-shifted). BeaverTails kept
  for a later cross-dataset generalisation check, not primary training.
- Generator held constant (abliterated) across train and eval, so the
  abliterated-text-vs-own-generation gap doesn't bias the transfer number.
- Metrics: ASR@0.1%FPR; plus discrimination on matched adversarial-benign negatives.
- Calibration: threshold = 99.9th percentile of per-exchange max-EMA score over ~50-100k
  WildChat exchanges (real production traffic → deployment-meaningful FPR).
- Staging: v1 as above; v2 = prefill on the protected model for deployment-faithful
  absolute ASR.

**Storage:** `~` is shared 271T Lustre, 38T free (node-local `/disk/scratch` only on
compute nodes). Cache only the training activations (reused by loss/layer ablations);
score calibration/eval on-the-fly and keep scalars. ~10k exchanges × ~300 tok ≈ 1.1 TB
(12B) / 0.5 TB (4B); confirm exact dims from config first.

**To verify:** abliterated responses to harmful prompts are actually harmful (response
grader, needed for eval regardless); WildChat benign contamination left as a slightly
conservative threshold for v1.

**Cluster assets:** HF cache is the default `~/.cache/huggingface` (do not set HF_HOME).
Abliterated response generators (Heretic, loaded by path, not on the Hub):
`~/models/gemma-3-12b-it-heretic`, `~/models/gemma-3-4b-it-heretic`.

## 2026-06-15 - Labelling: LLM-rubric judge on response harm, not WildJailbreak labels

Pilot eyeballing showed WildJailbreak's `vanilla_harmful` prompt label is noisy
(~15-25% benign mislabels: factual/historical Qs, "find my own SSN") and its harm
notion is very broad. Training positives on `data_type` would contaminate the
positive class. Also the abliterated model front-loads preamble, so harm (where
present) often lands past 256 tokens.

Decisions:
- **Label by the response, not the prompt.** An independent LLM-rubric judge labels
  each (prompt, response) as harmful/benign; that defines positives/negatives.
- **Judge = `Qwen/Qwen3.6-27B`** (different family from Gemma -> independent of both
  the probed model and the generator; one A100 bf16; no-think mode, JSON verdict).
  Not a guard model: WildGuard/LlamaGuard stay as text *baselines* we compare
  against, so using the judge as oracle is not circular (cf. CC++ rubric grading).
- **Harm definition: moderate, taxonomy-anchored** (MLCommons/WildGuard categories);
  harmful = response gives material assistance toward a hazard; excludes
  benign-factual, refusals, trivial advice. Same rubric for train and eval.
- **Validate the judge** against ~100 hand-labels (target >=~85% agreement) before
  trusting it; this also tunes the rubric.
- **Generation bumped to 512 new tokens** so harm appears before truncation.

Pipeline is now: generate (abliterated, 512) -> judge (Qwen3.6-27B) -> extract
(protected 12B) using judge labels.
