# Confirmatory Finish Plan

Decision date: 2026-07-25. Supersedes the post-core plan of 2026-07-23, whose
gated method route is closed: Gate A produced only a weak definitive-label
layer curve, Gate B (depth-coherent aggregation) failed, and Gate C (fusion
with the strongest text guard) failed. Steps 5 onwards of that plan are void
and are not to be revived.

This document is the **confirmatory track**: bounded work with high probability
of landing, each item implementable without further scientific judgement.
Speculative work lives in `DISSERTATION_EXPLORATORY_IDEAS.md` and must never
displace anything here.

## Execution state

**Start here.** Take the next item whose status is `not started` and whose
dependencies are all `done`. Do **only that item**. Finish by updating this
table, adding a short `RESEARCH_LOG.md` entry, and committing.

One session owns one item (C1 and C2 may share a session — both are CPU-only).
Do not start a second item in the same session. Do not run exploratory `X`
items unless the confirmatory track is blocked and their dependencies are met.

| ID | Item | Status | Depends on | Jobs |
|---|---|---|---|---:|
| C1 | Unlabelled threshold transport | done (acceptance met), figure outstanding | — | 0 |
| C2 | Layer aggregation diagnosis | done, figure outstanding | — | 0 |
| C3 | Large negative pool / 0.1% FPR | not started | C4 (shares scoring job) | 1 |
| C4 | Qwen3Guard + Llama Guard 4 | Qwen3Guard done; Llama Guard 4 blocked on the Meta licence (HF 403) | — | 1 of 2 |
| C5 | Normalise-then-guard | not started | C4 | 2--3 |
| C6 | Comprehension-conditional monitoring | not started | — | 1--2 |
| C7 | External-source confirmation | not started | C4 | 3--4 |

### When to touch the exploratory track

Exploratory items live in `DISSERTATION_EXPLORATORY_IDEAS.md`, which has its own
state table. A session may start one **only** when David has named it in the
session prompt. Do not infer that an exploratory item is due because the
confirmatory queue looks idle — starting a speculative branch is a research
priority decision, not a scheduling one.

Two exceptions, which need no instruction:

- **Standing riders below.** These are mandatory parts of confirmatory jobs.
- **X7's output-side kill test.** One short job, no dependencies, exists to
  close a question cheaply. Run it whenever a queue slot is free.

### Standing riders

Any job that runs a forward pass over prompts **must** additionally save the
following. Each costs almost nothing at extraction time and costs a full
GPU job to recover later.

1. **Class-conditional means per layer, per condition** — the harmful mean, the
   benign mean, and the diagonal of each covariance. This is 62x5376 floats per
   class per condition against 81 GB of raw activations. It unblocks X4
   entirely and gives the difference-in-means readout of X5 for free on any job
   that touches the train split.
2. **Payload-mean pooling alongside `t_inst`** — the same forward pass, pooled
   over the payload tokens instead of read at the last instruction token. Store
   both score sets. This unblocks the second half of X5.

Neither rider changes any confirmatory result: the primary detector remains the
all-layer `t_inst` logistic probe. Riders are stored, not analysed, unless the
corresponding X item is separately authorised.

Pilot scripts already committed and reproducing:
`phase1/analyse_unlabelled_calibration.py` (C1) and
`phase1/analyse_layerwise_selection.py` (C2). Extend these rather than writing
new ones. For the strict/oracle reference rows use the repo's existing
`phase1.analyse_phase1.threshold_at_one_percent` so numbers match the frozen
tables — the pilots use a plain quantile and differ in the last decimal.

## Status of the dissertation

The core is complete and sufficient to submit. Frozen records:
`phase1/RESULTS.md`, `phase2/RESULTS.md`, `phase3/RESULTS.md`, matrix in
`RESULTS.md` §4.

Established findings:

1. The all-layer `t_inst` activation probe beats every evaluated text baseline
   on AUROC *and* TPR@1%FPR in all five language conditions under both strict
   and condition-matched calibration. Matched plain→zulu: probe 75.4%→51.2%,
   ShieldGemma 45.2%→15.3%, WildGuard (native) 97.4%→0.4%.
2. Swahili preserves harmfulness ranking but shifts calibration; `t_inst`
   transfers the frozen English threshold far better than `t_post-inst`
   (37.3% vs 10.7% TPR) at effectively identical AUROC.
3. Block-31 Gemma Scope SAE probes retain 95.8% of dense above-chance Swahili
   AUROC with a moderately stable top-feature core, but lose substantial
   low-FPR performance.
4. Two honest negatives: depth-coherent aggregation gives no material low-FPR
   gain (Gate B), and probe+ShieldGemma fusion cannot beat the probe alone at
   the operating point — the free Neyman--Pearson weight lands at 0.96--0.98 on
   the probe, discarding the text guard (Gate C).

## Common evaluation backbone

Every experiment below inherits this unless it explicitly overrides it. Do not
vary these to improve a number.

- Labels: prompt-level operational harm from `data/judged_main_prompts.jsonl`.
  Positives exclude `hate & harassment targeting protected groups` and the
  off-taxonomy `harassment` category. Benign rows retained regardless of
  WildJailbreak stratum.
- Split: seed 0, train/tune/test = 5,341 / 1,781 / 1,781. Test = 568 harmful /
  1,213 benign. Never refit anything on test.
- Position: `t_inst`, final token of the user instruction.
- Primary detector: all-layer logistic at `t_inst`.
- Latent baseline: Zhao-style layer-averaged cosine centroid.
- Primary metric: TPR at a threshold calibrated to 1% FPR on tune negatives.
  Secondary: AUROC. Report realised test FPR beside every TPR.
- Two calibrations, always both: **strict** (threshold from *plain* tune
  negatives) and **condition-matched** (threshold from same-condition tune
  negatives, using no shifted harmful examples).
- Uncertainty: 10,000-repeat paired bootstrap over held-out prompts, including
  threshold re-estimation.
- Conditions: plain, french, hindi, swahili, zulu, reverse. Report all six
  every time. Vowel-removal appears only where the artefact already has it.

Pinned revisions: `google/gemma-3-27b-it`
`005ad3404e59d6023443cb575daa05336842228a`; ShieldGemma-9b
`b8b636016df4540721a098c7aab91c97ec6ee508`; WildGuard
`cbba4823f3e8020e5a74a5e29bf85072def6f2ff`; NLLB
`f8d333a098d19b4fd9a8b18f94170487ad3f821d`.

Existing artefacts to reuse rather than regenerate:

| File | Contents |
|---|---|
| `data/phase1_activation_multilingual_27b.npz` | probe + centroid scores, all six conditions, tune and test |
| `data/phase1_baselines_multilingual.npz` | ShieldGemma, WildGuard, TF--IDF scores, all six conditions |
| `data/phase1_layerwise_27b.npz` | per-layer logistic + centroid scores, 62 layers × {plain, swahili, reverse, vowel} |
| `data/phase1_small_guard.npz`, `data/phase1_multilingual_e5.npz` | DeBERTa-v3-small guard, multilingual-e5-base |
| `data/phase3_scores.npz` | dense vs SAE scores, blocks 31 and 40 |
| `data/phase1_translations/*.jsonl` | frozen NLLB translation manifests |

Raw activations were never persisted (score-forward-only storage). Anything
needing raw vectors requires a new extraction job.

## Reporting rules

- Condition-matched results are **adaptation** results. Never describe them as
  zero-shot or calibration-invariant.
- Report every preselected condition. Do not retain only favourable ones.
- Anything developed after seeing the Phase 1 test is exploratory on that test
  until confirmed on the external set of C7.
- Say "best among evaluated open guards under this protocol" unless every
  system received comparable inputs, labels and calibration data.

---

# C1. Unlabelled threshold transport

**Cost:** CPU only, no cluster job, ~half a day. **Confidence: high — already
verified on the saved scores while planning this document.**

## Claim

The condition-matched operating point does not require labelled shifted
negatives. A threshold estimated from a few hundred *unlabelled* same-condition
prompts, contaminated at a realistic harmful base rate, recovers most of the
oracle-calibrated TPR; and the activation probe gains far more from this than
the strongest text guard does.

## Why it matters

This converts the dissertation's weakest framing ("the headline number needs
condition-matched calibration, which is an adaptation result") into a
deployment claim. Shifted *benign traffic* is something a deployment actually
has; shifted *labelled harmful examples* are not.

## Method

For each condition in {french, hindi, swahili, zulu, reverse} and each detector
in {all-layer probe, ShieldGemma, multilingual-e5, DeBERTa guard, TF--IDF}:

1. Draw `k` prompts at random from the **tune** split of that condition, with
   `npos ~ Binomial(k, pi)` drawn from harmful tune rows and the rest from
   benign tune rows. Labels are *not* used to filter — this simulates
   unlabelled traffic at prevalence `pi`.
2. Set the threshold at the 99th percentile of those `k` scores.
3. Apply to the frozen test split; record TPR and realised FPR.
4. Repeat 400 times; report mean and 2.5/97.5 percentiles.

Fixed grid — do not extend it: `k ∈ {100, 300, 1000, 3000}`,
`pi ∈ {0.00, 0.01, 0.02, 0.05, 0.10}`. Seed 0.

Report against two reference rows per cell: **strict** (frozen plain-English
threshold) and **oracle** (all 1,213 labelled tune negatives).

## Expected result

Already measured during planning (`swahili`, `k=300`, `pi=0.01`): probe
55.4% TPR / 1.06% FPR, against oracle 59.3% / 1.32% and strict 37.3% / 0.25%.
ShieldGemma under the identical procedure reaches 30.0%. Zulu behaves the same
way (probe 47.5% vs oracle 51.2%). Recovery degrades sharply at `pi = 0.05`
(probe 26.1%), which is a genuine and reportable boundary, not a failure.

## Acceptance

Report the full grid regardless of outcome. The headline sentence is only
warranted if at `k = 300, pi = 0.01` the probe recovers ≥85% of its oracle TPR
in at least three of the four language conditions.

## Output

`data/c1_unlabelled_calibration.json` plus a TPR-versus-`k` figure faceted by
`pi`. Record in `RESULTS.md` as a new section.

## Outcome (2026-07-25)

Acceptance met in all four language conditions (recovery 86/88/93/93% at
`k=300, pi=0.01`). `RESULTS.md` §5. The figure is not drawn — matplotlib is not
installed in `msc-diss`, and the JSON holds every cell.

---

# C2. Layer aggregation destroys shift-robust signal

**Cost:** CPU only, no cluster job, ~half a day. **Confidence: high — verified
on `data/phase1_layerwise_27b.npz` while planning.**

## Claim

The reverse-condition failure is a readout-geometry failure, not absence of
signal. A single layer selected using **plain tune data only** retains
substantially more rank-level harmfulness information under representational
shift than the all-layer concatenated logistic probe, which is the CC++-style
default.

## Why it matters

It is the mechanistic explanation the dissertation currently lacks for its
biggest negative, and it is a methods warning of independent interest:
concatenating every layer maximises in-distribution fit while destroying
out-of-distribution transfer.

## Method

All scores already exist. For each of `{logistic, centroid}` and each layer
`l ∈ [0, 61]`:

1. Compute test AUROC per condition {plain, swahili, reverse, vowel}.
2. Define the selection rule **once**: `l* = argmax_l AUROC_l` on **plain tune**
   scores. No transformed data, no test data, enters selection.
3. Report the selected detector across all four conditions, beside the
   all-layer logistic and the layer-averaged centroid.
4. Also report, as a clearly-labelled upper bound only, the argmax on each
   condition's own tune scores — to show how much the plain-only rule leaves on
   the table.
5. Report TPR/FPR at 1% under both calibrations alongside AUROC.

## Expected result

Measured during planning:

| detector (plain-tune selection) | plain | swahili | reverse | vowel |
|---|---|---|---|---|
| all-layer logistic | 0.988 | 0.967 | 0.556 | — |
| layer-avg centroid | 0.939 | 0.895 | 0.767 | — |
| L53 centroid | 0.971 | 0.925 | **0.829** | **0.879** |
| L33 logistic | 0.989 | 0.973 | 0.668 | 0.832 |

## Mandatory honesty constraint

The gain is rank-level and **does not survive at the operating point**: reverse
matched TPR is 5.5% for the L53 centroid against 13.6% for the layer-averaged
centroid and 2.1% for the all-layer probe. Report this in the same paragraph as
the AUROC table. Do not present a plain-tune-selected layer as a cipher
detector, and do not turn this into a layer or aggregation sweep — Gate B
already closed that route.

## Output

`data/c2_layerwise_selection.json`, a per-layer AUROC figure with one line per
condition, and a `RESULTS.md` subsection.

## Outcome (2026-07-25)

Confirmed with 10,000-repeat paired intervals: L54 centroid +0.273 reverse AUROC
[+0.246, +0.299] over the all-layer probe, +3.3 matched TPR points [+0.2, +6.9],
against +10.2 for the plain layer-averaged centroid. `RESULTS.md` §6. Per-layer
curves are in the JSON; the figure is not drawn (no matplotlib in `msc-diss`).

---

# C3. Repair the low-FPR tail with a large negative pool

**Cost:** 1 cluster job (27B score-forward over prompts, short sequences),
~1 day. **Confidence: high.**

## Claim

The reported operating points are estimated from roughly twelve tune negatives
(1% of 1,213), which is why every detector overshoots the nominal 1% FPR. With
a large realistic negative pool the same comparisons hold, and a 0.1% FPR
operating point — the point CC++ actually uses — becomes reportable.

## Why it matters

It is the single largest precision weakness in the current matrix and an
obvious examiner question. It also makes the headline comparable to CC++ and
Segment-Level Coherence, which report at 0.1%.

## Method

1. Take WildChat first-user-turn prompts. **Do not reuse
   `data/wildchat_scores.jsonl`** — those are old response-level EMA scores
   from `probe_v1`, not prompt scores from the current probe.
2. Sample 50,000 prompts, seed 0, deduplicated, English-only filter off. Freeze
   the ID manifest and record its SHA-256 before scoring.
3. Score with the frozen all-layer probe and centroid (`t_inst`), and with
   ShieldGemma and Qwen3Guard from C4. One 27B forward pass, score-forward only.
4. ~~Also produce a Swahili-translated copy of the same 50,000 prompts using the
   pinned NLLB revision, to give a shifted background of matching size.~~
   **Cut 2026-07-25.** The frozen pool is only 47% English-labelled (24% Chinese,
   12% Russian) and the repo's NLLB path translates with `src_lang="eng_Latn"`,
   so a whole-pool copy would mis-specify the source language for most of it and
   confound language shift with source mis-specification and 256-token
   truncation. The plain pool is the C3 deliverable. An English-labelled-subset
   Swahili arm (~23.5k rows) remains available in
   `phase1/translate_wildchat_pool.py` as a clearly-labelled sensitivity
   analysis, to be run only after the plain result and the chapters are secure.
5. Recompute the C1/C2 and headline tables with thresholds set on this pool at
   1% and 0.1%.

WildChat is unlabelled. Call the resulting quantity a **background alert rate**,
never an FPR, exactly as the existing protocol does.

## Acceptance

Report at both operating points. The main-text headline stays at 1% FPR on
labelled tune negatives; the WildChat pool is the precision and deployment
supplement.

## Output

`data/c3_wildchat_prompt_scores.npz`, manifest JSON with SHA-256, and updated
tables.

---

# C4. Current-generation guard baselines

**Cost:** 2 cluster jobs (1× L40S each), ~2 days. **Confidence: high that it
runs; the outcome is genuinely uncertain and that is the point.**

## Claim

The activation advantage under language shift holds against **current** open
guards, not only against 2024-era ShieldGemma and WildGuard.

## Why it matters

ShieldGemma-9b and WildGuard are old. `papers/notes/52_trojan_speak...md:16`
already benchmarks Qwen3Guard and Llama Guard 4, so their absence is a fair
examiner objection. Qwen3Guard is purpose-built multilingual, which is exactly
the axis the dissertation claims text guards fail on — beating it is a
substantially stronger result than beating ShieldGemma, and losing to it is
something to discover now rather than in the viva.

## Models — fixed, no substitutions

- `Qwen/Qwen3Guard-Gen-8B` (Apache-2.0, created 2025-09-23). Primary new
  comparator. Size-comparable to ShieldGemma-9b.
- `meta-llama/Llama-Guard-4-12B` (licence now available). Secondary.

Pin and record the exact revision SHA of each before scoring. Do **not**
substitute `Llama-Guard-3-8B`: running the older model at its native binary
threshold does not answer the "current comparator" objection.
`Llama-Prompt-Guard-2-86M` is a prompt-injection detector, a different
construct — do not use it as a harm comparator.

## Method

Follow the pattern in `phase1/extend_multilingual_guards.py`.

1. Score all six conditions, tune and test splits, all 1,781 + 1,781 prompts.
2. Extract a **continuous** score, not just the generated verdict: use the
   probability of the unsafe label token at the decision position, as already
   done for ShieldGemma. If a model exposes a severity level, record it too but
   keep the label-token probability as the primary score.
3. Where a model only supports a native binary decision, report it separately
   in the WildGuard-style native-decision table — a binary verdict cannot
   support the matched 1%-FPR comparison.
4. Run the same equivalence audit used for the multilingual matrix: rescore a
   frozen 48-cell subset and record max absolute score difference.
5. Analyse with `phase1/analyse_text_encoders.py`, unchanged.

Offline preflight is mandatory before the batch submit: run a `--limit 8` smoke
with `HF_HUB_OFFLINE=1` after staging weights, since compute nodes have no
internet.

## Acceptance

Report the full six-condition matrix for both guards whatever the outcome. If
either guard matches or beats the probe in a language condition, the headline
must be restated as an advantage over the specific guards evaluated, and the
affected chapter rewritten before submission.

## Output

`data/c4_modern_guards.npz`, updated `RESULTS.md` §4, and a `RESEARCH_LOG.md`
entry recording revisions and the audit result.

## Outcome (2026-07-25)

Qwen3Guard done; the probe still leads in all five language conditions, so no
chapter rewrite. **"Strongest guard from C4" is condition-dependent**: Qwen3Guard
leads on plain/french/hindi, ShieldGemma on swahili/zulu. C5 and C7 must either
carry both or say which condition selected the arm. Llama Guard 4 remains
outstanding on the Meta licence — it is a missing comparator, not a resolved one.

---

# C5. Normalise-then-guard

**Cost:** 2--3 cluster jobs, ~2 days. Depends on C4. **Confidence: high it
runs; outcome uncertain.**

## Claim

The activation advantage survives the obvious deployment countermeasure:
normalise the input to plain English first, then apply the strongest text guard.

## Why it matters

The current result shows the probe beats guards that consume transformed text
*directly*. It does not show the probe beats a system permitted to decode
first. Without this, the headline is attackable in one sentence.

## Method

Five arms, all evaluated on the same frozen test prompts, all six conditions:

1. direct text guard (strongest from C4);
2. **translate-then-guard** — back-translate to English with the pinned NLLB
   revision, then guard;
3. **paraphrase-then-guard** — ask `gemma-3-27b-it` for a plain-English
   paraphrase of the request, then guard. Use one fixed paraphrase prompt,
   frozen before any scoring, containing no harm labels or safety framing;
4. activation probe alone;
5. always-normalise upper bound (arm 2 for languages, arm 3 for ciphers).

Report detection, benign FPR, and **cost**: added model calls, added tokens,
and wall-clock per prompt. Audit 50 sampled normalisations by hand for whether
they preserved the request; report the failure rate.

## Acceptance

If normalise-then-guard matches or beats the probe in two or more conditions at
matched realised FPR, the dissertation claim narrows to a cost/latency argument
and must be rewritten accordingly. Report the cost accounting either way.

## Output

`data/c5_normalise_then_guard.npz`, cost table, hand-audit CSV.

---

# C6. Per-example comprehension-conditional monitoring

**Cost:** 1 generation job + 1 judge pass, ~3 days. **Confidence: moderate
(~65--70%).**

## Claim

Within a fixed condition, the frozen activation probe discriminates harmful
from benign prompts substantially better on prompts whose intent Gemma 3 27B
independently demonstrates it has recovered, than on prompts it has not.

## Why it matters

This is the project's registered novel claim (capability--detectability
coupling) at per-example granularity, replacing the current transform-level
version that rests on six points and a WebQuestions trivia proxy. It unifies
the positive language results with the negative reverse result under one
mechanism: an internal monitor can only read semantics the protected model has
itself recovered.

## Partial de-risking already done

On reverse, L53-centroid AUROC falls monotonically across prompt-length
quartiles: 0.910 / 0.905 / 0.771 / 0.722. Longer reversed text is harder to
decode. **But the honest control weakens this**: plain also falls
(0.981/0.992/0.963/0.933), so the evidence is the *interaction* — a drop of
0.19 on reverse against 0.05 on plain, 0.07 on swahili, 0.11 on vowel — not the
raw gradient. That is suggestive and confounded, which is precisely why direct
per-example measurement is worth a job.

## Method

1. **Comprehension task**, administered in a session entirely separate from
   activation scoring, so no category label or English paraphrase can
   contaminate the monitored forward pass. Forced-choice, four options:

   > Here is a message: `<transformed prompt>`. Which of the following best
   > describes what it asks for? (A) ... (B) ... (C) ... (D) ...

   The correct option is a neutral one-line summary of the request; the three
   distractors are neutral one-line summaries of *other* prompts drawn from the
   same condition and the same harm category (or the same benign pool), so
   distractors are topically hard and never harm-cued. Generate summaries once
   from the **plain English** prompt with a fixed prompt template, then reuse
   them for every condition so the task is identical across conditions.
2. Validate on a fixed hand-checked sample of 50 before the full run: the
   summaries must be faithful and the distractors must not be trivially
   eliminable. If hand validation fails, stop and escalate — do not adjust the
   task after seeing detection results.
3. Administer for conditions {swahili, zulu, reverse} on the full frozen test
   split, plus {plain} as the ceiling control. Score comprehension as correct
   or incorrect by exact option-letter match; parse failures are excluded and
   counted.
4. Join per prompt ID to the **already-frozen** probe scores in
   `data/phase1_activation_multilingual_27b.npz`. No new activation extraction.
5. Report, per condition: AUROC and matched TPR@1%FPR on the
   comprehension-correct subset versus the comprehension-incorrect subset, with
   paired bootstrap intervals; and the same split for the strongest text guard,
   which should show no such dependence.

## Pre-declared target

A material result is ≥0.10 AUROC or ≥15 TPR points between the
comprehension-correct and comprehension-incorrect subsets, in at least two
conditions, with intervals excluding zero. Swahili is expected to have little
comprehension variance (99% retention) and is the control, not evidence.

## Negative version

If comprehension-correct reverse prompts remain undetectable, the reportable
claim becomes: understanding the request is *not sufficient* for a
plain-trained linear harm direction to transfer, so comprehension and
representation alignment are separate bottlenecks. That is a genuine
explanatory result and blocks an overclaim the aggregate capability data
currently invites.

## Output

`data/c6_comprehension.jsonl`, `data/c6_comprehension_results.json`, the
hand-validation CSV, and a figure of AUROC split by comprehension per condition.

---

# C7. External-source confirmation

**Cost:** 3--4 cluster jobs, ~4--5 days. **Confidence: moderate (~55--60%).**

## Claim

The frozen probe, with no retraining and no method selection on the new data,
retains at least a 10-point matched-TPR advantage over the strongest text guard
on Swahili prompts from a source it was never trained on, at realised FPR
≤1.5%.

## Why it matters

The current split controls prompt identity but not dataset provenance. Train
and test are both WildJailbreak. `papers/notes/2602.14161-fomin-when-benchmarks-lie.md`
reports raw-probe AUC collapsing from near-perfect to ~0.912 under
leave-one-dataset-out evaluation. This is the difference between "a strong
within-source multilingual result" and a claim that survives review.

## Source selection — decided, not left open

Primary: `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` (CC-BY-4.0, 33,416
human-annotated interactions, both safe and unsafe prompts from one collection,
taxonomy covering weapons, criminal planning and cyber).

Secondary deployment-realism check if time allows: `lmsys/toxic-chat`
(CC-BY-NC-4.0, 10,165 real user conversations).

Acceptance rule for the source, applied **before** any scoring: after
relabelling under the frozen operational rubric it must yield ≥2,000 negatives
and ≥300 operational-harm positives, with no prompt overlap against the
existing train/tune/test/pilot IDs (exact-match and normalised-text check).
If Aegis fails this rule, use ToxicChat. If both fail, escalate — do not
substitute a harmful-only benchmark. StrongREJECT+XSTest may be reported as a
transparent stress test but never as a clean AUROC benchmark, because its
positives and negatives have different provenance.

## Method

1. Relabel with `judge_responses.py --mode prompt`, byte-identical to the
   validated judge, under the frozen operational rubric. Record the parse-error
   rate.
2. Hand-check 50 stratified rows against the rubric before proceeding.
3. Translate to Swahili with the pinned NLLB revision; freeze and hash the
   manifest.
4. Score with the **completely frozen** probe weights and intercept from
   `data/phase1_activation_multilingual_27b.npz`, the frozen centroid, and the
   strongest text guards from C4. One 27B score-forward job.
5. Calibrate thresholds using only external **negatives** — strict (plain
   external negatives) and matched (Swahili external negatives). No refitting,
   no layer reselection, no hyperparameter changes.

## Acceptance

Report the outcome unconditionally. If the advantage disappears, the
dissertation states plainly that language transfer within WildJailbreak is
strong but source transfer is the binding limitation. That negative is worth
reporting and must not be answered by tuning on the external set.

## Output

`data/c7_external_manifest.json` (with SHA-256), `data/c7_external_scores.npz`,
`data/c7_external_results.json`, and a `RESEARCH_LOG.md` entry.

---

# Order, budget and stop rules

| # | Item | Cluster jobs | Days | Confidence |
|---|---|---:|---:|---|
| C1 | Unlabelled threshold transport | 0 | 0.5 | high |
| C2 | Layer aggregation diagnosis | 0 | 0.5 | high |
| C3 | Large negative pool / 0.1% FPR | 1 | 1 | high |
| C4 | Qwen3Guard + Llama Guard 4 | 2 | 2 | high |
| C5 | Normalise-then-guard | 2--3 | 2 | high |
| C6 | Comprehension-conditional monitoring | 1--2 | 3 | moderate |
| C7 | External-source confirmation | 3--4 | 4--5 | moderate |

Run C1 and C2 first — they are CPU-only, near-certain, and immediately
writable. Submit C4 early, because a bad outcome there changes what the main
chapter says. C3 can share a job with C4's scoring pass.

Stop rules:

- Stop the confirmatory track when C1--C5 are written up and either C6 or C7
  has produced a clear positive or a diagnosed negative.
- Stop any individual item the moment its question is answered. Do not convert
  a failed bounded item into a sweep.
- If experiment work begins delaying dissertation chapters, stop experiments.
- No new experiment family in the final week. Only bounded runs whose code,
  inputs and claims were frozen the week before.

## Decisions reserved for David

Do not resolve these unilaterally:

- changing the harm construct, split, primary position, primary metric, or
  calibration protocol;
- substituting any pinned model, revision or dataset named above;
- accepting an external source that fails the C7 acceptance rule;
- redesigning the C6 comprehension task after seeing detection results;
- promoting any exploratory result into a dissertation claim.
