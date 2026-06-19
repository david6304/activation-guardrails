# Research Log

Short dated entries for important research decisions, results, and changes in
direction. Do not record routine coding work.

## YYYY-MM-DD - Short title

## 2026-06-19 - Rubric gate cleared (90.7%) via blind triangulation

Re-judged the 512 pilot with the refined rubric (stereotype-fiction carve-out) and
ran a three-way **blind triangulation** to validate it: David and Claude independently
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
