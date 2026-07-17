# Research Log

Short dated entries for important research decisions, results, and changes in
direction. Do not record routine coding work.

## YYYY-MM-DD - Short title

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
