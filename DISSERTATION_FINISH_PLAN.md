# Finish Plan

Rewritten 2026-07-27. Supersedes the confirmatory/exploratory split of
2026-07-25; C1, C2, C4, C5, C6 and X1–X9 as written there are closed or folded
in below. **C3's guard half is still outstanding** — no guard has been scored on
the WildChat pool (~17 GPU-h each); it sits in Extras.
`DISSERTATION_EXPLORATORY_IDEAS.md` is superseded as a queue — keep it only as a
record of ideas already rejected, and do not start items from it.

The frozen core (language transfer) is complete and submittable on its own:
`RESULTS.md` §§1–7, `phase1/RESULTS.md`, `phase2/RESULTS.md`,
`phase3/RESULTS.md`. Everything below either defends that core (C7) or opens the
one line of work that turns a comparison result into a mechanism: **where in the
computation harm becomes readable.**

## State

| ID | Item | Status | Jobs |
|---|---|---|---:|
| C7 | External-source confirmation (Aegis) | source prepared; judge pass pending | 3–4 |
| P1 | Input side — is harm represented before the model decodes? | not started | 2 GPU (capability gate, then scoring) + CPU preflight |
| P2 | Output side — when during generation does harm become readable? | not started | 3 GPU (generate MLP, judge, extract Eddie) |
| E* | Extras | unblocked, non-blocking, take whenever | 0–1 each |

Order: finish C7's judge pass → P1 preflight (CPU) → P1 → P2 gated on P1's
capability check, not on P1's outcome. One item per session.

## Established, do not re-derive

1. The plain-trained all-layer `t_inst` probe beats every text baseline on AUROC
   and TPR@1%FPR in all five language conditions, under strict and
   condition-matched calibration, including against Qwen3Guard-Gen-8B and
   Llama-Guard-4-12B. Matched plain→zulu: probe 75.4→51.2, ShieldGemma
   45.2→15.3, WildGuard (native) 97.4→0.4.
2. Matched calibration needs no labels: 300 unlabelled same-condition prompts at
   1% contamination recover 86–93% of oracle TPR (C1).
3. A 0.1% **background alert rate** — never an FPR, WildChat is unlabelled — on
   50,000 real prompts holds: probe 66.5% plain, 26.9% swahili (C3). The
   centroid collapses under pool calibration. **No guard has been scored on the
   pool**, so there is no cross-detector comparison at 0.1%.
4. The reverse failure is a readout-geometry failure, not absent signal: L54
   centroid 0.829 AUROC against the all-layer probe's 0.556, but only +3.3
   matched TPR points (C2).
5. Four closed negatives — do not reopen: depth-coherent aggregation (Gate B),
   probe+guard fusion (Gate C, NP weight 0.96–0.98 on the probe), layer sweeps,
   response-level probing in the WildJailbreak venue (0.81 vs TF–IDF 0.955).
6. Teacher-forcing beats on-policy abliterated extraction, measured here:
   protected teacher-forced **0.810** vs abliterated on-policy **0.772**
   (2026-06-29). Abliteration degrades the harm representation.
7. Base64 read+write is infeasible below ~280B; only DeepSeek-V4 280B succeeds
   (SAE4Safety Table 1, logged 2026-07-03). rot13 measured dead at 27B (0/30,
   2026-07-13). Gist survives ciphers even where exact decode does not.

## Evaluation backbone

Inherited by everything below unless explicitly overridden. Do not vary to
improve a number.

- **Labels, input side:** prompt-level operational harm from
  `data/judged_main_prompts.jsonl`. Positives exclude `hate & harassment
  targeting protected groups` and the off-taxonomy `harassment` category.
- **Labels, output side:** response-level operational harm, judged with
  `judge_responses.py --mode response` — the rubric whose 90.7% triangulation
  was validated. Negatives are responses to benign prompts.
- **Split:** seed 0, train/tune/test = 5,341 / 1,781 / 1,781. Test = 568 harmful
  / 1,213 benign. Never refit on test.
- **Primary detector:** frozen all-layer logistic probe, no retraining.
  Latent baseline: Zhao-style layer-averaged centroid.
- **Metric:** TPR at a threshold calibrated to 1% FPR; AUROC secondary; realised
  FPR reported beside every TPR. Two calibrations, always both: **strict**
  (plain-English threshold) and **condition-matched** (same-condition tune
  negatives, no shifted harmful examples).
- **Exact operating-point rule, do not restate loosely:** the threshold is
  NumPy's 0.99 quantile with `method="higher"` over the relevant tune negatives,
  and an example is flagged only when its score is **strictly greater than** the
  threshold (`phase1/RESULTS.md` §"Quantile and alert definition"). A different
  quantile or tie convention silently changes every frozen number.
- **Uncertainty:** 10,000-repeat paired bootstrap including threshold
  re-estimation.
- **Storage:** score-forward only. Never persist raw activations — apply the
  probe in-memory during the forward pass and store scalars.

Pinned: `google/gemma-3-27b-it` `005ad3404e59d6023443cb575daa05336842228a`;
ShieldGemma-9b `b8b636016df4540721a098c7aab91c97ec6ee508`; WildGuard
`cbba4823f3e8020e5a74a5e29bf85072def6f2ff`; Qwen3Guard-Gen-8B `4505cb1a…`;
Llama-Guard-4-12B `87acb4b9…`; NLLB `f8d333a098d19b4fd9a8b18f94170487ad3f821d`.
Abliterated generator: `~/models/gemma-3-27b-it-heretic` (Heretic Trial 186,
refusals 50/100, KL 0.0801) — **MLP only**.

Reuse rather than regenerate: `data/phase1_activation_multilingual_27b.npz`,
`data/phase1_baselines_multilingual.npz`, `data/phase1_layerwise_27b.npz`,
`data/c4_modern_guards_lg4.npz`, `data/phase1_translations/*.jsonl`,
`data/c3_pool_probe_plain.npz`.

## Reporting rules

- **Ciphers are instruments, not threats.** Reverse, rot13 and base64 all invert
  deterministically without a model, so a deployment decodes them for free and
  applies the plain guard. Never write that a cipher "defeats text guards" or
  that activations are the remaining defence there. Ciphers control *when* the
  model recovers semantics; that is their only job. Threat framing belongs to
  the language conditions, whose normalisation costs a model call and degrades
  with the shift, and to lossy encodings (vowel-removal), which cannot be
  inverted even in principle.
- **The output-side claim is latency, not necessity.** With English responses a
  text guard on the response does work. The claim is that the probe fires
  earlier, measured in tokens. Necessity is a motivation paragraph citing
  DeepSeek-V4 read+write, explicitly labelled future work.
- Condition-matched results are **adaptation** results; never call them
  zero-shot.
- Report every preselected condition, favourable or not.
- Everything developed after seeing the Phase 1 test is **exploratory** on that
  test. **C7 does not confirm P1 or P2** — it evaluates the frozen plain/Swahili
  *prompt* detector on an external source, and says nothing about cipher read
  positions or response-position probing. P1 and P2 are exploratory full stop,
  and must be written that way. There is no confirmatory set for them in the
  remaining budget; that is a stated limitation, not something to engineer
  around.
- Say "best among evaluated open guards under this protocol".

---

# C7. External-source confirmation

**Cost:** 3–4 jobs. **In flight.**

Train and test are both WildJailbreak, so the frozen result controls prompt
identity but not dataset provenance. Score the completely frozen probe, centroid
and strongest C4 guards on NVIDIA Aegis 2.0, relabelled under the frozen rubric
and translated to Swahili with the pinned NLLB revision.

Source pre-check passed: 23,489 rows after dedupe and truncation filtering,
11,288 Aegis-safe, **zero** normalised-text overlap with the 10,000 Phase 1
prompts. Manifest and sharded judge input are committed
(`data/c7_external_manifest.json`, SHA-256 recorded).

Next: run the judge pass, hand-check 50 stratified rows, then translate, freeze
and hash the manifest, then one 27B score-forward job.

**Freeze a disjoint external tune/test partition before scoring** — this was
left open in the log and must not stay open. Thresholds are set on external
*tune* negatives only; TPR and realised FPR are reported on the external *test*
partition. If the same negatives set the threshold and measure the FPR, C7 is
not held-out confirmation and the whole item is void. Calibrate strict on plain
external tune negatives, matched on Swahili external tune negatives. No
refitting, no layer reselection.

**Acceptance:** ≥10-point matched-TPR advantage over the strongest guard at
realised FPR ≤1.5%. Report the outcome unconditionally; if it disappears, state
plainly that language transfer within WildJailbreak is strong but source
transfer is the binding limitation, and do not tune on the external set.

---

# P1. Input side — is harm represented before the model decodes?

**Cost:** CPU preflight, then 2 GPU passes — the capability gate generates, so
it is not free — then scoring. **Confidence: high that it runs; the outcome is
genuinely uncertain.**

## Claim

The cipher failure is a property of the **read position**, not only of the
readout. Harm is not represented at the end of a ciphered instruction because
the model has not yet recovered the semantics. Where the transform is one the
model resolves during the prompt pass, the frozen probe should read harm at the
input; where recovery needs serial work, it should not.

`t_inst` was frozen as a protocol invariant to stop position-shopping. That
correctly prevented p-hacking and simultaneously made the read position
invisible as a research variable. This is the item that tests it.

## Gate — the capability check adjudicates a real conflict

The 2026-07-03 entry dropped base64 on SAE4Safety Table 1 (exact decode
collapses at 27B); David's hand test says 27B reads base64 and responds.
Both can hold — gist retention is not exact decode, and short strings are far
easier than long ones. **Run `capability_qa.py` on base64 at 27B first.** If it
returns gist-only on realistic prompt lengths, say so and read every P1 base64
number in that light. rot13 needs no check (0/30, already measured).

## Conditions

Added to the frozen split; labels, split and rubric unchanged.

| id | construction | role |
|---|---|---|
| `base64` | base64 of the plaintext, existing decoder-instruction wrapper | transform the model may resolve in the prompt pass |
| `base64_shuffled` | 4-character base64 blocks permuted, seed 0 — valid base64, same wrapper, same length | **mandatory control**: kills the surface-statistics explanation |
| `rot13` | rot13, decoder instruction identical in form to the reverse scaffold | local transform the model cannot do — isolates fluency from locality |

`rot13` must test decoding, not cipher identification: name the cipher exactly
as the reverse condition does.

## Preflight — all CPU, all before the job

1. **Verify the read position.** `phase1_activation.py` takes the token before
   `<end_of_turn>`. If the template appends plain-English text *after* the
   payload, `t_inst` sits downstream of the ciphertext. Read the template; do
   not assume it matches `capability_check.py`. Score **both** the final
   ciphertext token and the existing `t_inst`.
2. Token-length distribution per new condition; base64 inflates ~33% in
   characters and fragments under the tokeniser.
3. **Truncate the plaintext, then encode.** Never truncate an encoded string —
   base64 is phase-sensitive in 4-character groups.
4. **Decode-then-guard ceiling** for every cipher condition: a string operation
   plus the existing plain guard scores. Any cipher number that does not beat
   this ceiling is mechanistic evidence only. Put it on the page first.

## Method

One scoring job: probe and centroid, both read positions, all three conditions,
tune and test. Take the full layerwise curve — free in the same extraction, and
it separates "resolved early in depth, i.e. looked up" from "peaks late like
reverse, i.e. decode consumes depth". Comparators: strongest C4 guard per
condition, char TF–IDF, decode-then-guard. Do not rerun all eight baselines.

## Pre-declared acceptance — fix before running

Primary test is **`base64` against `base64_shuffled`**, paired, on **both** test
AUROC and condition-matched TPR@1%FPR, with 10,000-repeat paired intervals
excluding zero. An AUROC-only gain does not qualify: C2 reached 0.829 AUROC at
5.5% matched TPR and repeating that failure mode is the main risk here.

Secondary, reported regardless: predicted ordering `base64` > `vowel` >
`reverse` ≈ `rot13`; `rot13` ≈ chance attributes the effect to fluency rather
than locality.

If `base64` matches `base64_shuffled`, the input-side hypothesis is dead and the
finding is that no cipher is decoded during the prompt pass regardless of
fluency — a clean mechanistic result, and the motivation for P2.

**Identification limit, state it:** four observational conditions vary
transform, tokenisation, length and invertibility at once, and vowel-removal is
lossy where the others are reversible. This is an ordinal prediction, not a
factorial design. Per-prompt fluency is tested by the comprehension extra.

## Output

`data/p1_position_scores.npz`, `data/p1_position_results.json`,
`data/p1_decode_then_guard_ceiling.json`, `RESEARCH_LOG.md` entry recording read
positions, construction seeds and preflight findings.

---

# P2. Output side — when during generation does harm become readable?

**Cost:** 3 GPU stages — generation on MLP, a Qwen judge pass, extraction on
Eddie. Chain them into shared allocations where possible. **Confidence:
moderate. This is the item that could change what the dissertation is about.**

## Claim

**Given the same harmful prompt**, a monitor reading the protected model's
activations over a response stream identifies *which* generations are harmful
**earlier in the token stream** than a text monitor reading the identical
tokens.

If a generation contains harmful content the model represented that content
before emitting it, so the representation exists; its linear accessibility to a
frozen plain-trained probe is the empirical question. That makes a null
interpretable too: "not represented" is excluded by construction, so a failure
is unambiguously a readout/alignment result.

## The contrast — this is the part that is easy to get wrong

The naive design (positives = harmful responses, negatives = responses to benign
prompts) is **invalid**. The probe is trained on *prompt* harm, so it separates
those two classes at `k=0` from the prompt alone and the latency curve starts at
ceiling. It would produce a strong-looking headline that measures nothing.

Primary contrast, therefore: **within harmful prompts only**, judged-harmful
responses against judged-benign non-refusal responses. The prompt is held
roughly constant and the only thing varying is what the model actually produced.

- **Report the `k=0` prompt-only score explicitly.** It is the baseline the
  latency claim must beat; any gain must be over the prompt score, not over
  chance.
- Benign-prompt trajectories are reported **separately**, as the operational
  false-positive picture, never as the negative class of the primary contrast.

## Honest scope

Two limitations, both stated in the write-up rather than engineered around.

**Off-policy.** The abliterated model emits the response; the protected model
only reads it afterwards. So this does **not** establish what the protected
model represented before emitting text it never emitted. The claim is
**off-policy prefix detection**: given a response stream, an activation monitor
on the protected model flags it earlier than a text monitor. That is a real
deployment claim about monitoring a stream, and it is all this design supports.
The log already draws exactly this off-policy/on-policy distinction
(2026-06-15). If compliance turns out to allow it, an on-policy protected-model
arm is the deployment-eval upgrade — do not assume it will.

**English responses.** A text guard on an English response does work, so this
cannot show necessity — only latency, i.e. how many tokens of harmful output
stream to a user before a text monitor can act. The 2026-07-03 decision dropping
the read+write obfuscated-output venue is not reversed; 27B cannot write in
cipher, and necessity stays a motivation paragraph.

## Pipeline

1. **Generate (MLP, abliterated 27B).** Responses to the frozen test and tune
   prompts under {plain, base64}. Free generation, no decode scaffold — a
   scaffold that emits plaintext makes the result a text result.
2. **Judge (response-level), after a schema change.** `judge_responses.py
   --mode response` currently cannot support this: its rubric folds refusals,
   deflections, safe completions and incoherent output into `harmful=false`, and
   the parsed verdict has only `harmful` and `category` — no refusal field. The
   `REFUSAL_PREFIXES` heuristic in `generate_responses.py` is documented as "not
   used for labelling" and must not become a label. **Add an explicit refusal
   verdict to the schema and validate it on ~50 hand-checked rows before any
   filtering**, otherwise the benign-response class is mostly refusals and the
   contrast is easy for the wrong reason. Three strata result: harmful,
   benign-non-refusal, refusal. Refusals are excluded from the primary contrast
   and reported as a count.
3. **Extract (Eddie, protected 27B).** Teacher-force prompt+response through the
   protected model in **one forward pass per example** — no incremental
   generation, no activation cache, probe applied in-memory at every response
   position, scalars only. This is why teacher-forcing is right here beyond the
   0.810/0.772 result: it makes the entire latency curve cost one pass, it keeps
   "protected model" meaning what it means in every other chapter, and it splits
   the job across venues so the abliterated weights never leave MLP — only a
   JSONL of responses moves.
4. **Score, CC++-style.** Probe score at each response position; sequence rule =
   **max score so far**.

## Statistics — required, or the number means nothing

- **Sequence-level operating point.** Scoring every token is multiple looks, and
  EOS makes the later risk set class-dependent. Predeclare a horizon, calibrate
  every monitor's whole trajectory to the **same** sequence-level FPR on tune
  negatives, and treat EOS as censoring rather than dropping rows.
- **Identical token stream, defined once.** `generate_responses.py` saves
  decoded text and a token count, not token IDs, so re-tokenising on the reader
  side can move prefix boundaries. **Define every horizon *k* in the protected
  reader's tokenisation** and construct all comparator prefixes by decoding from
  it; the text comparators have their own tokenisers, so text must be derived
  from the reader's prefix, never the other way round. Alternatively save
  generated IDs plus tokenizer identity at generation time.
- **Plain control at the same positions.** The probe was fitted at `t_inst`;
  position shift degrades it on its own. If plain degrades equally, position is
  the cause, not decoding.
- **Compliance estimate first.** Require a stated minimum effective sample in
  *both* primary strata — judge-confirmed harmful and benign-non-refusal
  responses to harmful prompts — before any latency analysis. The abliterated
  27B refuses 50/100 on the Heretic benchmark, so the usable sample may be well
  below the prompt count.

## Detector choice

Primary is the **frozen prompt-trained probe**, no retraining — the strong,
zero-adaptation claim, and it tests whether a prompt-harm direction reads
harmful response content. Only if that fails outright, fit a response probe on
the train split and report it separately; note that response-level probing
already lost to TF–IDF in the WildJailbreak venue (0.81 vs 0.955), so a weak
result there is expected and is not news.

## Acceptance

Report the latency curves either way. Do **not** use "50% of asymptotic TPR" —
that rewards a weak detector with a low asymptote. Two pre-declared quantities
instead, both at matched sequence-level FPR:

- the smallest *k* at which each monitor reaches a **common absolute TPR
  target** (fix the target before running); and
- the censored first-crossing distribution per monitor, compared paired.

The headline sentence is warranted only if the probe's first crossing is
strictly earlier than every text comparator's, with paired intervals excluding
zero, **and** the probe's advantage exceeds what the `k=0` prompt-only score
already delivers.

## Output

`data/p2_responses.jsonl`, `data/p2_judged.jsonl`, `data/p2_latency_scores.npz`,
`data/p2_latency_results.json`, and a latency figure (TPR vs *k*, one line per
monitor, faceted by condition).

---

# Extras

Unblocked, independent, non-blocking. Take one whenever a queue slot is free or
a CPU afternoon appears. None may delay C7, P1 or P2. Ranked by value.

1. **Same-base text control** (1 job). Every text baseline is ≤12B while the
   probe reads a 27B model, so nothing currently separates "activations carry
   what text doesn't" from "27B beats 8B". Prompt `gemma-3-27b-it` itself as a
   harm classifier with ShieldGemma's composite guideline, take P(unsafe) at the
   decision position, score all six conditions. If the probe still leads, the
   claim upgrades to "the model represents harm it does not report".
2. **Harm versus refusal** (1 short job). Score the frozen plain-trained probe
   on `gemma-3-27b-it-heretic` prompt activations. Survives ⇒ the probe reads
   intent, not refusal precursors. Specified 2026-07-17 and dropped for time;
   the model now exists, and it doubles as the probe-transfer check P2 would
   want.
3. **Translation fidelity** (1 generation + 1 judge pass). Back-translate the
   568 test positives per language with `gemma-3-27b-it` — a different system
   from NLLB, so forward and backward error are not measured against each other
   — and re-judge. If 15% of Zulu positives no longer read as harmful, the
   ceiling in that condition is 85%, not 100%. Converts a concession into a
   quantified limitation.
4. **Vowel-removal, the one non-invertible condition** (rides P1's job + CPU).
   Vowel-removal has no row in `RESULTS.md` §4. Score the probe and the four
   guards on it at the frozen protocol. Then, on CPU, recompute the centroid's
   vowel operating point against the frozen 50,000-prompt WildChat pool — the
   centroid carries this claim and C3 showed it collapses under pool calibration
   (56.5%→23.1% on plain). If it collapses here too, cut the threat framing.
5. **Per-prompt comprehension** (1 job). The forced-choice task specified in the
   old C6: neutral one-line summaries generated once from the plain English
   prompt, three topically-hard distractors from the same condition and harm
   category, hand-validated on 50 rows before the full run. Administered in a
   session separate from activation scoring. This is what identifies fluency as
   the mechanism *within* a condition, which P1's four conditions cannot.
6. **Motivation paragraph** (0 jobs). DeepSeek-V4 280B succeeds at base64
   read+write where ≤35B models fail (SAE4Safety Table 1, logged 2026-07-03).
   Pair with the local guard-blindness numbers. Label as motivation for future
   work, never as evidence about frontier monitorability — capability does not
   imply monitorability, and reverse is this project's own counterexample.
7. **SAE cross-language feature overlap** (0–1 jobs). Phase 3 is currently a
   performance null because an SAE cannot beat the dense features it
   reconstructs. The interpretability question is whether the *same* features
   fire for harm in English and Swahili. Check whether the Phase 3 job saved
   feature activations or only scores; if only scores, this needs one job and is
   the last one to spend.
8. **C3's guard half** (~17 GPU-h per guard). ShieldGemma and Qwen3Guard on the
   frozen 50,000-prompt WildChat pool, giving the cross-detector comparison at a
   0.1% background alert rate that §7 currently cannot make. Expensive for what
   it adds; take it only if a long slot is otherwise idle.
9. **Harm-direction rotation** (0 jobs if riders honoured). Per layer, per
   condition, per class: mean activation and covariance diagonal. Then
   `cos(Δμ_plain, Δμ_cond)`, the norm ratio, and the projection onto the frozen
   probe weights, correlated against transfer TPR. One geometric quantity for
   the whole matrix, and the likely explanation if P2 nulls.

**Standing rider.** Any job running a forward pass over prompts must also save
class-conditional means per layer per condition (harmful mean, benign mean,
covariance diagonal). It costs nothing at extraction time, costs a full job to
recover later, and it unblocks extra 9 entirely.

---

# Stop rules

- Stop when C7 and P1 are written up and P2 has produced a clear positive or a
  diagnosed negative.
- Stop any item the moment its question is answered. Do not convert a failed
  bounded item into a sweep.
- If experiment work starts delaying chapters, stop experiments.
- No new experiment family in the final week — only runs whose code, inputs and
  claims were frozen the week before.

# Decisions reserved for David

- changing the harm construct, split, primary detector, metric or calibration;
- substituting any pinned model, revision or dataset;
- accepting an external source that fails C7's acceptance rule;
- promoting any P1/P2 result to a confirmed claim before C7 lands;
- starting anything not listed above.
