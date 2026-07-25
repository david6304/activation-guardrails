# Exploratory Ideas

Decision date: 2026-07-25. Companion to `DISSERTATION_FINISH_PLAN.md`.

Everything here is a **bet**. Each item would materially strengthen the
dissertation if it worked, and it is acceptable if none do. Nothing here may
delay or displace the confirmatory track. The dissertation is already
submittable from the frozen core; this document exists because that safety
margin buys the right to take risk.

The common evaluation backbone, pinned revisions, reuse table and reporting
rules in `DISSERTATION_FINISH_PLAN.md` apply here unchanged. Raw activations
were never persisted, so every item that needs vectors rather than scores costs
a new extraction job.

## Execution state

**A session may start an item here only when David names it in the session
prompt** — except X7, which is cheap triage and may run whenever a queue slot
is free. An idle confirmatory queue is not authorisation. Update this table
when an item lands, exactly as in the confirmatory plan.

| ID | Item | Status | Unblocked when | Jobs |
|---|---|---|---|---:|
| X1 | Causal comprehension x scale | blocked | C6's comprehension task exists | 4--8 |
| X4 | Harm-direction rotation | waiting on rider | any extraction job has run with rider 1 | 0--1 |
| X2 | Payload vs wrapper translation | ready | — | 1--2 |
| X5 | Two targeted readouts | partly riding | rider 1 and 2 stored; else needs own job | 0--1 |
| X3 | Leave-one-family-out | ready | — | 1--2 |
| X6 | Benign-only transport | ready | C2's selected layer recorded | 2--3 |
| X7 | Output-side kill test | **runnable now, no gate** | — | 1 |
| X8 | Adaptive attack | gated | confirmatory track written up; threat model approved | 2--3 |
| X9 | Causal steering | ready | C2's selected layer recorded | 1--2 |

"Ready" means the item has no technical blocker, not that it should be started.
X4 and X5 are cheap only if the standing riders in `DISSERTATION_FINISH_PLAN.md`
were honoured by an earlier job; if they were not, both need a dedicated
extraction and should be repriced accordingly.

## Why the current negatives should not be accepted at face value

The saved evidence points at a readout failure, not absent information:

- reverse all-layer logistic AUROC is 0.556, but the layer-averaged centroid
  reaches 0.767 and a single plain-tune-selected layer reaches 0.829;
- vowel-removal reaches 0.848 on tune and 0.875 on test at a single layer;
- yet the best depth-aggregated method managed only 3.17% matched TPR at
  1.40% FPR on reverse.

So rank information survives representational shift and fails to become
extreme-tail separation. "Reverse is not represented" is the wrong diagnosis.
Several items below attack that gap directly.

The other live vulnerabilities, which the falsifiers below target: one protected
model, one dataset source, the same harm families in train and test, text guards
never permitted to normalise their input, SAE features mixing harm with refusal
and formatting, and roughly twelve tune negatives determining the 1% tail.

---

# X1. Causal comprehension manipulation across model scale

**Bet size: large. Probability of the full result ~45%; of the 27B-only
interaction ~60%. Cost: 4--8 jobs, 3--5 days.**

## Claim

Holding the ciphertext byte-identical, manipulating whether Gemma can decode it
changes activation detectability. Across Gemma 3 sizes, the effect of model size
on detection is mediated by *which individual prompts each model understands*,
not by parameter count as such.

## Why this beats a plain scaling curve

A four-point regression over model sizes is n=4 and confounds depth, width and
probe capacity simultaneously: the all-layer probe gets 62×5376 features at 27B
against far fewer at 4B, so "bigger is better" is guaranteed for reasons that
have nothing to do with comprehension. The scaffold manipulation removes that
confound by varying comprehension *within* a model on *the same* prompt, and the
scale arm then becomes a per-prompt mediation test rather than a curve fit.

## Method

**Arm A — scaffold manipulation, 27B only, run this first.**

Four variants of every test prompt in {reverse, vowel-removal}, with the
harmful payload identical throughout:

1. correct decoder instruction;
2. incorrect decoder instruction (names the wrong transformation);
3. no decoder instruction;
4. correct decoder instruction demonstrated on an unrelated *benign* worked
   example.

For each variant, measure independently: (a) comprehension, using the C6
forced-choice task; (b) frozen-probe detection. Score in separate sessions.

**Arm B — scale ladder, only if Arm A shows an interaction.**

`google/gemma-3-{4b,12b}-it`, adding `1b` only if 4B still shows signal. Verify
layer counts and hidden dims with `AutoConfig` on CPU before writing extraction
code; do not hardcode. Per model: fit the probe on the same plain train split,
measure per-prompt comprehension, and score all six conditions.

**Capacity control, mandatory.** Report alongside the all-layer probe a
capacity-matched variant: per-layer probes taken at matched *relative* depth
(e.g. 0.5·L, 0.85·L) so every model contributes the same feature count. If the
scale effect exists only for the all-layer probe and vanishes under capacity
matching, the effect was probe capacity, and the claim dies.

## Pre-declared target

≥15 points matched TPR or ≥0.10 AUROC between comprehension-correct and
comprehension-incorrect prompts within a condition, replicated in Arm A across
at least two scaffold contrasts, and surviving capacity matching in Arm B.

## Probability justification

27B language capability retention is 88--100% with probe AUROC 0.962--0.980;
reverse retention is 51% with AUROC 0.556. 27B exceeds 12B capability on every
existing transform. Against this: Swahili retention is 99% while detection is
only 59.3% matched TPR, so comprehension is clearly necessary but not
sufficient — which caps the probability well below certainty.

## Reuse vs new extraction

Reuse labels, splits, translations, frozen 27B probe and centroid, and existing
12B/27B WebQuestions capability results. New GPU work: scaffold-variant scoring
at 27B, and full probe fitting plus scoring for each additional model size. Fit
and reduce in-memory within the job; never persist raw activations.

## Negative value

High. If comprehension does not predict detection, or the ladder is
non-monotonic under capacity matching, the comprehension-boundary explanation is
wrong and capability and monitorability scale differently. That is a genuine
boundary result.

## Stop rule

If Arm A shows no interaction at 27B, do not run Arm B.

---

# X2. Payload-versus-wrapper factorial translation

**Probability ~50%. Cost: 1--2 jobs, 2--3 days. Falsifier.**

## Claim

Activation scores move more when the harm-bearing span is translated than when
neutral wrapper text is translated. If wrapper-only translation moves scores as
much, the probe is tracking a global language-conditioned representation shift
rather than harmful semantics.

## Why it matters

Full-prompt translation changes language, length, tokenisation and register at
once. Nothing in the current design separates "the probe reads meaning" from
"the probe reads which language this is". This is the cleanest available test of
the dissertation's central interpretive claim, and it is cheap.

## Method

Four pre-declared variants per prompt, constructed on the frozen test split:

1. all-English (control);
2. all-Swahili (current condition);
3. harm-bearing span translated, wrapper left English;
4. wrapper translated, harm-bearing span left English.

Span identification: mark the action/target span once, using a fixed
LLM-assisted annotation prompt, then hand-audit 50 spans before scoring. For
benign prompts, mark the corresponding request span.

Report score change relative to variant 1 for harmful and benign prompts
separately, plus AUROC per variant.

## Probability justification

Under Swahili the harmful and benign mean scores move in *opposite* directions
(harmful −4.51, benign +0.96), which is inconsistent with a pure global offset
and suggests something semantic is happening. But it does not localise the
cause, which is what this experiment adds.

## Negative value

High, and publishable as a mechanistic limitation: it would show that the
apparent semantic robustness is substantially language and template
entanglement.

---

# X3. Leave-one-harm-family-out

**Probability ~35% of hitting the full target. Cost: 1--2 jobs, 2--3 days.
Falsifier.**

## Claim

A probe trained with one operational-harm macro-family entirely held out still
reaches AUROC ≥0.85 and matched TPR ≥30% at 1% FPR on that unseen family, in
both English and Swahili.

## Why it matters

The current split allows the same harm categories in train and test, so the
probe may be reading topic (cyber, fraud, weapons) rather than a transferable
harmful-*intent* construct. The Phase 3 top features mixing harm, anticipated
refusal and formatting make this a live worry.

## Method

Three or four macro-families only — do not run eight tiny category exclusions;
current test category counts range from 10 to 180. Suggested grouping, fixed
before running: {violent crime + weapons/CBRN}, {cyber}, {fraud + non-violent
crime}, {privacy/PII}. Train one probe per held-out family on the plain train
split, evaluate on that family's test rows under plain and Swahili. All probes
can be fitted from a single in-memory extraction.

## Probability justification

Existing Swahili category TPR already ranges from 23.5% to 80%, so
heterogeneity is substantial and a uniform pass across all families is unlikely.

## Negative value

High. Failure would undermine the "generic harmfulness direction" reading while
leaving the empirical language result intact — a clear topic-versus-intent
diagnosis is a contribution in itself.

---

# X4. Harm-direction rotation predicts transfer

**Probability ~80%. Cost: piggybacks on any extraction job, ~1 day of analysis.**

## Claim

The angle between the plain-English harm direction and the same direction
computed under a transformation predicts that transformation's detection
transfer. One geometric quantity explains the whole results matrix.

## Why it matters

The dissertation currently reports *that* transfer degrades with language
resource level without saying what changes inside the model. A per-layer
`cos(Δμ_plain, Δμ_condition)` curve, where `Δμ` is the class-conditional mean
difference, gives a single-number mechanism and a figure that ties the language
results, the reverse failure and the layer-selection finding together.

## Method

During any job that already extracts activations, additionally save per layer,
per condition, per class: the mean activation vector and the covariance
diagonal. This is 62×5376 floats per class per condition — trivial storage
against the 81 GB of raw activations.

Then compute per layer and condition: `cos(Δμ_plain, Δμ_cond)`; the norm ratio
`‖Δμ_cond‖ / ‖Δμ_plain‖`; and the projection of `Δμ_cond` onto the frozen probe
weight vector. Correlate each against that condition's transfer TPR and AUROC
across all six conditions.

## Prediction

Cosine high for french/hindi, lower for swahili/zulu, near zero for reverse; and
the late-layer cosine should be markedly higher than the mid-layer cosine on
reverse, which would explain the C2 layer-selection result.

## Negative value

Moderate. If rotation does not predict transfer, the shift is not a simple rigid
rotation of the harm direction and the geometry section is a null — still worth
a paragraph, not a chapter.

---

# X5. Two readouts targeted at diagnosed failures

**Probability ~40% that either materially improves worst-condition low-FPR
performance. Cost: 1 job, 2 days.**

## Claim

Two specific, principled changes to how activations are read — not an
architecture search — recover low-FPR performance under shift.

## Why only these two

Gate B already failed on a generic aggregation family; a broad layer × head ×
pooling × classifier search would burn the remaining researcher degrees of
freedom with no confirmatory set. These two are each motivated by a *diagnosed*
failure in the existing data.

**(a) Difference-in-means / whitened LDA instead of logistic regression.**
Logistic regression will happily use low-variance directions that separate the
training distribution and do not survive a representation shift; mass-mean
probing is constrained to the dominant class-difference direction. The reverse
failure — all-layer logistic 0.556 against a centroid's 0.767, where the
centroid *is* essentially a mass-mean readout — is exactly that pathology, and
says the effect is already visible in the saved scores.

**(b) Payload-mean pooling instead of last-token.** L53-centroid AUROC on
reverse falls monotonically across prompt-length quartiles (0.910 / 0.905 /
0.771 / 0.722) while plain falls far less (0.981 → 0.933). A last-token readout
degrades as the payload gets further from the read position. Mean-pooling over
payload tokens is the direct fix, and it was flagged as a sensitivity in the
2026-07-21 log entry and never run.

## Method

Fit all readouts on the plain train split only. Select any hyperparameter on
plain tune only. Compare on all six conditions: all-layer logistic (incumbent),
layer-averaged centroid (incumbent), whitened difference-in-means, and each of
those two under payload-mean pooling instead of `t_inst`. That is six cells, a
closed set — do not extend it.

## Acceptance

Retain a readout only if it improves worst-condition matched TPR at ≤1.32%
realised FPR (the probe's current realised value) without breaching that FPR.
An AUROC-only improvement does not qualify — that was exactly the Gate B failure
mode.

## Negative value

Low in isolation, moderate as part of the readout-geometry diagnosis begun in C2.

---

# X6. Benign-only representation transport

**Probability ~30%. Cost: 2--3 jobs, 3--5 days.**

## Claim

A fixed low-capacity alignment fitted only on paired benign or unlabelled
plain/transformed activations raises reverse or vowel-removal to ≥0.75 AUROC
and ≥15% TPR at ≤1.5% FPR, without costing more than five points of Swahili TPR,
using no transformed harmful labels at any stage.

## Why it matters

A positive here is more interesting than training on labelled ciphered harm: it
says a frozen English harm direction can be transported into an obfuscated
representation using only safe deployment data. It also completes the story
started by C1 — first unlabelled *calibration*, then unlabelled *alignment*.

## Method

Exactly three alignments, in this order, stopping at the first that fails:

1. per-layer score centring and scaling;
2. orthogonal Procrustes fitted on paired benign activations at a tune-selected
   layer or fixed small band;
3. a trajectory-shape readout using differences between adjacent layer scores.

No MLP, no domain-adaptation framework. Keep the English labels and classifier
frozen throughout; the alignment stage may see transformed benign or unlabelled
pairs, never transformed harmful labels. Develop entirely on tune using
`data/phase1_layerwise_27b.npz`; confirm once on a held-out set.

## Probability justification

There is real residual signal to transport (reverse 0.829 at a single layer,
vowel 0.879). Against that, Gate B showed that improving reverse AUROC does not
repair the 1% tail: matched reverse TPR is 2.1% all-layer, 13.6% centroid, and
5.5% for the best plain-selected single layer. The opening is real; the tail is
the hard part.

## Reuse vs new extraction

Development uses saved layerwise scores. Procrustes needs **raw paired vectors**,
so it requires one extraction restricted to preselected layers, fitted in memory.

---

# X7. Output-side monitoring — one kill test, not a programme

**Probability the full output-probe claim works: ~15%. Cost of the kill test: 1
short job, 1 day. Cost of the full programme: 8--12 jobs — do not start it.**

## Why this is nearly dead already

Two independent reasons:

1. It has been tried here. `RESEARCH_LOG.md` (2026-06-29) records the response
   probe at 0.81 in-distribution against TF--IDF's 0.955, and 0.684 against
   0.839 under vanilla→adversarial transfer. The diagnosis was that
   WildJailbreak responses keep harmful semantics legible on the surface, so
   bag-of-words suffices and the probe has no room. Switching the activation
   source to the abliterated model made it worse (0.772), ruling out
   off-policy mismatch as the cause.
2. The mechanism argues against it. If the model complies with a reversed
   prompt it will most likely reply in plain English — in which case a *text*
   guard on the response already closes the cipher hole, and the branch yields
   a text result, not an activation result.

## The kill test

Generate ~200 responses at 27B: 100 harmful and 100 benign reverse-condition
test prompts, fixed IDs, seed 0. Then answer three questions and stop:

1. What fraction of responses to ciphered prompts are in plain English rather
   than in-cipher?
2. What is the harmful-compliance rate (judge-labelled)?
3. Does a TF--IDF classifier on the response text already separate harmful from
   benign at ≥0.85 AUROC?

## Decision rule, fixed in advance

Proceed to response-activation extraction **only if** compliance ≥15--20% *and*
response-text TF--IDF fails to solve the task. If responses are in plain English
and TF--IDF solves it, record the finding — "input-side obfuscation is defeated
at the output stage by a text baseline" — as a discussion point and close the
branch permanently. That is a legitimate and useful deployment observation; it
is simply not an activation contribution.

---

# X8. Adaptive attack against the probe

**Probability of a clean result ~50%. Cost: 2--3 jobs, 3--4 days. Highest risk
to the thesis, therefore highest information.**

## Claim

A bounded, non-adaptive-to-adaptive comparison shows whether the activation
advantage survives an attacker who optimises against the probe rather than
against text guards.

## Why it matters

`papers/notes/40_bailey2024_obfuscated_activations.md` shows latent-space
monitors can be bypassed by activation obfuscation. Every claim in the
dissertation is currently non-adaptive. An examiner will ask. Discovering the
answer is better than conceding it.

## Method

Fixed, bounded threat model declared before running: a prompt-level attacker
with a fixed query budget, optimising a universal suffix or prefix on
*development* prompts only, evaluated for transfer to held-out prompts. Compare
transfer against the English-trained text guard, the multilingual guard, the
activation probe, and a frozen combined system.

Do not optimise per-prompt, do not exceed the declared budget, and do not
present adaptive robustness conclusions from static cipher tests.

## Negative value

High either way. If the probe is easier to attack than text guards, that is a
significant and publishable qualification. If it is harder, it strengthens the
deployment case considerably.

## Stop rule

This is the last thing to start. Do not begin it unless the confirmatory track
is fully written up.

---

# X9. Causal validation of the harm direction

**Probability ~50% of an interpretable result. Cost: 1--2 jobs, 2--3 days.
Lowest priority.**

## Claim

Adding the probe's harm direction to a benign prompt's activations raises the
model's refusal rate, and subtracting it from a harmful prompt lowers it —
establishing that the direction the probe reads is causally the model's own harm
representation rather than a correlate.

## Why it matters

It would convert the whole dissertation from correlational probing to a
mechanistic claim, and it is the natural bridge to the Phase 3 SAE features.

## Method

Steer at the layer selected in C2, with coefficients on a fixed small grid
chosen on tune prompts only. Measure refusal rate, benign utility (WebQuestions
accuracy, to catch generic degradation), and judge-labelled response
harmfulness. Include a random-direction control at matched norm — without it the
result is uninterpretable.

## Risk

Post-hoc steering very easily produces generic capability degradation that looks
like a safety effect. The random-direction control and the utility measurement
are what distinguish the two; both are mandatory.

---

# Ranking and what to run first

| # | Idea | Probability | Jobs | Days | Kind |
|---|---|---:|---:|---:|---|
| X1 | Causal comprehension × scale | 45--60% | 4--8 | 3--5 | mechanism |
| X4 | Harm-direction rotation | 80% | 0--1 | 1 | mechanism |
| X2 | Payload vs wrapper translation | 50% | 1--2 | 2--3 | falsifier |
| X5 | Two targeted readouts | 40% | 1 | 2 | method |
| X3 | Leave-one-family-out | 35% | 1--2 | 2--3 | falsifier |
| X6 | Benign-only transport | 30% | 2--3 | 3--5 | method |
| X7 | Output-side kill test | n/a | 1 | 1 | triage |
| X8 | Adaptive attack | 50% | 2--3 | 3--4 | falsifier |
| X9 | Causal steering | 50% | 1--2 | 2--3 | mechanism |

**Run X1 Arm A first.** It is the only item that could turn the dissertation
from "an activation classifier beats several text guards" into a mechanistic
claim about when internal monitoring is possible at all, and Arm A alone is a
single 27B scoring job.

**Attach X4 to whichever extraction job runs first.** It costs almost nothing
because it only needs class-conditional means, and it is the most likely of
everything here to produce a usable figure.

**Run X7's kill test early and cheaply**, purely to close the output-side
question before it consumes a week.

**Do not run** a combined layer × head × MLP × token-position search, a second
depth-aggregation family, more cipher varieties in search of a positive number,
or the full output-activation programme. The first three are exhausted degrees
of freedom; the last is a different dissertation.

## Decisions reserved for David

- promoting any result here into a dissertation claim;
- starting X8, which needs a separately approved threat model and safety
  protocol;
- proceeding past X7's decision rule;
- any change to the frozen core, which nothing in this document may touch.
