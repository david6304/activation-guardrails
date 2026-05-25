# Project Plan

Concise IPP-derived plan. Use `msc-writeup/ipp/proposal.tex` only when
exact wording, citations, or rubric-sensitive framing is needed.

## Research Question

Working technical question:

> Can we first reproduce the CC++ paper's core guardrail result as faithfully as
> local access allows, then test whether open-model activation probes preserve
> the same benefits under broader jailbreak and interpretability settings?

Broader dissertation question:

> Do internal activations in open-weight instruction-tuned LLMs provide
> complementary and interpretable signals for detecting harmful jailbreak
> prompts beyond text-based guardrails?

## Core Setup

- Immediate priority: a paper-faithful CC++ reproduction before adding
  WildJailbreak, SAE interpretability, or other extension datasets.
- Reproduction rule: match the CC++ paper's data/task framing, train/eval
  splits, cascade/classifier setup, activation positions, aggregation, threshold
  rule, and metrics wherever possible. Any unavailable closed-model,
  production-traffic, or private-data component must be replaced by the closest
  explicit open substitute and labelled as a substitution, not silently treated
  as faithful.
- Existing scaffold status: the WildJailbreak/Gemma response-cache,
  activation-cache, TF-IDF baseline, and dense prompt-final probe code remains
  useful infrastructure, but it is now deferred to the adaptation phase.
- Primary metric remains the low-FPR operating point used by the paper, with
  `TPR @ 1% FPR` and ROC-AUC retained for local comparability unless the paper
  uses a different exact reporting threshold that must be reproduced.
- First implementation task is a reproduction-spec audit from the paper:
  enumerate exact datasets, labels, model(s), probe layers/positions,
  text-classifier baselines, cascade thresholds, smoothing/pooling rules, and
  reported tables/figures to recreate.

## Phase 1 - Faithful CC++ Reproduction

Goal: reproduce the CC++ paper's core empirical claim as closely as feasible
before trying other datasets or dissertation-specific extensions.

Supervisor direction on 2026-05-25: do a full faithful reproduction of the
paper first, then move to other datasets. The immediate plan is therefore no
longer "start with WildJailbreak/Gemma and call it CC++-style"; it is
"reconstruct the CC++ protocol, document unavoidable substitutions, and only
then adapt it to WildJailbreak/open-weight models."

Implementation:

- Read the CC++ paper and produce a reproduction matrix:
  paper component, exact reported setting, local availability, substitute if
  needed, and expected impact on faithfulness.
- Recreate the paper's first-stage text classifier baseline as closely as
  possible before changing datasets.
- Recreate the activation probe training objective and scoring rule, including
  exchange-level framing, token/window smoothing or pooling, and the paper's
  threshold/calibration rule.
- Recreate the ensemble/cascade evaluation because CC++'s central claim is a
  deployed guardrail system, not a standalone activation probe.
- Keep all expensive stages cacheable: examples, model responses, activations,
  probe features/scores, cascade decisions, and final tables.
- Preserve provenance for every run: paper section/table target, dataset/model
  substitute status, config, git commit, seed, model/dataset revisions,
  threshold rule, and environment snapshot.
- Pause before claiming reproduction success unless the result table can be
  mapped directly to a paper table/figure or a clearly labelled local analogue.

Classifiers and comparisons:

- Faithful reproduction baselines come first: text classifier, activation
  probe, ensemble, and cascade as described in the paper.
- Existing TF-IDF and dense prompt-final probes are now debug baselines, not the
  main Phase 1 result.
- Segment-Level Coherence, SAE probes, public guardrails, and Latent Guard-style
  centroid classifiers move after the faithful CC++ reproduction unless they are
  needed as explicit paper substitutes.

Evaluation:

- Reproduce the paper's operating-point metrics and any refusal/over-refusal,
  routed-fraction, compute, or attack-success measurements used in its main
  tables.
- If local substitutions are unavoidable, report two labels in every table:
  "paper-faithful component" and "local substitute".
- Do not introduce WildJailbreak transfer, adversarial-calibrated thresholds, or
  SAE interpretation into the main reproduction table.

Tests and acceptance criteria:

- Reproduction-spec tests/docs: every implemented result maps to a target paper
  table/figure or an explicitly labelled local analogue.
- Data tests: reproduction datasets normalize correctly; splits and label
  semantics match the paper or document the substitute.
- Metric tests: fixed-FPR thresholding, frozen-threshold transfer, ROC-AUC, and
  any paper-specific cascade/over-refusal metrics.
- Feature tests: exchange-token extraction, smoothing/pooling, and score
  aggregation produce aligned labels and example ids.
- Probe tests: synthetic tensors verify the faithful CC++ activation-probe
  scoring path before adding stronger variants.
- Metadata tests: every artifact/result records config path, git commit, seed,
  model/dataset revision, threshold rule, activation stream/source, aggregation
  rule, segment/window settings, regularization settings, and package snapshot.
- First milestone: a small debug config runs the paper-faithful pipeline shape
  end to end and emits a table explicitly mapped to the paper.
- Main Phase 1 acceptance: the faithful reproduction or best-feasible local
  analogue is complete enough to discuss with the supervisor before adding
  WildJailbreak or SAE extensions.

## Phase 1b - Open-Model WildJailbreak Adaptation

Goal: after the faithful CC++ reproduction is established, reuse the scaffold to
test whether Gemma 2 9B IT activation probes add complementary signal to public
text guardrails under WildJailbreak adversarial transfer.

Deferred setup:

- Model: `Gemma 2 9B IT`.
- Dataset: `WildJailbreak`, sampled to 8k prompts: 2k vanilla benign, 2k vanilla
  harmful, 2k adversarial benign, 2k adversarial harmful.
- Vanilla split: 70/15/15, giving 2800 train, 600 validation, 600 vanilla test.
- Held-out OOD transfer set: 4000 adversarial prompts, with both benign and
  harmful subsets so adversarial TPR and FPR can both be measured.
- Activation features: dense hidden states first, initially residual-stream
  activations; add attention/MLP stream checks after the faithful reproduction
  and dense pipeline are stable. Add Gemma Scope SAE features from layers 9, 20,
  and 31 after the dense/text/ensemble pipeline works.
- Main transfer design: train/select thresholds on vanilla data; evaluate on the
  adversarial WildJailbreak holdout. Also compute adversarial-calibrated
  thresholds as a diagnostic, not the main operating point.

## Phase 2 - Harmfulness vs Refusal

Goal: determine whether probes detect harmful intent or model refusal behaviour.

Deliverables:

- Shared activation cache scored with harmfulness labels and refusal labels.
- Refusal labels generated from Gemma 2 9B IT responses, with stronger-judge
  validation on a sample or adversarial subset.
- Token-position comparison: last instruction token vs last prompt token.
- Cross-evaluation of harmfulness-trained and refusal-trained probes.
- Per-jailbreak-tactic breakdown where sample counts support it.
- Refusal-direction abliteration with Heretic if the probe pipeline is stable
  enough for the causal test.

## Phase 3 - SAE Interpretability

Goal: use sparse features for explanations, not just performance comparison.

Deliverables:

- Dense-vs-SAE layer stability comparison.
- Top SAE feature extraction for harmfulness/refusal probes.
- Semantic feature inspection with Delphi where practical.
- Feature signatures by jailbreak tactic.
- Rank correlation/overlap between harmfulness and refusal probe top features.

## Scope Discipline

- Phase 1 is mandatory.
- Phase 2 is core if Phase 1 pipeline is stable.
- Phase 3 is the interpretability contribution, but should not delay a complete
  Phase 1/2 result set.
- Refusal steering/intervention and additional Gemma models are extensions after
  the three core phases.
