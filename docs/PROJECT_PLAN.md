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

- Immediate priority: implement the reviewed WildJailbreak/Gemma 3 open-weight
  analogue in `docs/WILDJAILBREAK_CCPP_IMPLEMENTATION_PLAN.md`. This is the
  current best-feasible CC++ replication path after the generated CBRN dataset
  proved almost perfectly text-separable.
- Reproduction rule: match the CC++ paper's data/task framing, train/eval
  splits, cascade/classifier setup, activation positions, aggregation, threshold
  rule, and metrics wherever possible. Any unavailable closed-model,
  production-traffic, or private-data component must be replaced by the closest
  explicit open substitute and labelled as a substitution, not silently treated
  as faithful.
- Existing scaffold status: the exchange schema, generation helpers, metrics,
  activation cache, TF-IDF diagnostic, and minimal SWiM probe are reusable
  starting points. They do not yet satisfy the reviewed WildJailbreak protocol.
- Primary metric remains the low-FPR operating point used by the paper, with
  `TPR @ 1% FPR` and ROC-AUC retained for local comparability unless the paper
  uses a different exact reporting threshold that must be reproduced.
- First implementation task is a reproduction-spec audit from the paper:
  enumerate exact datasets, labels, model(s), probe layers/positions,
  text-classifier baselines, cascade thresholds, smoothing/pooling rules, and
  reported tables/figures to recreate.

## Phase 1 - Faithful CC++ Reproduction

Goal: reproduce the CC++ paper's core empirical mechanism as closely as
feasible with public data and an open-weight model.

Supervisor direction on 2026-05-25 remains the governing constraint: reproduce
CC++ before dissertation-specific extensions. The CBRN generated-data attempt
failed the text-separability diagnostic, so the current route uses
WildJailbreak while retaining the paper's exchange framing, SWiM probe,
exchange classifier, ensemble/cascade, low-FPR calibration, and explicit
substitution labels.

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
  analogue is complete enough to discuss with the supervisor before adding SAE,
  public-guard, or other dissertation-specific extensions.

## Phase 1b - Open-Model WildJailbreak Analogue

Goal: implement the best-feasible public CC++ analogue and test whether Gemma 3
4B activation probes add complementary signal to an exchange text classifier
under paired WildJailbreak adversarial transfer.

Current setup:

- Model family: Gemma 3 4B, with the refusal-ablated artifact as the primary
  protected-model analogue subject to runtime verification.
- Dataset: pinned `allenai/wildjailbreak`, grouped by underlying vanilla intent.
- Primary label: independently adjudicated exchange harmfulness; source intent
  remains secondary provenance.
- Splits: separate fit, development, threshold-calibration, vanilla-test, and
  sealed adversarial-transfer groups.
- Activation features: dense hidden states and exact SWiM first. SAE features
  remain a post-primary extension.
- Main transfer design: all model and system selection uses vanilla data;
  evaluate once on paired adversarial benign and harmful exchanges at the
  frozen calibration threshold.
- Full executable contracts, sample-size gates, controls, GPU pilots, and
  commit boundaries live in
  `docs/WILDJAILBREAK_CCPP_IMPLEMENTATION_PLAN.md`.

## Phase 2 - Harmfulness vs Refusal

Goal: determine whether probes detect harmful intent or model refusal behaviour.

Deliverables:

- Shared activation cache scored with harmfulness labels and refusal labels.
- Refusal labels derived from the same protected-model exchanges, with
  stronger-judge validation on a sample or adversarial subset.
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
