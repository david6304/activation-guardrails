# Project Plan

Concise proposal-derived plan. Use `msc-writeup/ipp/proposal.tex` only for exact
wording, citations, or rubric-sensitive framing.

## Research Question

Do internal activations in open-weight instruction-tuned language models
provide complementary and interpretable signals for detecting harmful
jailbreak exchanges beyond text-only guardrails?

## Phase 1 - Activation Guardrail Baseline

Goal: establish a faithful or clearly labelled best-feasible reproduction of
the core CC++ activation-guardrail mechanism.

Required comparison:

- strong text classifier;
- activation probe;
- ensemble or cascade matching the reproduced system;
- frozen low-FPR evaluation.

Decision gates:

- the dataset and labels support the intended claim;
- train, selection, threshold-calibration, and final-evaluation roles are
  isolated;
- text shortcuts and simple confounds are measured;
- implemented components are labelled faithful, substituted, or local;
- a small end-to-end run passes before a reportable run is authorized.

Phase 1 is complete only when the result maps to a paper result or an explicitly
bounded local analogue and is suitable for supervisor review.

## Phase 2 - Harmfulness And Refusal

Goal: test whether learned activation signals represent harmfulness, refusal,
or a mixture.

Candidate deliverables after Phase 1 is stable:

- harmfulness and refusal labels over shared exchanges;
- position and layer comparisons;
- cross-evaluation of harmfulness-trained and refusal-trained probes;
- breakdowns by attack or prompt group where sample sizes support them;
- a causal refusal-direction test only if the observational pipeline is sound.

## Phase 3 - SAE Interpretability

Goal: assess whether sparse features provide useful explanations of the dense
probe signal.

Candidate deliverables after Phase 2:

- dense-versus-SAE performance and stability;
- top-feature inspection for harmfulness and refusal;
- overlap and rank comparisons between concepts;
- tactic-level feature signatures where statistically defensible.

## Scope And Ordering

- Phase 1 is mandatory.
- Phase 2 follows only after the baseline and evaluation protocol are stable.
- Phase 3 must not delay a complete Phase 1 or Phase 2 result.
- Additional models, public guardrails, steering, or intervention studies are
  extensions, not default infrastructure.
- Each phase advances through bounded milestones defined and accepted under
  `docs/WORKFLOW.md`.

## Reportable Evidence

Every reported result must identify:

- claim and target comparison;
- git commit and config;
- model, tokenizer, and dataset revisions;
- labels, group unit, split roles, and seed;
- feature source and aggregation rule;
- metric and threshold-selection rule;
- environment and hardware;
- exploratory versus reportable status;
- known substitutions, limitations, and review outcome.
