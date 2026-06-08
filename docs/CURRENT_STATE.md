# Current State

Short project snapshot. Replace stale facts rather than appending history.

## Now

- The repository contains a CC++-style data, generation, activation, metric,
  text-diagnostic, and SWiM-probe scaffold.
- The implemented CBRN experiment is parked. Its harmful and matched-benign
  examples were almost perfectly text-separable, so it cannot support the
  intended claim that activations add useful signal beyond text.
- The CBRN implementation remains intact and is documented in
  `docs/PARKED_CBRN.md`.
- Earlier WildJailbreak implementation plans and agent-specific multi-agent
  workflows were discarded during the documentation reset. Their
  non-authoritative snapshot is recoverable from commit `280d363`.
- No replacement experiment protocol is currently frozen.

## Next Bounded Milestone

Select and freeze the next empirical milestone before changing experiment code.
The milestone must specify:

1. the precise claim and comparison;
2. dataset, model, labels, split roles, and leakage controls;
3. exploratory or reportable status;
4. primary metric and threshold-selection rule;
5. required baselines and acceptance criteria;
6. provenance and review requirements.

This is a planning milestone. It does not authorize implementation of a
WildJailbreak pipeline, SAE pipeline, public-guard comparison, or other future
stage.

## Stable Decisions

- Reproduce the core activation-guardrail mechanism before adding
  dissertation-specific extensions.
- Evaluate activation methods against a strong text baseline.
- Keep harmfulness and refusal conceptually distinct.
- Use group-safe splits and threshold calibration separated from final
  evaluation.
- Treat `TPR @ 1% FPR` as the intended primary operating-point metric when the
  negative denominator supports it; retain ROC-AUC as a secondary metric.
- Keep expensive stages cacheable and provenance-linked.
- Diagnose surprising or near-ceiling results before extending the pipeline.

## Verification Baseline

Verification at reset:

- 2026-06-08: `python -m pytest` passed with 61 tests in 15.75 seconds.
- 2026-06-08: `python -m ruff check src/agguardrails scripts tests` passed.

These results describe only the checked-out commit and recorded environment.
They must not be treated as evidence that later revisions pass.

## Open Decision

The next protocol needs a defensible dataset and label design that is not
dominated by surface-form shortcuts. Because this choice can change the
dissertation's empirical claim, it requires plan-first scientific review under
`docs/WORKFLOW.md` before implementation.
