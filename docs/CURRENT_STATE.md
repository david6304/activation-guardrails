# Current State

Short project snapshot. Replace stale facts rather than appending history.

## Now

- The experiment implementation and tests have intentionally been removed so
  the repository can be rebuilt through small, understandable milestones.
- The previous implementation is recoverable from commit
  `d389a35dd888ef773e4ecc5c69d2e17abb61a2e2`.
- The tracked repository currently contains only the research scaffold, active
  project guidance, dependency metadata, and directory ownership documentation.
- No replacement experiment protocol is currently frozen.

## Next Bounded Milestone

Define and review a planning-only dataset and label design milestone for the
Phase 1 activation-guardrail baseline. Compare a small number of defensible
options against the intended claim, including label provenance, group unit,
split roles, leakage and surface-form risks, availability, and reproduction
faithfulness.

The deliverable is a decision proposal with alternatives, risks, unresolved
questions, and acceptance evidence for a later implementation milestone. Do not
download data, inspect final-evaluation examples, or add experiment code. No
empirical implementation is authorized until the scientific choices and
observable acceptance criteria are explicitly approved under
`docs/WORKFLOW.md`.

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

This is a documentation-and-scaffold repository. There is currently no
experiment test suite or executable pipeline. Verification should check
documentation, tracked structure, relative links, formatting, and diff scope.

## Open Decision

The next protocol needs a defensible dataset and label design that is not
dominated by surface-form shortcuts. Because this choice can change the
dissertation's empirical claim, it requires plan-first scientific review under
`docs/WORKFLOW.md` before implementation.
