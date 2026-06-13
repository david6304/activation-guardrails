# Current State

Short project snapshot. Replace stale facts rather than appending history.

## Now

- The experiment pipeline is being rebuilt through small, understandable
  milestones rather than restoring the previous implementation wholesale.
- The previous implementation is recoverable from commit
  `d389a35dd888ef773e4ecc5c69d2e17abb61a2e2`.
- The accepted WildJailbreak manifest implementation reproducibly builds and
  validates the approved 400-example, group-safe smoke manifest. Manifest
  schema v2 omits `tactics`: the pinned `train/train.tsv` source exposes only
  `vanilla`, `adversarial`, `completion`, and `data_type`, so tactics are
  recorded as unavailable rather than derived or substituted.
- The response-generation implementation is locally validated with CPU-safe
  mocks and dry-run coverage. No protected-model response generation has been
  run.
- A two-hour, single-GPU Wintermute job script is prepared and locally
  syntax-checked. It has not been submitted.
- A replacement end-to-end protocol is approved only for a non-reportable
  smoke test. Its labels, metrics, and results are provisional and cannot
  support dissertation claims.

## Next Bounded Milestone

Review and authorize the exact cluster target and command for the approved
non-reportable response-generation smoke run.

Scope:

- use the accepted smoke manifest without changing its examples or splits;
- load Heretic Gemma 3 4B and its processor from the protected local cache;
- generate one sampled response per example with the approved provisional
  decoding settings and deterministic per-example seeds;
- preserve resumable per-example outputs and provenance without logging prompt
  or response text;
- keep all outputs explicitly non-reportable.

For account `s2296274`, the cluster shorthand
`~/models/gemma-3-4b-it-heretic` resolves to
`/home/s2296274/models/gemma-3-4b-it-heretic`. The tokenizer/processor defaults
to that same path unless inspection of the cached checkpoint establishes a
different path.

Local implementation does not authorize remote access or execution. Before a
run, inspect the exact command, cluster target, resources, cached processor and
model loading, output location, and dry-run summary. Do not judge responses,
extract activations, or begin downstream stages in this milestone.

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

## Approved Smoke Protocol

The first end-to-end run is exploratory, provisional, and non-reportable:

- use 400 WildJailbreak prompts balanced across harmful/benign and
  vanilla/adversarial quadrants;
- train on both vanilla and adversarial examples using lineage-preserving,
  group-safe 70/15/15 train/calibration/test splits;
- sample one response per prompt from Heretic Gemma 3 4B with temperature
  `0.7`, `top_p` `0.9`, deterministic per-example seeds, and a provisional
  4096-token response cap;
- judge response harmfulness and refusal separately with aligned Gemma 3 12B,
  then derive `exchange_unsafe = response_harmful`;
- train a TF-IDF logistic-regression full-exchange baseline and an all-layer
  CC++ SWiM linear activation probe;
- simulate streaming detection causally over assistant tokens;
- use TPR at 1% FPR only as a pipeline check, alongside ROC-AUC, PR-AUC,
  confusion counts, and quadrant breakdowns.

Judge labels and all smoke metrics are provisional. Public guard models,
cascades, tactic holdouts, 0.1% FPR, and reportable runs are deferred.

## Verification Baseline

Local verification covers pinned-source schema preflight, manifest
construction, response artifact contracts, resume behavior, exact chat-template
invocation, decoding metadata, termination classification, argument parsing,
and dry-run behavior. It does not establish that the protected checkpoint loads
on the cluster, fits the selected GPU, or produces scientifically valid
responses.

## Open Decisions

The smoke protocol does not freeze a reportable dataset, judge, threshold, or
baseline comparison. Surface-form shortcuts, judge validity, lineage quality,
split feasibility, reproduction faithfulness, and the evidential role of
refusal labels must be diagnosed before any reportable protocol is proposed.
