# ML Research Code Review

Review for bugs that could invalidate evidence. Lead with findings, ordered by
severity, and cite the file and line. For each finding, explain the failure mode,
its effect on results, and the smallest correct fix. Skip categories with no
finding. End with residual risks or checks that could not be run.

Prioritise:

- **Labels and splits:** leakage across train, validation, calibration, or test;
  fitting transforms before splitting; duplicated examples; mismatched labels;
  accidental use of prompt intent where response harm is the target.
- **Metrics and thresholds:** wrong aggregation unit, unequal-batch averaging,
  micro/macro confusion, threshold selection on evaluation data, direction of
  optimisation, NaN/Inf handling, or denominator changes between runs.
- **Losses and masking:** reduction semantics, padding and ignore masks,
  off-by-one targets, weighting, gradient accumulation, and numerical stability.
- **Model state and activations:** `train()`/`eval()` state, inference mode,
  token position and layer selection, tokenizer/chat-template mismatch, cached
  activation provenance, dtype/device conversions, and model revision drift.
- **Comparability:** seeds and manifests, generator or judge changes, differing
  sample filters, overwritten outputs, ambiguous checkpoints, and missing run
  metadata needed to reproduce a reported number.
- **Cluster execution:** requested hardware matching the measured requirement,
  offline cache assumptions, unique output paths, dirty/unidentified commits,
  resume safety, and shell/Python argument agreement.

Do not require deterministic algorithms, exhaustive validation, production
hardening, or extra tests by default. Raise them only when their absence could
materially change the experiment's conclusion.
