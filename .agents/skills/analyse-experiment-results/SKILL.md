---
name: analyse-experiment-results
description: Inspect, summarise, and compare saved ML experiment outputs while preserving the repository's metric definitions and scientific boundaries. Use for requests to analyse a run directory, interpret CSV/JSON/JSONL metrics or logs, compare runs or sweeps, diagnose anomalous curves, or identify the evidence behind a reported result.
---

# Analyse Experiment Results

1. Read `README.md`, the newest relevant `RESEARCH_LOG.md` entries, and the
   launcher or command that produced the outputs. Establish the intended unit of
   analysis, labels, split, metric, threshold, seed, and run identity. Do not
   infer a new primary metric or success criterion.
2. Resolve the requested path. If none is given, identify plausible recent
   output locations and state which one is being analysed. Do not mix artifacts
   from different runs because their filenames look similar.
3. Inspect manifests and metadata before loading bulk results. Record model id or
   revision, input or manifest, parameters, seed, command, commit, and output path
   when present. Flag missing provenance that prevents a fair comparison.
4. Parse the existing structured format directly. Use stdlib or existing
   dependencies; do not add a framework. Compute only summaries appropriate to
   the metric, such as count, mean, standard deviation, range, final/best value,
   class balance, or per-stratum results.
5. Diagnose before interpreting. Check NaN/Inf, empty or duplicated rows,
   incomplete runs, frozen metrics, inconsistent step axes, unequal sample
   counts, threshold drift, and train/eval contamination. A rising loss or a
   train/validation gap is a signal to inspect, not automatic proof of divergence
   or overfitting.
6. Compare runs only after confirming aligned data, labels, metrics, thresholds,
   and aggregation units. Separate exploratory observations from evidence that
   can support a dissertation claim.
7. Report a compact table plus the material anomalies, limitations, and exact
   artifact paths. Do not update `RESEARCH_LOG.md` unless asked or unless the
   user invokes `$checkpoint-research` after accepting the interpretation.
