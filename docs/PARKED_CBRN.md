# Parked CBRN Experiment

This document records the preserved CBRN experiment without treating it as
active work.

## Status

The experiment is parked because its generated harmful examples and bespoke
matched-benign examples were almost perfectly separable with a TF-IDF text
baseline. That makes the dataset unsuitable for the intended activation-versus-
text comparison. Passing generation, length, or on-policy checks did not resolve
this central confound.

Do not resume or extend this experiment unless a new milestone explicitly
reactivates it with a defensible data redesign.

## Preserved Implementation

- Config: `configs/ccpp/gemma3_4b_it_public_cbrn_probe.yaml`
- Data and generation entrypoints: `scripts/ccpp/`
- Cluster helpers: `scripts/cluster/`
- Reusable code: `src/agguardrails/`
- Tests: `tests/`
- Local caches and outputs: `artifacts/ccpp/` and `results/ccpp/`
- Recovery tag predating the fresh scaffold: `pre-ccpp-fresh-start`
- Full pre-reset documentation snapshot: commit `280d363`

These files remain code history and potentially reusable components. Their
presence does not make the CBRN protocol active or reportable.

## Original Design

- Harmful prompts came primarily from ClearHarm `rep40`, grouped by underlying
  prompt identity; HarmBench chemical/biological rows were a small
  supplementary source.
- Benign prompts were generated as dual-use-adjacent CBRN/science prompts with
  safety, diagnostic, policy, education, or legitimate-lab framing.
- Harmful and benign completions used the same protected-model analogue to
  reduce generator/style mismatch.
- Splits and evaluation were intended to operate at prompt-group level rather
  than treating paraphrases as independent.
- The planned primary operating point was `TPR @ 1% FPR`, subject to enough
  independent negative groups; ROC-AUC was secondary.

## Useful Lessons

- On-policy generation and length matching do not rule out lexical shortcuts.
- Unique source groups, not paraphrase rows, determine effective sample size.
- A low-FPR claim requires enough independent negative groups to estimate the
  operating point.
- Text diagnostics must run before expensive activation extraction.
- Near-ceiling text performance is a reason to redesign the comparison, not to
  proceed directly to a larger activation experiment.

## Reactivation Gate

A reactivation proposal must:

1. explain how harmful and benign examples are matched without reproducing the
   prior lexical/style shortcut;
2. define independent group counts adequate for the stated metric;
3. freeze split, text-baseline, leakage, and threshold rules;
4. state which existing artifacts remain valid;
5. pass a small text-separability gate before GPU-scale extraction;
6. receive scientific-risk review under `docs/WORKFLOW.md`.
