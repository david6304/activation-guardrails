# Current State

Short project snapshot. Overwrite stale items; do not append history here.

## Now

- Repo reset to a CC++ fresh-start skeleton.
- Latest cleanup commit: `59841bf Reset codebase for CC++ fresh start`.
- Safety tag for old work: `pre-ccpp-fresh-start`.
- Phase 1 cacheable scaffold is underway: WildJailbreak normalization,
  response-cache generation, activation-cache contracts, and the first TF-IDF
  logistic text baseline now exist. A dense prompt-final logistic probe scaffold
  now consumes activation-cache NPZ/index artifacts and writes model, scores,
  metrics, and table outputs with provenance metadata.
- Proposal plan in `msc-writeup/ipp/proposal.tex` is the source of truth, but
  use `docs/PROJECT_PLAN.md` first unless exact proposal wording is needed.

## Active Focus

Current implementation focus is the small debug pipeline before any full Gemma
run: data contract, response cache, mock activation cache, cheap baselines, then
the first dense final-token probe.

Next implementation target after this scaffold:

1. Add real activation extraction behind an explicit gate, after a tiny Gemma
   response-cache smoke test is approved.
2. Add a fixed-threshold result-table combiner once both TF-IDF and dense probe
   real/debug outputs are useful to compare side by side.
3. Keep tests around data schema, split construction, metrics, artifacts, and
   metadata.

## Decisions So Far

- Use `main` as stable-ish solo working history.
- Use short-lived branches only for substantial experiment/code chunks.
- Keep `docs/research_log.md` local and ignored.
- Track concise scaffold docs; ignore large/private/generated context.
- Full Gemma response generation is not required yet and remains explicitly
  gated.
- Dense prompt-final probe trains/selects thresholds on vanilla splits only and
  evaluates adversarial transfer at the frozen validation threshold.

## Latest Local Checks

- Mock end-to-end cache/probe path ran locally on all 8000 normalized examples
  using generated outputs under `/private/tmp/agguardrails_dense_probe_mock_e2e`.
- Mock dense probe table used threshold `0.5412154772012588`; because the
  activations are deterministic schema checks rather than model features, the
  near-chance metrics are not reportable experiment evidence.

## Open Checks Before Experiments

- Confirm local/cluster versions for `transformers`, `datasets`, `torch`, and
  `sae-lens`.
- Verify WildJailbreak access and exact Hugging Face dataset revision.
- Run a tiny approved real-generation smoke test before any full response cache.
