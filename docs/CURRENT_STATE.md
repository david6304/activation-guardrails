# Current State

Short project snapshot. Overwrite stale items; do not append history here.

## Now

- Repo reset to a CC++ fresh-start skeleton.
- Latest cleanup commit: `59841bf Reset codebase for CC++ fresh start`.
- Safety tag for old work: `pre-ccpp-fresh-start`.
- Phase 1 scaffold exists from the earlier WildJailbreak/Gemma direction:
  response-cache generation, activation-cache contracts, and the first TF-IDF
  logistic text baseline now exist. A dense prompt-final logistic probe scaffold
  now consumes activation-cache NPZ/index artifacts and writes model, scores,
  metrics, and table outputs with provenance metadata.
- Supervisor direction on 2026-05-25 changed the immediate priority: first do a
  faithful CC++ paper reproduction, then adapt to WildJailbreak/other datasets.
- Proposal plan in `msc-writeup/ipp/proposal.tex` is the source of truth, but
  use `docs/PROJECT_PLAN.md` first unless exact proposal wording is needed.

## Active Focus

Current implementation focus is no longer expanding the WildJailbreak/Gemma
debug pipeline. The next step is a reproduction-spec audit of the CC++ paper,
then implementing the closest faithful pipeline shape before adding other
datasets.

Next implementation target after this scaffold:

1. Build a CC++ reproduction matrix: paper component, exact setting, local
   availability, substitute if needed, and faithfulness impact.
2. Identify the first paper table/figure to reproduce and write a small config
   for that target.
3. Reuse the existing cache/probe infrastructure where it matches CC++; defer
   WildJailbreak-specific extensions until the reproduction baseline is clear.

## Decisions So Far

- Use `main` as stable-ish solo working history.
- Use short-lived branches only for substantial experiment/code chunks.
- Keep `docs/research_log.md` local and ignored.
- Track concise scaffold docs; ignore large/private/generated context.
- Full Gemma response generation is not required yet and remains explicitly
  gated.
- Dense prompt-final probe trains/selects thresholds on vanilla splits only and
  evaluates adversarial transfer at the frozen validation threshold.
- WildJailbreak/Gemma work is now adaptation infrastructure, not the immediate
  Phase 1 acceptance target.

## Latest Local Checks

- Mock end-to-end cache/probe path ran locally on all 8000 normalized examples
  using generated outputs under `/private/tmp/agguardrails_dense_probe_mock_e2e`.
- Mock dense probe table used threshold `0.5412154772012588`; because the
  activations are deterministic schema checks rather than model features, the
  near-chance metrics are not reportable experiment evidence.

## Open Checks Before Experiments

- Confirm local/cluster versions for `transformers`, `datasets`, `torch`, and
  `sae-lens`.
- Re-read the CC++ paper and extract exact reproducibility requirements before
  more implementation.
- Decide which CC++ components are inaccessible and define explicit substitutes.
- Run tiny smoke tests only after the faithful reproduction target is specified.
