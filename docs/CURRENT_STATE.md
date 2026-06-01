# Current State

Short project snapshot. Overwrite stale items; do not append history here.

## Now

- Repo reset to a CC++ fresh-start skeleton.
- Latest cleanup commit: `59841bf Reset codebase for CC++ fresh start`.
- Safety tag for old work: `pre-ccpp-fresh-start`.
- Old WildJailbreak/Gemma implementation code has been removed from the active
  tree as part of the fresh-start plan.
- CC++ reproduction matrix exists at `docs/ccpp_reproduction_matrix.md`.
- Phase 2/3 scaffolding has started:
  - normalized CC++ exchange schema and dataset gates live in
    `src/agguardrails/ccpp_data.py`;
  - `scripts/ccpp/build_dataset.py` builds from curated local JSONL and blocks
    if harmful/compliant assistant completions have not been confirmed;
  - metric contracts for ROC-AUC, fixed-FPR thresholding, frozen-threshold
    evaluation, log-space low-FPR AUC, and flag-at-any-token scoring live in
    `src/agguardrails/metrics.py`.
- Supervisor direction on 2026-05-25 changed the immediate priority: first do a
  faithful CC++ paper reproduction, then adapt to WildJailbreak/other datasets.
- Proposal plan in `msc-writeup/ipp/proposal.tex` is the source of truth, but
  use `docs/PROJECT_PLAN.md` first unless exact proposal wording is needed.

## Active Focus

Current implementation focus is no longer expanding the WildJailbreak/Gemma
debug pipeline. The reproduction-spec audit has a first-pass matrix; the next
step is to confirm/curate a public positive-completion dataset and then run the
probe-only vertical slice.

Next implementation target after this scaffold:

1. Inspect candidate public sources (`AlignmentResearch/ClearHarm`, HarmBench,
   and WildChat) for harmful/compliant assistant completions and matched benign
   CBRN/science-adjacent negatives.
2. Create or curate `data/processed/ccpp/public_cbrn_exchanges.jsonl` only
   after the positive-completion gate is satisfied.
3. Implement activation extraction and SWiM probe training against the
   normalized schema.

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

- 2026-06-01: `python -m pytest` passes with 12 tests.
- 2026-06-01: `python -m ruff check src/agguardrails scripts/ccpp tests`
  passes.

## Open Checks Before Experiments

- Confirm local/cluster versions for `transformers`, `datasets`, `torch`, and
  `sae-lens`.
- Inspect public dataset schemas and decide whether public harmful/compliant
  assistant completions are adequate.
- If no adequate public completions exist, define the controlled generated
  positive-completion substitute before training any probe.
- Run tiny activation/probe smoke tests only after the normalized dataset passes
  gates.
