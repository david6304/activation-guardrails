# Current State

Short project snapshot. Overwrite stale items; do not append history here.

## Now

- Repo reset to a CC++ fresh-start skeleton.
- Latest cleanup commit: `59841bf Reset codebase for CC++ fresh start`.
- Safety tag for old work: `pre-ccpp-fresh-start`.
- Phase 1 cacheable scaffold is underway: WildJailbreak normalization,
  response-cache generation, and activation-cache contracts now exist.
- Proposal plan in `msc-writeup/ipp/proposal.tex` is the source of truth, but
  use `docs/PROJECT_PLAN.md` first unless exact proposal wording is needed.

## Active Focus

Current implementation focus is the small debug pipeline before any full Gemma
run: data contract, response cache, mock activation cache, then cheap baselines.

Next implementation target after this scaffold:

1. Add the first cheap text baseline/result-table scaffold.
2. Add real activation extraction behind an explicit gate, after a tiny Gemma
   response-cache smoke test is approved.
3. Keep tests around data schema, split construction, metrics, artifacts, and
   metadata.

## Decisions So Far

- Use `main` as stable-ish solo working history.
- Use short-lived branches only for substantial experiment/code chunks.
- Keep `docs/research_log.md` local and ignored.
- Track concise scaffold docs; ignore large/private/generated context.
- Full Gemma response generation is not required yet and remains explicitly
  gated.

## Open Checks Before Experiments

- Confirm local/cluster versions for `transformers`, `datasets`, `torch`, and
  `sae-lens`.
- Verify WildJailbreak access and exact Hugging Face dataset revision.
- Run a tiny approved real-generation smoke test before any full response cache.
