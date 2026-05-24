# Current State

Short project snapshot. Overwrite stale items; do not append history here.

## Now

- Repo reset to a CC++ fresh-start skeleton.
- Latest cleanup commit: `59841bf Reset codebase for CC++ fresh start`.
- Safety tag for old work: `pre-ccpp-fresh-start`.
- Old code/results are intentionally out of the active tree.
- Proposal plan in `msc-writeup/ipp/proposal.tex` is the source of truth, but
  use `docs/PROJECT_PLAN.md` first unless exact proposal wording is needed.

## Active Focus

Ready to start the CC++ replication implementation. The next coding task should
be the data/config/metadata contract, not model loading.

Next implementation target after this scaffold:

1. Build the WildJailbreak data/config contract.
2. Implement cacheable CC++-style stages for Gemma 2 9B IT.
3. Add tests around data schema, split construction, metrics, and metadata.

## Decisions So Far

- Use `main` as stable-ish solo working history.
- Use short-lived branches only for substantial experiment/code chunks.
- Keep `docs/research_log.md` local and ignored.
- Track concise scaffold docs; ignore large/private/generated context.

## Open Checks Before Experiments

- Confirm local/cluster versions for `transformers`, `datasets`, `torch`, and
  `sae-lens`.
- Verify WildJailbreak access and exact Hugging Face dataset revision.
- Decide first config names and artifact metadata schema.
