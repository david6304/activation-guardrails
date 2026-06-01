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
    `src/agguardrails/metrics.py`;
  - activation cache contracts and mock/real extraction entrypoint live in
    `src/agguardrails/activations.py` and
    `scripts/ccpp/extract_streaming_activations.py`;
  - a minimal linear SWiM probe trainer lives in
    `src/agguardrails/swim_probe.py` and `scripts/ccpp/train_swim_probe.py`.
  - `scripts/ccpp/build_generation_prompts.py` builds a HarmBench prompt-only
    manifest for the generated-completion substitute path.
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
3. Run the mock activation/probe vertical slice end to end, then replace mock
   activations with Gemma2 activations once a gated dataset exists.

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

- 2026-06-01: `python -m pytest` passes with 30 tests.
- 2026-06-01: `python -m ruff check src/agguardrails scripts/ccpp tests`
  passes.
- 2026-06-01: after Hugging Face access was granted,
  `scripts/ccpp/build_generation_prompts.py --limit 3` successfully built a
  local HarmBench prompt-only manifest under `/tmp`.

## Open Checks Before Experiments

- Confirm local/cluster versions for `transformers`, `datasets`, `torch`, and
  `sae-lens`.
- Inspect public dataset schemas and decide whether public harmful/compliant
  assistant completions are adequate.
  - `AlignmentResearch/ClearHarm` is reachable, but inspected
    `proxy_gen_target` values look like short prefills and are marked
    `positive_prefill_only`, not accepted as full positive completions.
  - `walledai/HarmBench` is accessible after Hugging Face approval, but the
    inspected configs (`standard`, `contextual`, `copyright`) expose prompts and
    categories/tags, not assistant completions.
  - `allenai/WildChat` is reachable and has conversation-level moderation
    fields usable for benign hard-negative candidates.
- The likely next path is a controlled `generated_uncensored` completion
  substitute from the HarmBench prompt manifest, with raw harmful outputs kept
  local and out of logs.
- Run tiny activation/probe smoke tests only after the normalized dataset passes
  gates.
