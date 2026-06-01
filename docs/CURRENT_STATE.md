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
  - generated-completion datasets now have stricter gates for single-generator
    on-policy metadata, assistant-length balance, and text-only separability.
  - `docs/matched_benign_prompt_spec.md` defines the dual-use CBRN benign prompt
    design for the primary generated-completion dataset.
  - `scripts/ccpp/build_matched_benign_prompts.py` assembles the matched
    dual-use benign prompt manifest (prompt-only) from the curated seed library
    in `src/agguardrails/ccpp_benign.py`, writing JSONL + metadata under
    `data/interim/ccpp/`. It enforces unique prompt IDs, a harmful-intent
    tripwire, and a near-duplicate diversity check; metadata carries
    domain/intent histograms and unique-group counts.
- Supervisor direction on 2026-05-25 changed the immediate priority: first do a
  faithful CC++ paper reproduction, then adapt to WildJailbreak/other datasets.
- Proposal plan in `msc-writeup/ipp/proposal.tex` is the source of truth, but
  use `docs/PROJECT_PLAN.md` first unless exact proposal wording is needed.

## Active Focus

Current implementation focus is no longer expanding the WildJailbreak/Gemma
debug pipeline. The reproduction-spec audit has a first-pass matrix; the next
step is to create a matched-generator generated dataset that avoids off-policy,
generator-style, topic, and length confounds before running the probe-only
vertical slice.

Next implementation target after this scaffold:

1. Build ClearHarm CBRN harmful prompts as the primary scaled source, grouped by
   underlying prompt hash, plus matched dual-use benign CBRN/science-adjacent
   prompts. Effective positive N is unique groups, currently 179, not the 7160
   `rep40` rows. Keep HarmBench `chemical_biological` as a small
   supplementary/smoke slice.
2. Pick one refusal-ablated Gemma generator/protected-model analogue after a
   Heretic/OBLITERATUS bakeoff.
3. Generate both positive and benign completions with that same model and
   decoding setup, recording `generator_model_id == protected_model_id`.
4. Run dataset gates plus TF-IDF text separability before activation extraction.
5. Run the mock activation/probe vertical slice end to end, then replace mock
   activations with the chosen ablated Gemma activations once the dataset passes.

## Decisions So Far

- Use `main` as stable-ish solo working history.
- Use short-lived branches only for substantial experiment/code chunks.
- Keep `docs/research_log.md` local and ignored.
- Track concise scaffold docs; ignore large/private/generated context.
- Full Gemma response generation is not required yet and remains explicitly
  gated.
- Normal safety-trained Gemma is not the primary protected model for generated
  harmful-completion experiments unless on-policy jailbreak-elicited positives
  are available. The primary generated path uses the refusal-ablated generator
  itself as the protected-model analogue.
- Dense prompt-final probe trains/selects thresholds on vanilla splits only and
  evaluates adversarial transfer at the frozen validation threshold.
- WildJailbreak/Gemma work is now adaptation infrastructure, not the immediate
  Phase 1 acceptance target.

## Latest Local Checks

- 2026-06-01: `python -m pytest` passes with 53 tests.
- 2026-06-01: `build_matched_benign_prompts.py` assembles 1000 unique benign
  groups across the 7 matched subdomains with balanced topic/intent histograms
  and zero near-duplicates (Jaccard threshold 0.9).
- 2026-06-01: `python -m ruff check src/agguardrails scripts/ccpp tests`
  passes.
- 2026-06-01: after Hugging Face access was granted,
  `scripts/ccpp/build_generation_prompts.py --limit 3` successfully built a
  local HarmBench prompt-only manifest under `/tmp`.
- 2026-06-01: full HarmBench `standard` prompt manifest filtering keeps 28
  `chemical_biological` rows out of 200 total rows.
- 2026-06-01: ClearHarm `rep40` exposes 7160 positive rows over 179 unique
  prompt contents; use this as the primary scaled prompt source with grouped
  splits.
- 2026-06-01: reportable metrics must be group-level. The benign prompt target
  is now oversized for the low-FPR axis: about 1000 independent benign groups,
  with 300 as the minimum rough TPR@1%FPR denominator.
- 2026-06-01: training class ratio is decoupled from the oversized benign
  evaluation pool. The current probe config samples training negatives toward a
  maximum 3:1 negative:positive ratio, while evaluating on the full group-level
  pool. Fixed-FPR outputs include Wilson TPR/FPR intervals.

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
  - `AlignmentResearch/ClearHarm` `rep40` is accessible and large enough for the
    primary prompt manifest, but still requires generated completions.
  - `allenai/WildChat` is reachable and has conversation-level moderation
    fields usable for benign hard-negative candidates.
- The likely next path is a controlled `generated_uncensored` completion
  substitute from the HarmBench prompt manifest, with raw harmful outputs kept
  local and out of logs. This must use matched benign prompts and the same
  generator for both labels; WildChat is secondary reference data only.
- Text-only ROC-AUC is a required marginal-value control. The current
  configured design-smell threshold is `0.95`; above that, harden the benign
  prompt set rather than proceeding to activation extraction.
- Run tiny activation/probe smoke tests only after the normalized dataset passes
  gates.
