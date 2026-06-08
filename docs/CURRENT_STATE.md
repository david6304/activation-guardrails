# Current State

Short project snapshot. Overwrite stale items; do not append history here.

## Now

- The repository contains a fresh CC++ implementation scaffold.
- Safety tag for the removed earlier pipeline: `pre-ccpp-fresh-start`.
- Current HEAD at planning time: `203a18f`.
- The generated ClearHarm CBRN positives and bespoke matched-benign negatives
  proved almost perfectly text-separable. That dataset is not suitable for the
  next activation experiment.
- The current implementation target is the reviewed WildJailbreak open-weight
  CC++ analogue in `docs/WILDJAILBREAK_CCPP_IMPLEMENTATION_PLAN.md`.
- The plan passed the double-agent planning workflow with mandatory preflight
  gates.
- Implement one checkpoint at a time. Start with C0 only.

Existing reusable scaffold:

- normalized exchanges and dataset gates in `src/agguardrails/ccpp_data.py`;
- generation helpers in `src/agguardrails/ccpp_generation.py`;
- metrics in `src/agguardrails/metrics.py`;
- activation contracts in `src/agguardrails/activations.py`;
- minimal SWiM code in `src/agguardrails/swim_probe.py`;
- runnable scripts under `scripts/ccpp/`;
- CC++ source mapping in `docs/ccpp_reproduction_matrix.md`.

The existing code is a starting point, not a verified implementation of the
reviewed WildJailbreak protocol. In particular, activation storage, SWiM window
semantics, split roles, exchange labels, and system calibration must change.

## Active Focus

Implement checkpoint C0:

1. Add `configs/ccpp/wildjailbreak_gemma3_4b_exchange.yaml`.
2. Freeze an exchange-harmfulness rubric and judge prompt hash.
3. Add the runtime/protocol validator.
4. Resolve dataset, model, tokenizer, judge, duplicate-detection, PEFT, and
   provenance contracts.
5. Mark every component `faithful`, `substitute`, `our_default`, or
   `pending_evidence`.
6. Commit C0 as a recovery point before constructing manifests or running GPU
   jobs.

Do not begin C1 until C0 tests and preflight gates pass.

## Decisions

- Primary model family: Gemma 3 4B.
- Primary protected-model analogue: the refusal-ablated Gemma artifact, subject
  to C0 identity and weight-hash verification.
- Dataset: pinned `allenai/wildjailbreak`, subject to access, schema, count, and
  pairing checks.
- Primary target: adjudicated exchange harmfulness.
- Secondary target: WildJailbreak source prompt intent.
- Refusal is a diagnostic variable, not the primary label.
- Expensive stages must be cacheable, checksummed, resumable, and provenance
  linked.
- Fit, development selection, final threshold calibration, vanilla test, and
  adversarial transfer use separate group-level partitions.
- Test and transfer remain sealed until models, score transforms, system
  choices, and final threshold are frozen.
- Primary operating point: `TPR @ 1% FPR`.
- `0.1% FPR` is not a headline claim for the planned public-data denominators.
- Reportable inference is group-level and includes multiple training seeds.
- SAEs and public guards are deferred extensions after the dense CC++ system is
  evaluated.

## Latest Verified Checks

- 2026-06-01: `python -m pytest` passed with 60 tests in the then-current
  environment.
- 2026-06-01: `python -m ruff check src/agguardrails scripts/ccpp tests`
  passed.
- 2026-06-02: the generated CBRN dataset passed on-policy and length gates but
  failed the text diagnostic near the ceiling. The user has confirmed this
  finding as the reason to try WildJailbreak first.
- 2026-06-07: the WildJailbreak implementation plan completed the double-agent
  planning, review, revision, verification, and adjudication workflow.

Do not infer that current tests pass today from the historical checks. The local
workstation environment inspected during planning lacked the project ML
packages; C0 must capture the actual implementation environment.

## Open C0 Preflights

- Pin and load the WildJailbreak revision.
- Verify all four data types, pairing semantics, and sufficient unique groups.
- Discover the Gemma runtime model class, dimensions, hidden-state ordering,
  maximum sequence length, and assistant-token boundaries.
- Verify the refusal-ablated artifact identity and weight hash.
- Confirm an independent exchange-harmfulness judge or stop for manual labels.
- Pin and validate the sentence-embedding duplicate stack.
- Verify `torch`, `transformers`, `datasets`, `safetensors`, `peft`, and cluster
  compatibility.
- Verify the exchange-classifier label-scoring strategy.
- Capture package, CUDA, driver, hardware, Git, config, and command provenance.

## Research Log

The CBRN text-separability result and switch to WildJailbreak changed the
research direction. `docs/research_log.md` is local and may contain an
inaccurate or premature WildJailbreak entry. Ask the user before replacing or
correcting that entry.
