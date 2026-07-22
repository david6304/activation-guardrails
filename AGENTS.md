# Agent Instructions

## Project

- Solo MSc research project, not production software, a shared library, or a team
  codebase. Optimise for fast iteration and reliable evidence over polish.
- Submission is due in just over four weeks (from 2026-07-22), and the project
  does not yet have its definitive primary result. Until that result exists,
  prioritise the shortest scientifically valid path to evidence over extensions,
  robustness machinery, or polished infrastructure.
- When a simple approach and a more comprehensive approach are both valid, start
  with the simple one. Record worthwhile extensions for later rather than making
  them prerequisites. Prefer an explicit limitation in the dissertation to an
  extra experiment or abstraction unless the limitation would invalidate the
  main claim.
- Keep the immediate critical path visible: make one bounded change, run the
  smallest meaningful check, then obtain the result. Do not let optional
  comparisons, sensitivity analyses, or future phases delay the primary run.
- Read `README.md` first — it documents the folder structure. Use
  `RESEARCH_LOG.md` for recent decisions and `papers/notes/lit-review-index.md`
  before opening paper notes or PDFs. For papers: read the note for orientation;
  read the PDF only when you need exact formulas or hyperparameters.
- `msc-writeup/` is a separate git repo. `archive/` is historical, not current
  state.
- `msc-writeup/ipp/proposal.tex` sets the broad direction; newer `RESEARCH_LOG.md`
  entries override its detailed decisions.
- Local Python environment: `msc-diss` (Python 3.12). Run commands as
  `conda run -n msc-diss ...` so checks do not depend on the caller's activated
  shell.
- For changed Python files, run `conda run -n msc-diss ruff check <files>` and
  `conda run -n msc-diss python -m py_compile <files>`. For an entrypoint, also
  inspect `conda run -n msc-diss python <script>.py --help`.
- There is no default full test suite. Validate the smallest real boundary the
  change affects. Treat `run_*.sh` as cluster launchers and read
  `docs/CLUSTER.md` before running or editing them.

## Code

- Default to one flat, readable script with plain functions and explicit
  arguments. Justify any structure beyond that before adding it — no package or
  class hierarchies, generic pipelines, CLIs, config systems, registries,
  trackers, services, or CI unless the current task needs it.
- Write the minimum for the current experiment or present data. Duplication beats
  premature abstraction; a one-off may stay one-off. Implement no speculative
  cases, future datasets, error handling, or production concerns
  (backwards-compat, exhaustive validation, observability, packaging).
- Prefer editing an existing script over creating a new one with the same
  purpose. Do not refactor adjacent code as part of a bounded change.
- Validate at real boundaries (dataset rows, saved artifacts, model outputs,
  external APIs), not impossible internal states.
- Prefer the stdlib and existing dependencies; ask before adding a heavy one.
- Verify unfamiliar package, model, dataset, or helper interfaces from the
  installed source before use — do not invent plausible-looking APIs. Reuse the
  accessor the repo already uses (grep for it) instead of re-deriving it; a CPU
  config/shape check (e.g. `AutoConfig`) confirms model interfaces with no GPU.

## Research integrity

- The user approves scientific choices: datasets, labels, splits, metrics,
  models, thresholds, and claims.
- State material assumptions that affect design or interpretation; do not
  silently pick between plausible interpretations.
- Work one bounded task at a time. Diagnose surprising results before expanding
  scope. Distinguish exploratory work from dissertation experiments.
- Keep one source of truth for data, splits, labels, thresholds, and metrics;
  keep a snapshot only when reproducibility or historical comparison requires it.
- For dissertation experiments, record what reproduces the result: input or
  manifest, model id and revision, parameters, seed, command, and output
  location.
- Record only consequential decisions, results, and direction changes in
  `RESEARCH_LOG.md`.

## Evidence

- Tests are not a default. Prefer one small real run with inspected output or an
  assertion over a suite. No coverage targets, no hypothetical-edge-case tests.
- Add a focused test only when logic is easy to get subtly wrong and an error
  would invalidate results, or after a regression.
- Do not treat mocked or synthetic runs as evidence that a real model, dataset,
  package, or cluster integration works. A fake must not stub the interface under
  test; before a full cluster submit, run a tiny `--limit` real-model smoke.

## Review

- For code-review requests, read `CODE_REVIEW.md`. Report correctness and
  research-validity findings first, ordered by severity, with file and line
  references. Do not spend the review on style unless it obscures a result.

## Cluster

- Read `docs/CLUSTER.md` before any cluster script, resource request, or
  submission command.
- The user owns the cluster and approves *what* work runs on it. Once the user
  has authorised a specific piece of work, carry it out end-to-end without a
  fresh prompt per command: connect over SSH, inspect state, `sbatch
  --test-only`, submit that job, and cancel or resubmit it. Read-only inspection
  (`squeue`, `sinfo`, `scontrol show`, `sacct`, `ls`, `wc`, `cat`, `tail`) never
  needs authorisation. Ask first before: a large or long job (multi-GPU, >4 h
  wall-clock, or writing >100 GB), deleting or overwriting existing data or
  artifacts, cancelling a job you did not submit, copying files off the cluster,
  or when the target node or command is ambiguous.
- Check locally first (imports, args, paths, config, dataset shape, a tiny CPU
  path, output creation); use the cluster only for what genuinely needs CUDA.
  Say when a check needs cluster resources and why (CUDA, VRAM, runtime,
  storage).
- Tune batch size with a short measured pilot for the exact model, precision, and
  sequence length. Before a real run, check live GPU and queue state and
  `sbatch --test-only` the eligible types, then submit the earliest-start one.

## When done

- Stop when the question is answered or the artifact works and passed the
  smallest meaningful check. No unsolicited cleanup or generalisation.
- Review the final diff and remove abstractions, comments, defensive checks, and
  formatting churn the change introduced. When complexity is questioned, simplify
  before defending it.
