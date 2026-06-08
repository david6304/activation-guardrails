# Workflow

Lightweight workflow for a solo dissertation repo.

## Git

- `main` is stable-ish working history: it should install, tests should pass, and
  docs should not knowingly point to stale workflows.
- Use short-lived branches for substantial chunks such as `ccpp-data`,
  `activation-cache`, `probe-training`, or `sae-interpretability`.
- Work directly on `main` for small docs, typo fixes, config tweaks, and narrow
  test fixes.
- Rebase/merge back quickly; do not keep long-running parallel branches unless
  there is a genuine risky fork.

Commit when a future reader would want a recovery point:

- scaffold or workflow decisions are made
- a pipeline stage runs end to end
- a data/config/schema contract changes
- a result becomes reportable or changes direction
- a bug fix changes metrics or artifact interpretation

Commit messages:

- imperative, specific, and short
- examples: `Add CC++ workflow scaffold`, `Build WildJailbreak split config`,
  `Fix fixed-FPR threshold selection`

Tag sparingly:

- before major rewrites or cleanup: `pre-<change>`
- after a frozen reportable run: `run-<phase>-<date>` or
  `submission-<milestone>`
- never tag every routine experiment.

## Experiment Workflow

1. State the hypothesis, dataset, metric, and threshold rule before coding.
2. Put run choices in `configs/`, not only command-line history.
3. Make expensive stages cacheable: data, responses, judging, activations, SAE
   features, probes, reports.
4. Save metadata beside every result: git commit, config path, seed, model id and
   revision, dataset id and revision, package versions, hardware, timestamp.
5. Treat pilot/debug runs as disposable. Treat final/reportable runs as immutable.
6. If a result is surprising, diagnose labels/splits/thresholds/artifacts before
   adding new experiment branches.

## Logging

Use `docs/research_log.md` for "what we did and why". It is local and ignored.

Log:

- result changes the project direction
- non-obvious design decision
- surprising finding and diagnosis
- supervisor question or answer
- final/reportable run summary

Do not log:

- routine implementation details
- failed commands with no research consequence
- every cluster submission attempt

## Dependencies And APIs

- `requirements.txt` gives a flexible working range, not a final lockfile.
- For fast-moving packages (`transformers`, `datasets`, `torch`, `sae-lens`),
  verify current behaviour from the local environment or official docs/source
  before implementing against memory.
- For final experiments, write an environment snapshot to the run metadata, for
  example `python -m pip freeze`, CUDA/PyTorch versions, and key package versions.
- Pin exact package versions only when reproducing a final run, debugging a
  version-specific issue, or preparing the dissertation artifact.

## Results

- Keep large artifacts and generated results out of git by default.
- Commit only small, final tables/plots if they are directly used in the write-up.
- Do not overwrite reportable metrics silently; write a new run directory or
  include a clear run id.
