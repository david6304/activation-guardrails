# Workflow

Lightweight Codex and experiment workflow for a solo dissertation repository.

## One Milestone At A Time

Before editing, define one bounded milestone:

- objective and claim, if scientific;
- files or components in scope;
- explicit exclusions;
- observable acceptance criteria;
- verification commands;
- whether the work is exploratory or reportable.

Finish or deliberately stop that milestone before starting another. Do not add
future-stage infrastructure, speculative extension hooks, generalized
frameworks, or unrelated cleanup.

## Default Scope Limits

Unless the task explicitly broadens scope:

- change only files required by the milestone;
- preserve existing interfaces and artifact formats;
- use the repository's current patterns;
- avoid new dependencies;
- avoid abstractions used by only one concrete path;
- do not implement later research phases;
- do not change scientific choices while fixing engineering defects;
- do not turn a local script into a platform.

If necessary work exceeds these limits, stop and revise the milestone before
continuing.

## Observable Acceptance

Acceptance criteria must describe inspectable outcomes, not intentions.
Depending on the milestone, include:

- exact command and expected exit status;
- created artifact, schema, or metadata fields;
- deterministic fixture or invariant;
- metric calculation on a known example;
- cache/resume behaviour;
- absence of changes outside the declared scope.

Tests support acceptance but do not replace scientific validation.

## Exploratory And Reportable Experiments

Classify a run before execution.

### Exploratory

Use exploratory runs to debug feasibility, schemas, memory, runtime, or obvious
confounds.

- Small samples and provisional choices are allowed.
- Outputs must be labelled non-reportable.
- Test or final-evaluation data must not influence model or threshold choices.
- Exploratory success does not authorize a scientific claim.
- Keep only the metadata needed to reproduce a consequential finding.

### Reportable

A reportable run requires a frozen protocol and immutable run identity.

- Commit code and config before execution.
- Pin model, tokenizer, dataset, and judge revisions where applicable.
- Record seeds, commands, environment, hardware, timestamps, and artifact
  checksums.
- Separate fit, selection, threshold calibration, and final evaluation.
- Keep final evaluation sealed until choices are frozen.
- Write a new run directory; never silently overwrite reportable outputs.
- Record confidence intervals or repeated seeds when the claim requires them.
- State substitutions and deviations from the target method.

## Planning Modes

Use the least process that matches the risk.

### Direct Mode

Use for narrow fixes, documentation, tests, or implementation of an already
frozen contract. Define the milestone, inspect the relevant code, implement,
self-review, and verify.

### Plan-First Mode

Use before edits when work changes:

- dataset, model, label, split, metric, threshold, or baseline;
- a shared artifact or schema contract;
- a reportable experiment protocol;
- several modules or pipeline stages;
- the interpretation of an existing result.

The plan must state alternatives considered, scientific risks, acceptance
criteria, and deferred work. Obtain user approval when requested or when the
choice changes the project's empirical direction.

## Review Modes

### Self-Review

Default for low-risk, bounded changes. Review the final diff against the
milestone, check failure paths, and run targeted verification.

### Independent Engineering Review

Use for broad changes, shared contracts, cache/provenance logic, cluster-facing
jobs, or code whose failure could waste a substantial run. It is one read-only
review of a frozen diff and acceptance evidence by a fresh Codex
context/subagent or a non-author human. It is not required for routine,
low-risk work covered by self-review.

### Scientific-Risk Review

Require independent review, supervisor discussion, or explicit user approval
before treating work as reportable when it changes:

- the estimand, labels, or unit of analysis;
- grouping, leakage controls, or split roles;
- baseline strength or comparison fairness;
- metric, threshold rule, or headline operating point;
- model/data substitutions affecting faithfulness;
- exclusion rules, judging, or result interpretation;
- a surprising result that changes project direction.

Engineering review may verify implementation of these choices but cannot
validate the scientific choice itself.

## Implementation And Verification

1. Inspect the relevant code, config, tests, and nearest folder README.
2. State the bounded milestone and acceptance criteria.
3. Implement the smallest coherent change.
4. Run targeted tests during development.
5. Review the final diff for scope and scientific-contract drift.
6. Run the milestone's acceptance commands.
7. Run broader pytest and Ruff checks when shared behaviour or reportable code
   changes.
8. Record limitations, blocked checks, and dated verification results.

For GPU work, validate imports, argument parsing, config loading, output paths,
dry-run behaviour, and CPU-safe tests locally before cluster submission.

## Git

- Keep `main` in a known, reviewable state.
- Use a short-lived branch for a substantial milestone or risky fork.
- Commit when a future reader needs a recovery point: frozen protocol, schema
  change, completed stage, direction-changing result, or metric-affecting fix.
- Use imperative, specific commit messages.
- Tag only major pre-rewrite recovery points or frozen reportable runs.
- Never commit secrets, gated data, large caches, or generated harmful content.

## Reproducibility

Every reportable result must include or reference:

- git commit and dirty-tree status;
- config path and content hash;
- seed and exact command;
- model, tokenizer, dataset, and judge identities and revisions;
- split/grouping and threshold rules;
- package, CUDA, driver, and hardware details;
- artifact lineage and checksums;
- timestamp and exploratory/reportable classification.

`requirements.txt` contains working ranges, not a final lockfile. Capture the
exact environment for reportable runs.

## Results And Logging

- Keep large artifacts and debug results out of Git.
- Commit only compact outputs directly required for a reported claim.
- Diagnose labels, splits, thresholds, text shortcuts, and artifact integrity
  before extending a surprising result.
- Summarize consequential scientific decisions and direction-changing
  conclusions in tracked authoritative documentation.
- Ask before adding supporting detail to the ignored `docs/research_log.md`;
  it must never be the sole record of a consequential decision or conclusion.
