# Workflow

Lightweight Codex, Claude Code, and experiment workflow for a solo dissertation
repository. `AGENTS.md` is the canonical instruction entrypoint for both tools.

## Interaction Cycle

Use this cycle for one milestone at a time:

1. **Understand:** inspect the routed documentation and relevant existing code,
   then restate the objective, current behaviour, assumptions, and unknowns.
2. **Plan:** define scope, exclusions, expected files, acceptance evidence,
   scientific risks, and stop conditions.
3. **Approve:** obtain explicit user approval before implementing a planned
   milestone. Scientific choices always require user approval.
4. **Implement:** make the smallest coherent approved change without adding
   future-stage infrastructure.
5. **Verify:** run mechanical checks and state separately what they do not
   establish scientifically.
6. **Explain:** walk through the diff, data flow, assumptions, and failure
   modes.
7. **Review:** inspect the complete diff against the approved milestone.
8. **Commit:** commit only after the user understands and accepts the change.

Explanation and planning requests are read-only unless the user explicitly
authorizes edits. Approval of a plan authorizes only its stated scope.

## One Milestone At A Time

Before editing, define one bounded milestone:

- objective and claim, if scientific;
- files or components in scope;
- explicit exclusions;
- expected files and approximate diff size;
- observable acceptance criteria;
- verification commands;
- whether the work is exploratory or reportable.

Finish or deliberately stop that milestone before starting another. Do not add
future-stage infrastructure, speculative extension hooks, generalized
frameworks, or unrelated cleanup.

## Default Scope Limits

For a normal implementation milestone, expect:

- one to three implementation files plus directly associated tests or config;
- no more than about five changed files in total;
- roughly 50 to 250 substantive changed lines.

These are reviewability guidelines, not quality measures. A cohesive mechanical
rename, generated metadata, or an inseparable schema migration may be larger.
Identify and justify such an exception before continuing.

Stop and replan before exceeding about six files or 300 to 400 substantive
lines, or when the work would introduce a dependency, add a new abstraction,
cross a pipeline stage, or change an empirical contract. Separate generated or
purely mechanical churn from substantive diff size.

Within the approved scope:

- change only files required by the milestone;
- preserve existing interfaces and artifact formats;
- use the repository's current patterns;
- avoid new dependencies;
- search for existing helpers, schemas, and boundaries before adding new ones;
- avoid abstractions used by only one concrete path;
- do not implement later research phases;
- do not change scientific choices while fixing engineering defects;
- do not turn a local script into a platform.

If another issue is discovered, report it rather than fixing it opportunistically.

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

## Understanding Gate

Before accepting a milestone, the user should be able to:

- state the input and output contract;
- trace one representative example through the changed path;
- identify each scientific choice and where it is encoded;
- explain what the tests and checks do not prove;
- name the commit or clean state to which the work can be restored.

If any item is unclear, explain a smaller unit or revise the implementation
before acceptance. Passing tests alone is not enough.

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
criteria, and deferred work. Obtain explicit user approval before editing.

## Review Modes

### Self-Review

Default for low-risk, bounded changes. Review the final diff against the
milestone, check failure paths, and run targeted verification.

### Independent Engineering Review

Use for broad changes, shared contracts, cache/provenance logic, cluster-facing
jobs, or code whose failure could waste a substantial run. It is one read-only
review of a frozen diff and acceptance evidence in a fresh agent chat or by a
non-author human. It is not required for routine, low-risk work covered by
self-review.

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
3. Obtain approval for the stated milestone.
4. Implement the smallest coherent change.
5. Run targeted tests during development.
6. Review the final diff for scope and scientific-contract drift.
7. Run the milestone's acceptance commands.
8. Run broader pytest and Ruff checks when shared behaviour or reportable code
   changes.
9. Explain the diff and apply the understanding gate.
10. Record limitations, blocked checks, and dated verification results.

For GPU work, validate imports, argument parsing, config loading, output paths,
dry-run behaviour, and CPU-safe tests locally before cluster submission.
Remote access and job submission require explicit authorization for the target
host, resources, and action.

## Recovery And Context

If an agent takes the wrong direction:

1. stop editing;
2. report changed files, unapproved decisions, and any independently useful
   work;
3. inspect the diff without discarding unrelated user changes;
4. propose either a narrow salvage or restoration to the last accepted commit;
5. obtain approval before applying the recovery.

Never use destructive Git commands to recover an agent change unless the user
explicitly approves the exact operation. Claude checkpoints can help rewind
Claude-made edits, but Git remains the repository recovery record.

Start a fresh chat when changing to an unrelated milestone, when the current
context contains abandoned assumptions, or for an independent read-only review.
Keep the current chat when implementing an approved plan whose context is still
accurate.

## Prompt Templates

### Explain Unfamiliar Code

```text
Read [paths] and the routed documentation. Do not edit. Explain the purpose,
call and data flow, scientific assumptions, failure modes, and tests. Identify
anything you cannot infer.
```

### Plan One Issue

```text
Plan one issue without editing. State the objective, exclusions, expected
files, estimated diff size, acceptance evidence, scientific decisions requiring
my approval, and conditions that would make you stop and replan.
```

### Implement An Approved Issue

```text
Implement only the approved milestone below. Inspect relevant code and the
nearest folder README first. Do not broaden scope or change scientific choices.
Stop if the approved limits or assumptions are exceeded.

[approved milestone]
```

### Review A Diff

```text
Review this frozen diff without editing. Check correctness, scope drift,
scientific-contract drift, leakage, failure handling, and missing tests.
Distinguish mechanical verification from scientific validation.
```

### Diagnose A Surprising Result

```text
Do not extend the pipeline. Diagnose data integrity, labels, groups and splits,
leakage, thresholds, shortcuts, cache provenance, and metric implementation.
Rank hypotheses and propose the smallest discriminating checks.
```

### Stop And Recover

```text
Stop editing. Report changed files, unapproved decisions, and usable work. Do
not discard user changes. Propose either a narrow salvage diff or safe
restoration to the last accepted commit.
```

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
