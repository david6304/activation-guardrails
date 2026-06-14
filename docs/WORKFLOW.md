# Workflow

Lightweight Codex, Claude Code, and experiment workflow for a solo dissertation
repository. `AGENTS.md` is the canonical instruction entrypoint for both tools.

## Default Approach

This is a solo dissertation repository. The default goal is a small experiment
that works and can be understood, not production-quality software.

For one milestone:

1. Inspect only the relevant context.
2. Implement the simplest direct path.
3. Run the smallest real example available.
4. Inspect the output or failure.
5. Add only the hardening justified by that evidence.

Obtain explicit approval for scientific choices. Routine engineering details do
not need a separate planning ceremony when the request already authorizes the
work.

## One Milestone At A Time

Before editing, define one bounded milestone:

- objective and claim, if scientific;
- the smallest observable result that would show progress;
- whether the work is exploratory or reportable.

Finish or deliberately stop that milestone before starting another. Do not add
future-stage infrastructure, speculative extension hooks, generalized
frameworks, or unrelated cleanup.

## Default Scope Limits

For exploratory work, prefer one direct script and tens or low hundreds of
lines. This is a direction, not a quota. If the implementation grows because of
abstractions, generalized validation, metadata machinery, or tests rather than
the experiment itself, simplify it.

Within the approved scope:

- change only files required by the milestone;
- use the repository's current patterns;
- avoid new dependencies;
- use real small inputs and the exact external dependency when practical;
- avoid abstractions used by only one concrete path;
- do not implement later research phases;
- do not change scientific choices while fixing engineering defects;
- do not turn a local script into a platform.

If another issue is discovered, report it rather than fixing it opportunistically.

## Observable Acceptance

Acceptance criteria must describe inspectable outcomes, not intentions.
Depending on the milestone, include:

- exact command and expected exit status;
- one real example completing successfully;
- a small output that can be inspected directly;
- a metric calculation on a known example when relevant.

For model, dataset, package, or cluster integration, mocked tests do not count
as acceptance. The exact real path must run at least once. For exploratory work,
that real micro-run may be sufficient without automated tests.

## Understanding Gate

For consequential scientific work, the user should be able to:

- state the input and output contract;
- trace one representative example through the changed path;
- identify each scientific choice and where it is encoded.

Exploratory engineering does not require a formal acceptance ceremony. Explain
the important behavior and limitations briefly.

## Exploratory And Reportable Experiments

Classify a run before execution.

### Exploratory

Use exploratory runs to debug feasibility, schemas, memory, runtime, or obvious
confounds.

- Small samples and provisional choices are allowed.
- Prefer direct scripts and visible failures over infrastructure.
- Start with one synthetic example, then one real example, then a handful.
- Do not add resume, retry, generalized provenance, or broad test coverage until
  the basic path works and there is an observed need.
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

Reserve this for broad shared changes or expensive reportable runs. It is not a
default requirement for exploratory scripts or ordinary cluster jobs.

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

1. Inspect the directly relevant code and guidance.
2. Implement the smallest coherent change.
3. Run the exact acceptance command on the smallest useful input.
4. Inspect the result before extending the implementation.
5. Run focused tests only where they add useful confidence.
6. Use broader tests and provenance checks for shared or reportable work.

For GPU work, prioritize the real integration sequence: load the exact model,
generate one token, process one real example, then process a handful. Only then
submit the intended smoke run or add operational hardening.
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
Plan the smallest useful milestone without editing. State the real command or
output that would show it works and identify any scientific choice requiring my
approval. Avoid proposing infrastructure or tests without a concrete need.
```

### Implement An Approved Issue

```text
Implement the smallest direct version of the milestone below. Prefer a simple
script and a real micro-run. Do not add abstractions, broad tests, retries,
resumability, or generalized metadata unless this milestone already needs them.
Do not change scientific choices.

[approved milestone]
```

### Review A Diff

```text
Review this frozen diff without editing. Check correctness, scope drift,
scientific-contract drift, leakage, and whether the implementation is more
complex than the task requires. Request tests only for a concrete risk.
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
