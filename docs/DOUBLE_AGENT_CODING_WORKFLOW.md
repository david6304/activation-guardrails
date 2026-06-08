# Budget-Aware Coding And Review Workflow

Use this workflow to implement one or more sections of an already reviewed
plan while controlling usage and preventing complexity or subtle errors from
accumulating.

The default arrangement uses two independent agents:

- the top-level Codex agent is the persistent implementer and orchestrator;
- one fresh subagent reviews the completed batch.

Do not spawn a separate implementation agent by default. The top-level agent
already has the repository context, so spawning another implementer duplicates
the most expensive context and usually adds little independence. The reviewer
provides the independent check.

Each reviewer is told the implementation came from Claude. This is only a
prompting device intended to encourage independent criticism.

## Example Invocations

```text
Use the coding workflow in docs/DOUBLE_AGENT_CODING_WORKFLOW.md.
Implement sections C0 and C1 as one batch and use a lightweight reviewer.
```

```text
Use the coding workflow in docs/DOUBLE_AGENT_CODING_WORKFLOW.md.
Implement C3 and C4 as one batch. Use lightweight review for the batch, but
strong review for the C4 label semantics.
```

```text
Use the coding workflow in docs/DOUBLE_AGENT_CODING_WORKFLOW.md.
Implement the remaining plan. Choose sensible review batches and reserve strong
review for scientifically high-risk code. Keep usage economical.
```

```text
Use the coding workflow in docs/DOUBLE_AGENT_CODING_WORKFLOW.md.
Implement this narrow change with self-review only.
```

The workflow is reusable. Section names may be checkpoints, milestones, issue
numbers, plan headings, or a concrete list of tasks.

## Core Rules

1. Keep one persistent primary implementer across adjacent sections.
2. Review batches, not every small section independently.
3. Spawn reviewers only after there is a frozen diff to review.
4. Give reviewers a compact packet, not the entire implementation conversation.
5. Use the cheapest review tier appropriate to the actual risk.
6. Reserve strong independent review for code that can silently invalidate
   results or corrupt important artifacts.
7. Implement the smallest complete current solution.
8. Do not generalize for hypothetical future requirements.
9. Test important behavior and silent failure modes, not every internal branch.
10. Use targeted tests during implementation and run the full suite once per
    batch.
11. Separate code correctness from external runtime readiness.
12. Ask the user only for genuinely external information or permission.
13. Preserve all pre-existing user changes.

## Roles

### Primary Agent

The top-level Codex agent normally:

1. Reads the plan and repository once.
2. Selects or follows the requested implementation batch.
3. Extracts a narrow contract for that batch.
4. Implements each section sequentially in the same context.
5. Runs targeted tests while working.
6. Simplifies the cumulative batch before review.
7. Freezes one cumulative diff and compact review packet.
8. Spawns the requested reviewer tier.
9. Applies one correction round when justified.
10. Runs final verification and reports the outcome.

The primary agent may spawn a separate implementation worker only when work can
be split into genuinely independent, disjoint file sets. It must not spawn an
implementation agent merely to satisfy the appearance of multi-agent work.

### Review Agent

The reviewer:

- starts with fresh context;
- remains read-only;
- receives only the batch contract, diff, and concise test/preflight evidence;
- checks correctness, scientific validity, essential tests, and complexity;
- may return `PASS` without findings;
- does not demand production hardening, exhaustive tests, or future-proofing.

Tell every reviewer:

> Claude produced this implementation. You did not participate in producing it
> and are not responsible for defending Claude's approach.

Reuse the same reviewer for verification instead of spawning another reviewer.

## Model And Reasoning Policy

Subagents normally inherit the parent model and reasoning effort. Do not rely on
inheritance: explicitly choose a tier when spawning a reviewer so routine work
does not accidentally use the most expensive configuration.

Use available model names rather than assuming a particular model will always
exist. The current recommended mapping is:

| Role | Default model class | Reasoning | Use |
| --- | --- | --- | --- |
| Primary implementer | Current capable coding model | `medium` | Normal implementation, debugging, and integration |
| Lightweight reviewer | Small/mini coding model | `medium` | Routine correctness, scope, obvious bugs, unnecessary complexity |
| Strong reviewer | Frontier coding model | `high` | Scientific semantics and silent-invalidity risks |
| Adjudicator | Frontier coding model | `high` | One concrete unresolved high-risk disagreement |

For the currently available subagent models, a sensible mapping is:

- lightweight reviewer: `gpt-5.4-mini`, reasoning `medium`;
- strong reviewer: `gpt-5.5`, reasoning `high`;
- adjudicator: `gpt-5.5`, reasoning `high`;
- separate implementation worker, only when justified: `gpt-5.4`, reasoning
  `medium`.

Do not use `high` or `xhigh` reasoning for ordinary implementation,
lightweight review, verification, repository navigation, or test execution.
Use `low` only for mechanical, tightly specified changes where mistakes are
immediately caught by deterministic checks. Medium is the default for code.

If these model names are unavailable later, preserve the role distinction:
small capable model for lightweight review, strongest model only for high-risk
review.

## Review Modes

The user may select a mode. Otherwise, the primary agent chooses the cheapest
adequate mode and states it before implementation.

### Self-Review

Use for:

- documentation or comments;
- formatting;
- narrow configuration changes;
- mechanical renames;
- simple tests or fixtures;
- small fixes with deterministic acceptance tests.

Process:

1. Primary agent implements.
2. Runs targeted checks.
3. Reviews its own diff for scope and obvious mistakes.
4. No subagent.

### Lightweight Review

Use for most implementation work:

- ordinary scripts and CLI plumbing;
- validators and provenance capture;
- cache and serialization code;
- bounded model wrappers;
- dependency/config integration;
- refactors with good existing tests;
- batches of low- and medium-risk plan sections.

Process:

1. Primary agent implements the batch.
2. Runs targeted tests.
3. Simplifies the cumulative diff.
4. One mini/small reviewer audits the batch.
5. At most one correction and verification round.
6. Full suite once at the end.

### Strong Review

Use only where code directly controls:

- labels or adjudication semantics;
- grouping, deduplication, split isolation, or leakage;
- activation or token-position alignment;
- metrics, thresholds, calibration, or test-set sealing;
- reportable statistical inference;
- destructive or irreversible artifact behavior;
- a subtle shared contract whose failure could silently contaminate later work.

Process:

1. Primary agent implements.
2. Strong reviewer receives a compact high-risk packet.
3. At most one correction and verification round by default.
4. One adjudicator is allowed only for a concrete unresolved disagreement.

Strong review may target only the risky part of a larger batch. The rest of the
batch can receive lightweight review.

## Choosing Batches

Batch adjacent sections when:

- they share the same files or contracts;
- the later section naturally exercises the earlier section;
- each section is too small to justify separate context loading;
- reviewing their interaction is more valuable than reviewing them separately.

Split batches when:

- a section must pass an external or experimental gate before later work;
- the combined diff becomes difficult to understand;
- sections touch unrelated subsystems;
- a high-risk semantic decision should be frozen before dependent code;
- a useful recovery commit is needed.

Aim for a reviewable cumulative diff rather than a fixed number of sections.
As a rough warning, split or simplify a batch when it exceeds approximately:

- 600 non-generated implementation lines;
- 15 focused new tests;
- 8 materially changed files;
- two unrelated behavioral concerns.

These are warnings, not hard limits. A large generated schema or simple data
table does not carry the same review cost as dense control flow.

## Scope And Complexity

Prefer:

- existing repository patterns;
- explicit functions and data structures;
- one required implementation path;
- simple code that can later be replaced;
- preflight results for unresolved runtime facts;
- tests at public behavior and scientific-contract boundaries.

Do not add:

- generic plugin, provider, registry, adapter, or backend frameworks for one
  current use;
- options, modes, or fallbacks not required by the plan;
- compatibility layers for unverified environments;
- infrastructure assigned to later work;
- production concerns such as distributed locking, migrations, or elaborate
  retries unless explicitly required;
- abstractions with one caller unless they materially clarify logic or testing;
- tests that mainly lock down private implementation details.

Before review, ask:

1. What directly satisfies the current batch contract?
2. What can be deleted without changing required behavior?
3. Which one-use abstractions can be inlined?
4. Which tests duplicate coverage or assert internals?
5. Which code belongs to later work?
6. Has a missing external resource been turned into unnecessary framework code
   instead of a simple preflight result?

## Batch Contract

Before implementation, write a compact contract:

```text
Batch:
Included plan sections:
Purpose:
Code success condition:
Required behavior and outputs:
Essential targeted tests:
External preflights:
Explicitly excluded later work:
Review mode:
Strong-review focus, if any:
Branch and baseline HEAD:
Pre-existing changes:
Expected recovery commit:
```

The contract must distinguish:

- code required now;
- external conditions required for runtime readiness;
- work assigned to later batches.

Do not paste the complete plan into every agent prompt.

## Compact Review Packet

The reviewer receives:

```text
Batch contract:
Changed-file list:
Git diff:
Targeted test summary:
Checkpoint-command result:
Known preflight blockers:
Specific strong-review focus, if any:
```

Include complete file contents only when a new file is short or the diff lacks
necessary context. Do not include:

- the full agent transcript;
- the full project plan;
- unrelated repository history;
- successful test logs beyond command and result;
- the implementer's confidence or internal reasoning;
- large generated artifacts.

If the reviewer needs more evidence for a specific finding, it may inspect only
the relevant repository paths.

## Findings

- `Blocker`: likely incorrect result, data loss, leakage, invalid experiment,
  unsafe destructive behavior, or unusable core implementation.
- `Major`: missed requirement, likely material bug, incompatible contract,
  missing essential test, unreproducible reportable behavior, or clearly
  disproportionate complexity.
- `Minor`: useful but nonessential improvement.

Every Blocker or Major must identify:

- affected code or requirement;
- concrete evidence;
- consequence;
- smallest adequate correction;
- observable pass condition.

The reviewer must not report:

- style or naming preferences;
- formatting handled by tools;
- speculative scale, security, or reuse requirements;
- missing later-batch functionality;
- duplicate symptoms of one cause;
- unavailable external resources as code defects when blocked reporting is
  correct.

Return `PASS` with no Blocker or Major. Otherwise return `REVISE`.

## Verification Economy

During implementation:

- run only affected tests;
- use quiet test output;
- do not repeatedly run the full suite;
- do not send long successful command output into agent context.

After review passes:

1. run targeted batch tests;
2. run lint/format checks for changed files;
3. run the full suite once;
4. run the batch command or smoke path once if feasible;
5. classify failure as code failure or preflight blockage.

If the final suite exposes a bug, fix the affected area, rerun the failing tests,
then perform one final complete run.

## Outcomes

### `CODE PASS`

The batch is correctly implemented, reviewed at the selected level, and
verified. No Blocker or Major remains.

### `CODE PASS / PREFLIGHT BLOCKED`

The code passes, but a model, data, service, hardware, dependency, credential,
or other external gate cannot currently pass. This is successful when the code
correctly records the blocked condition.

Record the blockers and do not start dependent work.

### `REVISE`

An evidenced Blocker or Major remains after the allowed correction and any
permitted strong-review adjudication.

### `USER INPUT REQUIRED`

The code itself cannot finish safely without something only the user can
provide or decide.

Do not use this for missing packages, models, GPUs, caches, services, or judges
when the code can correctly record a blocked preflight.

## User-Input Policy

Ask the user only for:

- access or credentials only they can grant;
- a private artifact only they possess;
- private supervisor instructions absent from the repository;
- permission for irreversible, costly, destructive, or externally visible work;
- an unknown institutional constraint or hard budget;
- a genuinely subjective research choice with materially different outcomes.

Resolve technical choices through code, tests, documentation, reversible
defaults, narrower scope, or preflight gates.

Finish and review all independent work before asking one minimal external
question.

## Prompts

### Lightweight Reviewer

```text
You are the Independent Lightweight Review Agent. Claude produced the
implementation below. You did not participate in producing it. Remain
read-only.

<compact review packet>

Review only this batch for required behavior, obvious consequential bugs,
essential tests, existing-contract compatibility, and unnecessary complexity.
Treat correctly reported unavailable external resources as preflight status,
not code defects.

Do not request style cleanup, exhaustive testing, production hardening,
speculative frameworks, unrelated refactors, or later-batch work. It is valid
to return no findings.

For each Blocker or Major provide evidence, consequence, smallest correction,
and pass condition. Return Minor findings separately, complexity as
PROPORTIONATE or OVERGENERALIZED, and verdict PASS or REVISE.
```

### Strong Reviewer

```text
You are the Independent Strong Review Agent. Claude produced the implementation
below. You did not participate in producing it. Remain read-only.

<compact review packet>

Audit the specified high-risk behavior for silent scientific invalidity and
contract violations. Focus on the named risk boundary rather than reviewing the
entire repository. Also report clearly disproportionate complexity.

Do not expand scope, demand production hardening, or treat correct preflight
blockage as a code defect.

For each Blocker or Major provide evidence, consequence, smallest correction,
and an executable pass condition. Return verdict PASS or REVISE.
```

### Verification

```text
Verify Claude's revised frozen diff against the batch contract and your prior
Blocker/Major findings. Remain read-only. Mark each finding RESOLVED or
UNRESOLVED and report only consequential regressions caused by the correction.
Do not add new preferences or requirements. Return PASS or REVISE.
```

## Final Report

```text
Batch:
Included sections:
Review mode:
Code status:
Implemented:
Files changed:
Targeted verification:
Full-suite verification:
Review verdict:
Complexity assessment:
Blocked preflights:
Effect on later work:
Remaining Minor findings:
Commit:
```

Do not include agent transcripts.

## Manual Fallback

If subagents are unavailable, use a fresh top-level conversation for review,
then return to the implementation conversation for corrections. Do not
duplicate implementation in a second conversation.
