# Using Codex And Claude Code

Practical guide for working with coding agents in this repository. This file is
for the human user. `AGENTS.md` and `docs/WORKFLOW.md` remain the authoritative
agent instructions and workflow.

## Core Principle

Use an agent to help inspect, plan, implement, test, and explain one bounded
milestone. Do not delegate ownership of the research question or scientific
choices.

You remain responsible for:

- the claim being tested;
- datasets and labels;
- grouping, splits, and leakage controls;
- baselines, metrics, and thresholds;
- whether evidence supports a conclusion.

Tests can show that code behaves as specified. They cannot show that the
specification answers the right scientific question.

## Starting A Session

Open Codex or Claude Code in the repository root.

Both tools should receive the same repository instructions:

- Codex reads `AGENTS.md`.
- Claude Code reads `CLAUDE.md`, which imports `AGENTS.md`.

Tell the agent the outcome you want and whether it may edit. For the recorded
next task, ask it to read:

```text
Read AGENTS.md, README.md, docs/CURRENT_STATE.md, and docs/PROJECT_PLAN.md.
```

For later tasks, let `AGENTS.md` route it to the relevant document.

## The Normal Cycle

### 1. Understand

Start unfamiliar work read-only:

```text
Do not edit. Inspect the relevant documentation and code. Explain the current
behaviour, data flow, scientific assumptions, and anything you cannot infer.
```

### 2. Plan

Ask for one bounded milestone:

```text
Plan this issue without editing. State the objective, exclusions, expected
files, approximate diff size, acceptance evidence, scientific decisions needing
my approval, and conditions that would make you stop and replan.
```

### 3. Approve

Approve a specific plan, not a general direction:

```text
I approve this milestone exactly as stated. Implement only this scope. Stop and
ask before changing a scientific choice, adding a dependency or abstraction, or
exceeding the planned files or diff size.
```

Confirm scientific decisions in writing before implementation.

### 4. Implement And Verify

Let the agent implement the approved milestone and run the checks named in the
plan. Require it to distinguish:

- mechanical evidence: tests, linting, schemas, known examples, and commands;
- scientific evidence: leakage analysis, comparison validity, robustness, and
  whether the result supports the intended claim.

### 5. Explain And Review

Before accepting the work, ask:

```text
Explain the final diff in review order. Trace one representative example,
identify every scientific choice, describe failure modes, and state what the
tests do not prove.
```

Then inspect:

```bash
git diff --stat
git diff
git status --short
```

Accept it only if you can explain its contract, data flow, assumptions, and
limitations.

### 6. Commit

Commit after understanding and accepting the complete diff. A completed
milestone should normally form one meaningful recovery point.

## Choosing The Right Kind Of Session

| Need | Approach |
| --- | --- |
| Learn unfamiliar code | Current chat, read-only explanation |
| Make a scientific or protocol decision | Plan-first, no edits |
| Implement an approved narrow change | Continue the planning chat |
| Review risky or shared code | Fresh read-only chat |
| Diagnose a surprising result | Stop expansion and investigate |
| Work on an unrelated milestone | Start a fresh chat |
| Submit cluster work | Approve the exact host and action first |

A fresh review is useful when the authoring context could bias the review. It is
not necessary for every tiny change.

## Codex And Claude Code

Use the same prompts and repository workflow with both tools. Choose whichever
interface is more convenient; do not maintain separate project instructions.

The practical differences that matter here are:

- Codex loads `AGENTS.md` directly.
- Claude Code receives it through the `@AGENTS.md` import in `CLAUDE.md`.
- Each tool has its own planning, review, permission, and context controls.
- Claude checkpoints may help rewind Claude-made edits, but Git is the shared
  recovery record for both tools.

Do not ask one tool to implement a change and the other to implement the same
change independently. A second tool or fresh chat is most useful for read-only
review of a frozen diff.

## When Work Goes Wrong

Tell the agent:

```text
Stop editing. Report changed files, unapproved decisions, and independently
useful work. Do not discard any user changes. Propose a narrow salvage or safe
restoration to the last accepted commit, but do not apply it yet.
```

Inspect the diff before restoring anything. Never use a destructive Git command
without understanding exactly which changes it will remove.

## Current First Task

The current task in `docs/CURRENT_STATE.md` is planning-only: compare dataset and
label-design options for the Phase 1 baseline. Start a fresh chat with:

```text
Read AGENTS.md, README.md, docs/CURRENT_STATE.md, and docs/PROJECT_PLAN.md.

Work in planning mode only. Do not edit files, download datasets, or inspect
final-evaluation examples.

Plan the dataset and label design milestone recorded in CURRENT_STATE.md.
Compare a small number of defensible options by claim supported, label
provenance, grouping unit, split roles, leakage and shortcut risk,
availability, and reproduction faithfulness.

Return alternatives, risks, unresolved questions, acceptance evidence, and a
recommendation for my review. Do not implement the protocol.
```

Read the proposal critically. The next step is to approve, revise, or reject
its scientific choices, not to begin coding automatically.
