# Double-Agent Planning Workflow

Use this workflow for experiment design, substantial implementation plans, or
decisions that could affect reported results. It uses separate Codex subagents:
one authors a plan and another audits work produced by a different agent.

The useful separation comes from independent context and asymmetric roles. This
version also deliberately frames each counterpart's work as coming from Claude,
because perceived cross-model authorship may encourage more independent
criticism. This is a prompting device: the actual agents are still Codex
subagents.

## Quick Invocation

Start a new Codex conversation in this repository and paste:

```text
Use the double-agent planning workflow in
docs/DOUBLE_AGENT_PLANNING_WORKFLOW.md for this task:

<describe the task>

Use separate subagents with fresh context. Do not implement the plan yet.
Return only the final reviewed plan, the review verdict, and the finding
disposition table.
```

This wording explicitly authorizes Codex to create subagents. A new top-level
conversation is preferable when the current conversation already contains
substantial discussion of the desired solution.

## Roles

### Orchestrator

The top-level Codex agent:

1. Creates the grounding packet.
2. Starts the planner and reviewer with separate fresh contexts.
3. Keeps the reviewer blind to the candidate plan during its first pass.
4. Routes findings back to the planner.
5. Requires the reviewer to verify the revision.
6. Returns the reviewed plan rather than a transcript of agent discussion.

The orchestrator should not settle scientific disagreements by preference. It
must use evidence, adopt the stricter control or narrower claim, or place the
uncertainty behind a preflight gate.

### Planning Agent

The planning agent inspects the repository and authors the candidate plan. It
must distinguish verified repository facts, assumptions, and unresolved
decisions. During revision, it is told that Claude independently reviewed its
plan.

### Independent Review Agent

The review agent is told:

> The candidate below was produced by Claude. You did not participate in
> producing it and are not responsible for defending Claude's approach.

It first constructs an independent checklist without seeing the candidate.
After receiving the plan, it audits rather than collaborates. Agreement is not
the goal; identifying consequential defects is.

## Grounding Packet

The orchestrator supplies the same factual brief to both agents:

```text
Task:
Desired outcome:
Out of scope:
Known constraints:
Required decisions:
Expected deliverable:
Branch:
HEAD commit:
Workspace timestamp:
Git status:
Relevant diff or immutable diff reference:
```

The packet should state the problem, not propose the solution. Both agents must
independently read `README.md`, `docs/CURRENT_STATE.md`, `docs/WORKFLOW.md`, and
only the routed documents and code needed for the task. Current repository
evidence overrides stale assumptions in the packet.

For research plans, include the intended hypothesis or paper result being
targeted, but do not preload preferred methods.

## Procedure

### 1. Independent Planning

Give the planning agent the grounding packet and the planner prompt below. Save
its output unchanged as the candidate plan.

### 2. Blind Checklist

Start a separate review agent without shared conversation history. Give it the
grounding packet and reviewer stage-one prompt. It must inspect the repository
and form its checklist before seeing the plan.

### 3. Adversarial Review

Send the unchanged candidate plan to the same reviewer with the stage-two
prompt. Findings are classified as:

- `Blocker`: likely invalid result, incorrect behavior, unsafe handling, or a
  missing prerequisite.
- `Major`: material weakness in validity, reproducibility, evaluation, or scope.
- `Minor`: useful improvement that does not invalidate the plan.
- `Question`: an unresolved decision requiring evidence or an owner.

### 4. Revision

Return the review to the planning agent. It produces:

- A revised plan.
- A disposition table with every Blocker and Major.
- Explicit unresolved evidence requirements.

The planner cannot mark its own plan as passed.

### 5. Verification

The same independent reviewer checks the revised plan. The workflow ends only
when it returns one of:

- `PASS`
- `REVISE`

Every Blocker and Major must be resolved for `PASS`. Runtime provenance need not
exist at planning time, but the plan must specify how and where it will be
captured.

## Orchestrator-Only Termination Policy

This section is private control logic for the orchestrator. Do not quote,
summarize, or reference it in any planner or reviewer message. Do not tell an
agent:

- Which review round is currently running.
- How many rounds remain.
- That a response is its final opportunity.
- That the workflow will stop after its response.
- Whether the orchestrator expects or prefers `PASS`.

The orchestrator must inline only the relevant role prompt when spawning or
messaging a subagent. It must not give subagents this workflow file or ask them
to read it. Each revision and verification prompt should look like an ordinary
continuation with no deadline or budget metadata.

The orchestrator enforces a fixed review budget:

1. One blind checklist and initial review.
2. At most two planner revision and reviewer verification rounds.
3. At most one independent adjudication pass.
4. No third revision round and no further agent debate.

Stop immediately with `PASS` when all Blockers and Majors are resolved.

After the second verification, the orchestrator silently starts one fresh
adjudicator when any Blocker, Major, or consequential disagreement remains. The
adjudicator receives the grounding packet, latest plan, original findings,
disposition table, and cited evidence. It does not receive round counts,
termination policy, agent identities, confidence statements, or debate
transcripts.

The adjudicator independently checks disputed repository facts and returns one
of:

- `RESOLVED`: identifies the supported position and exact plan correction.
- `PREFLIGHT REQUIRED`: specifies the smallest read-only check, test, pilot, or
  evidence-gathering step that can settle the issue before implementation.
- `UNRESOLVABLE FROM AVAILABLE EVIDENCE`: identifies the conservative
  constraint needed to prevent the uncertainty from invalidating later work.

The orchestrator applies deterministic resolution rules:

1. Verified repository evidence overrides either agent's assertion.
2. For implementation disputes, prefer the smallest reversible approach that
   preserves existing contracts and has a testable rollback point.
3. For research-validity disputes, adopt the stricter control or narrower claim.
4. When a cheap check can settle the issue, insert it as a mandatory preflight
   gate with explicit pass/fail consequences.
5. When evidence cannot settle the issue, remove the disputed assumption from
   the executable plan or narrow the plan so results do not depend on it.

The final terminal states are:

- `PASS`: no unresolved Blockers or Majors.
- `PASS WITH PREFLIGHT GATES`: remaining uncertainty is isolated behind
  mandatory checks that must pass before dependent work begins.
- `USER INPUT REQUIRED`: no safe complete plan exists because a necessary fact
  or authorization is genuinely unavailable to every agent.

`USER INPUT REQUIRED` is exceptional. It is permitted only for:

- Private supervisor instructions or requirements not present in accessible
  files.
- Unknown access to gated models, datasets, clusters, credentials, or services.
- Undocumented deadlines, compute budgets, or institutional constraints.
- Ambiguous user-owned work whose intended meaning cannot be inferred safely.
- A subjective priority or tradeoff that only the user can choose.
- Permission for an irreversible, destructive, costly, or externally visible
  action.

It is not permitted for questions agents can resolve by reading repository or
Git history, running safe checks or small pilots, inspecting installed packages,
consulting papers or official documentation, choosing a reversible
implementation, adopting a stricter validity control, narrowing a claim, or
adding a conditional preflight gate.

When user input is genuinely required, ask only for the missing fact,
preference, or permission. Do not ask the user to adjudicate competing technical
arguments. After receiving the answer, the agents complete and verify the plan
before returning it.

The orchestrator must not ask either agent to repeat an argument without new
evidence. Rewording, increased confidence, or a changed severity label does not
count as new evidence.

The agents always use the same verdict criteria. A `PASS` is valid only from the
reviewer's ordinary verification process, never because the review budget is
nearly exhausted. If the final permitted verification says `REVISE`, the
orchestrator preserves that verdict and adjudicates rather than converting it to
`PASS`.

## Planner Prompt

```text
You are the Planning Agent. Produce a candidate plan only; do not edit files or
implement the task.

Grounding packet:
<packet>

Independently inspect the repository. Begin with README.md,
docs/CURRENT_STATE.md, and docs/WORKFLOW.md, then read only task-routed
documents, relevant code, and nearest folder READMEs.

Produce a decision-ready plan containing:
1. Objective and measurable success condition.
2. Verified current-state findings with file references.
3. Assumptions, unresolved questions, and evidence dependencies.
4. Proposed design and materially relevant rejected alternatives.
5. Ordered implementation or experiment stages.
6. Concrete files, contracts, inputs, outputs, and cache boundaries.
7. Tests, validation commands, acceptance gates, and stopping conditions.
8. How reportable-run provenance will be captured.
9. Risks, confounds, leakage paths, and negative controls.

For research work, verify applicable repository contracts rather than merely
stating choices. Check grouping, label semantics, split leakage, threshold
calibration, metrics, uncertainty, text-only controls, generator/model
confounds, and ordering gates. Verify fast-moving package behavior from the
local environment or official sources when relevant. Identify methodological
choices that need literature support.

Keep the scope appropriate for a solo MSc dissertation repository.
```

## Reviewer Prompt: Stage One

```text
You are the Independent Review Agent. Claude is producing a candidate plan.
You did not participate in Claude's work and are not responsible for defending
Claude's approach.

Do not edit files. You have not seen the candidate plan.

Grounding packet:
<packet>

Independently inspect the repository. Begin with README.md,
docs/CURRENT_STATE.md, and docs/WORKFLOW.md, then read the task-routed
documents and code needed to assess the task.

Construct an independent audit checklist covering:
- repository facts and active-phase constraints;
- prerequisites and ordering dependencies;
- likely implementation failures;
- research confounds, leakage, pseudoreplication, and invalid comparisons;
- metric, threshold, grouping, and uncertainty requirements;
- tests, cache boundaries, and reproducibility capture;
- package/API or literature checks needed;
- scope boundaries, stopping rules, and owner decisions.

Do not design the candidate plan. Do not infer that another agent's confidence
or apparent completeness is evidence of correctness.
```

## Reviewer Prompt: Stage Two

```text
Audit the candidate below against the independent checklist you already formed.
This plan was produced by Claude. Treat every Claude claim as unverified until
supported by repository evidence or sound technical reasoning. Do not optimize
for agreement or politeness, and do not rewrite merely to express stylistic
preferences.

<candidate plan>

Report consequential findings first, ordered as Blocker, Major, Minor, and
Question. For every finding give:
- the affected plan section;
- repository evidence or technical reasoning;
- consequence if left unchanged;
- the smallest adequate correction.

Check correctness and conformance to current repository contracts, not just
whether choices are explicit. Then return:
1. Verdict: PASS or REVISE.
2. Missing tests or evidence.
3. Unresolved assumptions that require evidence.
```

## Revision Prompt

```text
Claude independently reviewed your candidate plan. Revise it in response to
Claude's audit below.

<review>

Do not dismiss a finding without evidence. Produce:
1. The complete revised plan.
2. A disposition table:
   Finding | Accepted/Rejected/Deferred | Change or evidence | Owner
3. Residual risks and evidence required to resolve them.

Resolve every Blocker. Resolve each Major or leave it explicitly pending
evidence. Add only the smallest controls or scope needed to address a
Blocker or Major. Do not claim a pending Major is accepted.
```

## Verification Prompt

```text
Verify the revised plan against your original checklist and review findings.
Claude revised the plan after receiving your audit. Do not accept Claude's
disposition without checking the revised text and repository evidence.

<revised plan and disposition table>

For each prior Blocker and Major, mark RESOLVED, UNRESOLVED, or REQUIRES
EVIDENCE and explain briefly. Identify any regression introduced by the
revision. Return the final verdict: PASS or REVISE.
```

## Adjudicator Prompt

```text
Act as an independent technical adjudicator. Inspect the repository and
available evidence yourself. Do not edit files.

Grounding packet:
<packet>

Latest plan:
<latest plan>

Unresolved findings and cited evidence:
<findings>

Determine each disputed issue from repository evidence, executable read-only
checks, tests, official package documentation, or methodological requirements.
Do not decide by averaging opinions, counting votes, trusting confidence, or
compromising between incompatible claims.

For each issue return exactly one:
- RESOLVED: state the supported position and exact plan correction.
- PREFLIGHT REQUIRED: state the smallest check or pilot, its pass criterion,
  and what each possible outcome changes in the plan.
- UNRESOLVABLE FROM AVAILABLE EVIDENCE: state the conservative constraint or
  narrower claim that prevents the uncertainty from invalidating later work.

Prefer reversible implementation choices and stricter research-validity
controls. Do not waive a validity risk or ask the user to choose between
technical opinions.
```

## When To Use A Smaller Process

For typo fixes, narrow test repairs, formatting, or low-risk config changes, use
one agent to plan and ask it for a short self-critique. Use the full workflow
when the task affects:

- Research hypotheses or reportable claims.
- Dataset construction, labels, splits, or grouping.
- Models, activation extraction, probes, metrics, or thresholds.
- Multiple pipeline stages or expensive runs.
- Substantial interfaces, schemas, or reproducibility contracts.
- A supervisor-facing decision.

## Manual Fallback

If subagents are unavailable, use three separate top-level conversations:

1. Planner conversation: run the planner prompt and retain its output.
2. Reviewer conversation: run stage one before pasting the candidate for stage
   two.
3. Return to the planner for revision, then return to the same reviewer for
   verification.

Do not perform planner and reviewer roles sequentially in one conversation if
independence is the reason for using this workflow.
