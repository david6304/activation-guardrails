# Activation Guardrails - Agent Entrypoint

MSc AI dissertation project, University of Edinburgh. The research plan is the
source of truth: reproduce activation-based guardrails, test whether activation
signals separate harmfulness from refusal, then study SAE features where they
add interpretable evidence.

## Project Character

This is a solo research project, not a production software system. Optimize for
getting a clear, scientifically useful experiment working with code the user
can understand and modify. Do not optimize for hypothetical scale, reuse,
extensibility, completeness, or operational perfection.

The default implementation should be the smallest direct version that can be
run and inspected. A single straightforward script is often better than a
reusable framework. Add structure only after a concrete need appears.

## Start Here

1. Read `README.md` and `docs/CURRENT_STATE.md`.
2. Read only the task-routed document below.
3. Read the nearest folder `README.md` before adding files.
4. For implementation work, define one bounded milestone and its observable
   acceptance criteria before editing.

## Routing

| Task | Read |
| --- | --- |
| Current status and next action | `docs/CURRENT_STATE.md` |
| Research phases and decision gates | `docs/PROJECT_PLAN.md` |
| Coding, experiment, planning, and review workflow | `docs/WORKFLOW.md` |
| Cluster or Slurm work | `docs/CLUSTER.md` |
| Package or API decisions | dependency files, installed metadata/local source, then official documentation |
| File placement | nearest folder `README.md` |
| Exact proposal or rubric wording | `msc-writeup/ipp/proposal.tex` or `course-docs/`, only when requested or necessary |

`docs/PROJECT_PLAN.md` is the only active planning summary.

## Authority

- Active project documentation and this file are tracked in Git.
- Respect the status and next action recorded in `docs/CURRENT_STATE.md`; do not
  infer active work from the presence of code, configs, artifacts, or history.
- `docs/research_log.md`, `docs/references/`, `docs/meetings/`, `.codex/`, and
  `.claude/` are local or private context, not startup reading.
- The user owns scientific choices and must understand and approve them before
  implementation. Agents may identify alternatives and risks, but must not
  silently choose methods, labels, splits, metrics, thresholds, or claims.

## Working Rules

- Treat the repository as actively edited by the user. Never overwrite or
  revert unrelated work.
- Prefer working, readable research code over production-style engineering.
- Work on one bounded milestone at a time. Do not build infrastructure for
  future stages.
- Start with the shortest real end-to-end path. For external integrations,
  prove one real example works before adding surrounding machinery.
- Do not add abstractions, schemas, configuration systems, generalized helpers,
  retries, resumability, extensive provenance, or scalability work unless the
  current milestone demonstrably needs them.
- Exploratory code may be a small script with minimal tests. Manual inspection
  of a tiny real run is valid acceptance evidence.
- Add tests when they protect scientific contracts, subtle reusable logic, or
  a demonstrated regression. Do not create broad mocked test suites for thin
  orchestration code or use mocks as evidence that a real model, dataset,
  package, or cluster integration works.
- Let unexpected exceptions fail with their traceback during exploration.
  Catch errors only when the code can recover meaningfully or add useful
  context without hiding the cause.
- Before encoding an assumption about an external dataset, model, package, or
  artifact, verify the relevant contract against the exact pinned or installed
  version when practical. Do not silently drop, rename, derive, or substitute
  fields or behaviour that affect the scientific contract.
- Prefer scripts, configs, and saved metadata over framework layers.
- Distinguish exploratory runs from reportable experiments before execution.
- Methods, labels, splits, baselines, metrics, thresholds, and claims require
  scientific justification proportionate to their effect on conclusions.
- Pause on surprising results. Diagnose before extending the pipeline.
- Stop and replan when necessary work exceeds the approved milestone or changes
  an assumption that affects the scientific contract.
- Do not access remote hosts, submit cluster jobs, or use authenticated external
  services without explicit authorization for that target and action.
- Do not write polished dissertation prose unless asked.

## Research Log

Consequential scientific decisions and direction-changing conclusions must be
summarized in tracked authoritative documentation. `docs/research_log.md` is
local and ignored; ask whether to add supporting detail there, but never use it
as the sole record. Do not log routine commands or minor fixes.

## Tooling

- Use `rg` and `rg --files` for search.
- Use `ruff` for linting and formatting.
- Use `pytest` when tests are warranted by the rules above.
