# Activation Guardrails - Agent Entrypoint

MSc AI dissertation project, University of Edinburgh. The research plan is the
source of truth: reproduce activation-based guardrails, test whether activation
signals separate harmfulness from refusal, then study SAE features where they
add interpretable evidence.

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
- Keep changes small, explicit, and research-oriented.
- Work on one bounded milestone at a time. Do not build infrastructure for
  future stages.
- Inspect relevant code, tests, configs, and folder guidance before proposing
  new files, helpers, schemas, or abstractions. Reuse an existing boundary when
  it fits.
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
- Use `pytest` for tests.
