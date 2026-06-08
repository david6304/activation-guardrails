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

## Authority And History

- Active project documentation and this file are tracked in Git.
- Respect the status and next action recorded in `docs/CURRENT_STATE.md`; do not
  infer active work from the presence of code, configs, artifacts, or history.
- `docs/research_log.md`, `docs/references/`, `docs/meetings/`, `.codex/`, and
  `.claude/` are local or private context, not startup reading.
- The pre-reset plans preserved in commit `280d363` are a non-authoritative
  historical archive. Do not inspect or restore them by default. Use them only
  for an explicit provenance question or to recover a specific discarded
  detail that is absent from active documentation.

## Working Rules

- Treat the repository as actively edited by the user. Never overwrite or
  revert unrelated work.
- Keep changes small, explicit, and research-oriented.
- Work on one bounded milestone at a time. Do not build infrastructure for
  future stages.
- Prefer scripts, configs, and saved metadata over framework layers.
- Distinguish exploratory runs from reportable experiments before execution.
- Methods, labels, splits, baselines, metrics, thresholds, and claims require
  scientific justification proportionate to their effect on conclusions.
- Pause on surprising results. Diagnose before extending the pipeline.
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
