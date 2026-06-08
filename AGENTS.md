# Activation Guardrails - Agent Entrypoint

MSc AI dissertation project, University of Edinburgh. The proposal plan is the
source of truth: reproduce CC++-style activation guardrails on open-weight
models, then test harmfulness/refusal separation, then use SAE features for
interpretability.

## Start Here

1. Read `README.md` and `docs/CURRENT_STATE.md`.
2. Read only the routed doc needed for the task.
3. Inspect folder READMEs before adding files to a folder.
4. Do not read `msc-writeup/ipp/proposal.tex`, `course-docs/`, or large docs by
   default. Use them only when the task needs proposal wording, rubric details,
   or an exact citation/source check.

## Current Focus

- Fresh-start CC++ scaffold after commit `59841bf`.
- Safety tag before cleanup: `pre-ccpp-fresh-start`.
- Primary model: `Gemma 2 9B IT`.
- Primary dataset: `WildJailbreak`.
- Core metrics: `TPR @ 1% FPR` and `ROC-AUC`.
- Build simple, cacheable experiment stages before adding abstractions.

## Routing

| Task | Read |
| --- | --- |
| Current status / next action | `docs/CURRENT_STATE.md` |
| Research plan, phases, deliverables | `docs/PROJECT_PLAN.md` |
| Git, experiment, logging, reproducibility workflow | `docs/WORKFLOW.md` |
| Cluster or SLURM work | `docs/CLUSTER.md` plus existing cluster skill/instructions if available |
| Package/API decisions | `requirements.txt`, installed package metadata, local package docs/code, then official docs |
| Where files belong | nearest folder `README.md` |
| Proposal/rubric exact wording | `msc-writeup/ipp/proposal.tex` or `course-docs/`, only when needed |

`docs/PROJECT_PLAN.md` is the only active planning summary. Do not use older
planning docs if they appear in local history or ignored files.

## Working Rules

- Treat the repo as actively edited by the user. Do not overwrite or revert
  their changes unless explicitly asked.
- Keep changes small and research-oriented. This is a solo dissertation repo,
  not a production platform.
- Prefer explicit scripts, configs, and saved metadata over framework layers.
- Preserve reproducibility: every reportable result needs git commit, config,
  seed, data/model revisions, threshold rule, and environment provenance.
- Methods, baselines, datasets, and metric choices should be literature-backed.
- Pause on surprising results. Diagnose before extending the pipeline, and
  suggest a supervisor check-in at decision points.
- Do not write polished dissertation prose unless asked. Prefer talking points,
  outlines, or edit suggestions.

## Research Log

`docs/research_log.md` is local and ignored. Ask whether to log when a result
changes direction, a non-obvious decision is made, a surprising finding is
resolved, or a supervisor-relevant question emerges. Do not log routine command
runs or minor fixes.

## Tooling

- `ruff` for linting/formatting.
- `pytest` for tests.
- Use `rg`/`rg --files` for repo search.
