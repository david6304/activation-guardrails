# activation-guardrails

MSc AI dissertation repository, University of Edinburgh.

Core question: **Can internal model activations make LLM guardrails more
effective and interpretable than text-only detection?**

## Current Status

See `docs/CURRENT_STATE.md` for the current project status and next bounded
milestone.

## Repository

```text
configs/             versioned experiment choices
scripts/             runnable entrypoints
src/agguardrails/    reusable experiment code
tests/               automated tests
artifacts/           local cached intermediates
results/             generated metrics and report artifacts
docs/                tracked project and workflow documentation
course-docs/         local university guidance
msc-writeup/         local write-up repository
```

Folder READMEs define what belongs in each directory.

## Project Documents

- `docs/CURRENT_STATE.md`: current status and next bounded milestone
- `docs/PROJECT_PLAN.md`: research phases and decision gates
- `docs/WORKFLOW.md`: coding, experiment, planning, and review workflow
- `docs/CLUSTER.md`: cluster and Slurm guidance

## Environment

Install the working dependency ranges with:

```bash
pip install -r requirements.txt
```

Use `pytest` and Ruff for local verification. Reportable experiments must also
save exact environment and provenance metadata as defined in
`docs/WORKFLOW.md`.
