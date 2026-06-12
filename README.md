# activation-guardrails

MSc AI dissertation repository, University of Edinburgh.

Core question: **Can internal model activations make LLM guardrails more
effective and interpretable than text-only detection?**

## Current Status

The experiment implementation has intentionally been reset so the project can
be rebuilt through small, reviewable milestones. The previous implementation
is recoverable from commit
`d389a35dd888ef773e4ecc5c69d2e17abb61a2e2`.

The first empirical protocol is frozen only as a non-reportable smoke test.
See `docs/CURRENT_STATE.md` for the approved scope and next action.

## Repository

```text
configs/             future versioned experiment choices
scripts/             future runnable entrypoints
src/agguardrails/    minimal package scaffold
tests/               future automated tests
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
- `docs/USING_CODE_AGENTS.md`: practical guide to using Codex and Claude Code
- `docs/CLUSTER.md`: cluster and Slurm guidance

## Environment

The retained dependency files describe the anticipated research environment,
not a frozen experiment environment. Install the working ranges with:

```bash
pip install -r requirements.txt
```

The accepted dataset-manifest implementation and the locally validated
response-generation entrypoint are described in `docs/CURRENT_STATE.md`.
Response generation remains unexecuted pending explicit authorization for the
exact cluster target and command. Reportable experiments require a later
frozen protocol and must save exact environment and provenance metadata as
defined in `docs/WORKFLOW.md`.
