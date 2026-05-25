# activation-guardrails

MSc dissertation repo (Edinburgh AI MSc).

Core question: **Do LLM guardrails become more effective when they can use
internal model activations, not just text?**

## Current State

This repository has been reset to a clean implementation skeleton for the
proposal plan. The immediate implementation target is now a faithful
reproduction of the CC++ paper as far as local access allows. The existing
Gemma 2 9B IT + WildJailbreak scaffold remains useful for the later open-model
adaptation, followed by harmfulness/refusal separation experiments and
SAE-based interpretability analysis.

Older code and result tables were removed from the active tree. They remain
recoverable from git history and the `pre-ccpp-fresh-start` tag.

## Intended Structure

```text
activation-guardrails/
├─ configs/
│  └─ ccpp/              # model, data, and run configs for the new pipeline
├─ scripts/
│  └─ ccpp/              # runnable experiment entrypoints
├─ src/agguardrails/     # reusable library code
├─ tests/                # tests rebuilt around the new pipeline
├─ artifacts/
│  └─ ccpp/              # local cached activations/features/models
├─ results/
│  └─ ccpp/              # generated metrics, tables, and analysis outputs
├─ docs/                 # project context and working conventions
├─ course-docs/          # university guidance and requirements
└─ msc-writeup/          # write-up subrepo
```

For project context, read:

- `docs/CURRENT_STATE.md` for immediate status
- `docs/PROJECT_PLAN.md` for the proposal-derived phase plan
- `docs/WORKFLOW.md` for git, experiment, logging, and reproducibility rules

## Execution Principles

- Keep each experiment stage cacheable: data build, response generation,
  judging, activation extraction, SAE encoding, probe training, and reporting.
- Every reportable result must record config path, seed, model/dataset versions,
  thresholding rule, and git commit.
- Use `TPR @ 1% FPR` as the primary metric and ROC-AUC as the secondary metric.
- Prefer small explicit scripts over a large experiment framework.

## Environment

Install dependencies from:

```bash
pip install -r requirements.txt
```

`requirements.txt` keeps flexible research ranges. Final/reportable runs should
record an exact environment snapshot in their result metadata.
