# Working Conventions

## Project Purpose

This repo supports an MSc dissertation testing whether LLM guardrails improve when given access to internal model activations.

Current priority is an IPP MVP:
- compare text-only vs activation-based harmful-intent detection
- use a small, defensible setup
- optimise for fast iteration and easy reruns (batching, caching, simple scripts)

## Working Principles (Important)

- Do not overengineer. Research speed and clarity are more important than abstraction.
- Research quality is non-negotiable: methods, metrics, and evaluation design should be literature-backed where possible.
- This repo is intended to support dissertation-quality / publishable work; reproducibility and professional execution take priority over convenience.
- Prefer small composable scripts and reusable utility functions.
- Cache expensive GPU outputs (activations, embeddings, fitted probes) in `artifacts/`.
- Keep evaluation reproducible: fixed seeds, explicit split files/configs, saved thresholds.
- Default to concise outputs: one table first, extra analysis later.

## MVP Scope (Frozen Spec)

- Question: At fixed low FPR on benign prompts, does activation access improve detection?
- Model: `Qwen2.5-7B-Instruct`
- Harmful data: HarmBench sample
- Benign data: XSTest-safe sample
- Activation guardrail: linear probe on hidden states (`end-of-prompt` token)
- Text baseline: TF-IDF + logistic regression (frozen embeddings optional later)
- Metrics:
  - primary: `TPR @ 1% FPR`
  - secondary: `ROC-AUC`

## Repo Conventions

- `src/agguardrails/`: reusable code
- `scripts/`: runnable entrypoints for experiments
- `configs/`: experiment/model settings
- `results/`: final tables/plots
- `artifacts/`: cached intermediates (usually not committed)
- `docs/PROJECT_CONTEXT.md`: broader dissertation context and extension roadmap
- `docs/RESEARCH_WORKFLOW.md`: experiment hygiene, provenance, and reporting standards

## Collaboration Notes

- Preserve simplicity and readability.
- Follow `docs/RESEARCH_WORKFLOW.md` for experiment hygiene and provenance expectations.
- Prefer changes that reduce rerun time (batching, avoiding repeated forward passes).
- If introducing a dependency or architectural layer, justify it in one sentence.
- Keep docs current when changing experiment flow.
