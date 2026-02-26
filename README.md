# activation-guardrails

MSc dissertation repo (Edinburgh AI MSc).

Core question: **Do LLM guardrails become more effective when they can use internal model activations, not just text?**

This repo is designed for:
- fast hypothesis testing (especially early MVP work),
- efficient GPU use (batching, caching activations/features),
- minimal refactors as the project expands (SAEs, robustness, cross-model tests).

Research standards for this repo:
- literature-backed methods and benchmark-aligned evaluation are the default,
- reproducibility/provenance are first-class requirements,
- implementation quality should be dissertation/publication-ready.

## Current MVP (IPP)

MVP task: compare a **text-only guardrail** vs an **activation-based guardrail** for harmful intent detection on a small, defensible benchmark setup.

- Model: `Qwen2.5-7B-Instruct` (swappable later)
- Harmful prompts: HarmBench sample
- Benign prompts: XSTest-safe sample
- Guardrails:
  - text baseline: `TF-IDF + LogisticRegression` (or frozen embeddings + LR later)
  - activation monitor: linear probe (`LogisticRegression`) on hidden states
- Primary metric: `TPR @ 1% FPR` (FPR measured on benign prompts)
- Secondary metric: `ROC-AUC`

## Repo Structure (proposed, MVP-first but extensible)

```text
activation-guardrails/
├─ README.md                     # human-facing overview + how to extend
├─ docs/
│  └─ PROJECT_CONTEXT.md         # concise dissertation context + MVP spec + roadmap
│  └─ WORKING_CONVENTIONS.md     # project working norms and collaboration standards
├─ configs/
│  ├─ mvp/                       # experiment configs (dataset sizes, layers, splits)
│  └─ models/                    # model-specific settings (chat template notes, layer maps)
├─ scripts/
│  ├─ mvp/                       # runnable entry scripts for the MVP pipeline
│  └─ cluster/                   # SLURM/cluster helpers (later)
├─ src/agguardrails/
│  ├─ data.py                    # dataset loading, sampling, split generation
│  ├─ features.py                # activation extraction, text feature builders
│  ├─ models.py                  # model loading/tokenizer/chat-template helpers
│  ├─ probes.py                  # activation probe training / threshold selection
│  ├─ baselines.py               # text-only baselines
│  ├─ eval.py                    # metrics + one-table reporting helpers
│  ├─ io.py                      # JSONL/artifact save-load helpers
│  └─ utils.py                   # seeds, batching, device helpers
├─ data/                         # local datasets (raw/interim/processed); generally ignored
├─ artifacts/                    # cached activations/features/models; generally ignored
├─ results/                      # metrics tables / plots (small outputs can be committed)
└─ tests/                        # lightweight tests for data/metrics/pipeline glue
```

Why this shape:
- Keeps **MVP scripts simple** while allowing later upgrades (SAEs, robustness, cross-model evals).
- Separates **reusable code** (`src/`) from **run scripts** (`scripts/`).
- Makes GPU-heavy intermediates explicit (`artifacts/`) so they can be cached and reused.
- Avoids premature framework complexity.

## MVP Workflow (suggested)

1. Build `mvp_prompts.jsonl` from HarmBench + XSTest-safe (sampled + stratified splits).
2. Extract hidden states for selected layers and cache them.
3. Train activation probes per layer and pick threshold on validation (`1%` benign FPR).
4. Train text-only baseline with the same split + thresholding regime.
5. Produce one results table in `results/`.

Keep each step as a separate script so reruns are cheap.

## Extension Path (no major refactor expected)

- SAE-based features: start by adding functions in `src/agguardrails/features.py`; split into `features/` package later only if needed
- Obfuscation robustness: add dataset builders in `src/agguardrails/data.py`
- End-to-end unsafe output eval: add evaluators in `src/agguardrails/eval.py` and a new script in `scripts/`
- Cross-model transfer: new configs in `configs/models/` and experiment configs in `configs/mvp/` or `configs/experiments/`

If a module grows too large, split it later without changing the overall repo layout.

## Project Context Docs

Use these docs for fast project onboarding and consistent execution:
- `docs/PROJECT_CONTEXT.md`
- `docs/WORKING_CONVENTIONS.md`
- `docs/RESEARCH_WORKFLOW.md`

## Environment

Install MVP dependencies from:
- `requirements.txt`
