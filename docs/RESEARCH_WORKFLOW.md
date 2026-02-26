# Research Workflow (Publication-Grade)

This repo is for research intended to be dissertation-quality and potentially publishable.

## Non-Negotiables

1. Literature-backed decisions
- Use strong papers / benchmark papers as the default reference for methods, metrics, and evaluation framing.
- If a design choice is not standard, document why it was chosen and what alternatives were considered.
- Prefer benchmark-aligned datasets and metrics over bespoke ones unless there is a clear research reason.

2. Reproducibility first
- Every reported result must be traceable to:
  - code commit hash
  - config file
  - seed
  - model version
  - dataset version / sample file / split file
- Save thresholds and selection criteria (especially for `TPR @ fixed FPR`).

3. Professional execution
- Separate exploratory work from reported results.
- Keep scripts deterministic where possible.
- Avoid hidden manual steps.
- Record assumptions and limitations clearly.

## Standard Experiment Workflow

1. Define the experiment question
- State the exact hypothesis and metric(s) before coding.
- Link the relevant literature/benchmark framing in the experiment notes or config comments.

2. Freeze a small spec
- Model(s)
- Dataset(s) and sampling plan
- Splits
- Features
- Metrics
- Thresholding regime
- Selection rule (e.g., best layer on validation ROC-AUC)

3. Implement with cacheable stages
- `data` build/sampling
- `feature` extraction (GPU-heavy, cache outputs)
- `train` (probe/baseline)
- `eval` / reporting

4. Run and log provenance
- Save result metadata with:
  - timestamp
  - git commit hash
  - config path
  - random seed
  - hardware/device info (optional but useful)

5. Report conservatively
- Distinguish:
  - pilot / debug runs
  - final reported runs
- Do not overwrite final outputs silently.

## Coding Standards for This Repo

- Prefer simple scripts and explicit data flow over complex abstractions.
- Optimize for iteration speed:
  - batching
  - no repeated forward passes
  - reuse cached activations/features
- Add tests for:
  - data schema/splits
  - metrics (especially fixed-FPR thresholding)
  - serialization/loading of artifacts
- Use `src/` modules for reusable logic, `scripts/` for entrypoints.

## Research Etiquette / Evidence Rules

- Do not claim improvements without the exact operating point and baseline definition.
- For safety-related claims, report both detection and false-positive behavior.
- Note dataset limitations and distribution assumptions.
- If using a nonstandard prompt format or token position, document it explicitly.

## How To Work In This Repo

- Read `docs/WORKING_CONVENTIONS.md` and `docs/PROJECT_CONTEXT.md` first before starting a new work session.
- Before implementing a method, check whether the metric/eval protocol is already defined.
- Prefer literature-consistent implementations for metrics and baselines.
- When adding a method, update docs with:
  - what changed
  - why (literature or engineering reason)
  - impact on reproducibility

## Minimal Per-Experiment Output Contract (recommended)

Each experiment run should produce:
- metrics file (JSON/CSV)
- config snapshot (or config path + git hash)
- selected threshold(s)
- optional small summary table in `results/`

This keeps the project auditable and dissertation-ready without heavy process overhead.
