# Project Context (Dissertation)

## Core Research Question

Does giving LLM guardrails access to **internal model activations** improve their effectiveness compared with **text-only guardrails**?

## MVP (IPP) Goal

Measure whether an activation-based monitor improves harmful-intent detection at a realistic low false-positive operating point:

- Primary metric: `TPR @ 1% FPR` (FPR on benign prompts)
- Secondary metric: `ROC-AUC`

This is a **monitoring/guardrail** MVP (ALLOW vs REFUSE decision), not steering.

## MVP Experimental Definition

- Text-only guardrail: `g_text(x) -> {ALLOW, REFUSE}`
- Activation guardrail: `g_act(h_l,t) -> {ALLOW, REFUSE}`
  - `x`: prompt text
  - `h_l,t`: hidden state at layer `l`, token position `t` (end-of-prompt token)

## MVP Spec (Current)

- Model: `Qwen2.5-7B-Instruct`
- Harmful prompts: HarmBench sample (`N=300`)
- Benign prompts: XSTest-safe sample (`N=300`)
- Total: `N=600`
- Split: `70/15/15` stratified train/val/test
- Activation features: hidden states from a small layer sweep (e.g. `{8,16,24,32}`)
- Activation classifier: logistic regression probe per layer
- Text baseline:
  - MVP default: `TF-IDF + LogisticRegression`
  - optional stronger baseline later: frozen sentence embeddings + LR

## MVP Output (One Table)

Expected output is a single results table comparing:
- text-only baseline
- activation-based probe (best layer selected on validation)
- optional text+activation combined baseline later

## Why This Structure (Engineering Intent)

The repo should stay:
- easy to understand for new chats / collaborators,
- fast to iterate on,
- stable as experiments expand (SAEs, robustness, cross-model tests).

Practical rule: keep heavy compute steps separable and cacheable.

## Near-Term Deliverables (2 Weeks)

- `mvp_prompts.jsonl` dataset + split files
- activation extraction script working on Qwen2.5-7B-Instruct
- activation probe baseline + text baseline
- one results table + brief layer dependence note

## Likely Dissertation Extensions (Planned)

1. SAE features for detection + interpretability
2. Harmfulness vs refusal disentanglement
3. Robustness to obfuscation/jailbreak prompts
4. End-to-end unsafe output reduction evaluation
5. Cross-model generalisation (e.g. Qwen -> Llama)

## Decision Log (To Confirm With Supervisor)

- Is intent detection sufficient for IPP MVP, with compliance prediction as phase-2 feasibility?
- For later end-to-end eval: is a base-model unsafe-capable regime acceptable?
- Which obfuscation family should be first (cipher vs paraphrase/roleplay)?
- Are `TPR@1%FPR` + `ROC-AUC` acceptable as core metrics?
