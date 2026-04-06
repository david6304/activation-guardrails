# Current State

This file is a short current snapshot for future agents.

Rules:

- keep this file concise
- update or overwrite stale items rather than appending an endless history
- use it for current focus, completed milestones, active issues, and immediate next steps
- do not turn it into a full experiment log or changelog

## Current Focus

Phase 0 infrastructure for the WildJailbreak refusal experiment is now
implemented locally and CPU-validated. Next step is `C1`: run the full
WildJailbreak refusal pipeline on the cluster and inspect whether refusal and
harmfulness separate meaningfully on the same prompts. See
`docs/DISSERTATION_PLAN.md` for the full plan.

## Immediate Next Steps

### C1. First WildJailbreak refusal cluster run

- Build the vanilla + adversarial source datasets with
  `configs/main/refusal_wildjailbreak.yaml`
- Generate responses for vanilla and adversarial sets separately
- Judge both response sets with `label_refusals.py`
- Relabel both datasets so `label=refusal` and `source_label=harmfulness`
- Extract activations for vanilla splits plus adversarial in one job
- Encode SAE features, then train/evaluate refusal probes and the Latent Guard
  baseline

### Harmfulness vs refusal comparison

- Reuse the same activation and SAE caches with `--label-key source_label` to
  train harmfulness probes on the same prompts
- Compare text vs dense vs SAE vs Latent Guard under both targets before
  deciding what to emphasize in the write-up

### Early diagnostics after C1

- Check refusal rate on vanilla harmful, vanilla benign, and adversarial
  harmful prompts before over-interpreting probe metrics
- If the model refuses almost everything, inspect class balance and adversarial
  refusal rate before moving to extensions

## Completed Work

- `0a` token position abstraction:
  `src/agguardrails/features.py` now supports `"last"` and
  `"last_instruction"`, and
  `scripts/main/extract_activations.py` accepts `--token-position`
- `0b` WildJailbreak refusal dataset builder:
  `scripts/main/build_refusal_dataset.py` now supports a WildJailbreak mode,
  writes a separate adversarial refusal source set, and
  `configs/main/refusal_wildjailbreak.yaml` defines the 2000/2000/2000 slice
- `0c` dual-label support:
  activation and SAE caches now preserve both `label` and `source_label`, and
  `train_activation_probes.py`, `train_sae_probes.py`, and
  `train_text_baseline.py` accept `--label-key`
- `0d` Latent Guard baseline:
  `src/agguardrails/latent_guard.py` and
  `scripts/main/train_latent_guard.py` added, using the existing fixed-FPR
  evaluation path
- CPU-safe validation for the new Phase 0 infrastructure:
  targeted script `--help` checks, targeted `pytest`, and cluster-script tests
  all passed locally
- Core repo structure and end-to-end experiment pipeline
- Pilot run (Qwen2.5-7B-Instruct, HarmBench/XSTest) — pipeline validation only
- WildJailbreak harmfulness probing — negative result (text > dense > SAE)
- Refusal probing on AdvBench+Alpaca (120-train and 700-train splits)
- Cipher transfer (reverse) on 700-train — positive for dense, negative for SAE
- Full dissertation plan reframed around Zhao et al. and literature gap

## Key Context for Agents

- `CLAUDE.md` for repo conventions and agent rules
- `docs/DISSERTATION_PLAN.md` for the full research plan, core/extension breakdown, and decision rules
- `docs/WORKING_CONVENTIONS.md` for coding standards and testing expectations
- Existing pipeline code is in `src/agguardrails/` and `scripts/main/`
- Tests are in `tests/test_main_scripts.py` and `tests/test_cluster_scripts.py`
