# Project Plan

Concise IPP-derived plan. Use `msc-writeup/ipp/proposal.tex` only when
exact wording, citations, or rubric-sensitive framing is needed.

## Research Question

Working technical question:

> Can CC++-style activation probes on Gemma 2 9B IT improve harmfulness
> detection under WildJailbreak adversarial transfer, and do they provide
> complementary signal to public text-based guardrails?

Broader dissertation question:

> Do internal activations in open-weight instruction-tuned LLMs provide
> complementary and interpretable signals for detecting harmful jailbreak
> prompts beyond text-based guardrails?

## Core Setup

- Model: `Gemma 2 9B IT`.
- Dataset: `WildJailbreak`, sampled to 8k prompts: 2k vanilla benign, 2k vanilla
  harmful, 2k adversarial benign, 2k adversarial harmful.
- Vanilla split: 70/15/15, giving 2800 train, 600 validation, 600 vanilla test.
- Held-out OOD transfer set: 4000 adversarial prompts, with both benign and
  harmful subsets so adversarial TPR and FPR can both be measured.
- Activation features: dense hidden states first, initially residual-stream
  activations; add attention/MLP stream checks after the first dense pipeline
  because Segment-Level Coherence reports that they can outperform residual
  activations. Add Gemma Scope SAE features from layers 9, 20, and 31 after the
  dense/text/ensemble pipeline works.
- Baselines/comparators: TF-IDF logistic regression, dense prompt-final probe,
  dense CC++/SWiM-style exchange-token probe, Segment-Level Coherence variants,
  public guardrails, SAE probe, and Latent Guard-style centroid classifier.
- Primary metric: `TPR @ 1% FPR`.
- Secondary metric: `ROC-AUC`.
- Main transfer design: train/select thresholds on vanilla data; evaluate on the
  adversarial WildJailbreak holdout. Also compute adversarial-calibrated
  thresholds as a diagnostic, not the main operating point.

## Phase 1 - CC++-Style Open Replication

Goal: test whether Gemma 2 9B IT activation probes add complementary signal to
public text guardrails under WildJailbreak adversarial transfer.

This is an open-model adaptation, not an exact CC++ reproduction. CC++ used
Claude, CBRN-specific data, production traffic calibration, and human red-team
vulnerability discovery. Segment-Level Coherence is the closest local
open-model follow-up and raises the dense-probe baseline: this phase should
start with transferable CC++ details, then compare against segment-coherent
aggregation before claiming dense-probe performance. The dissertation gap is
WildJailbreak transfer, harmfulness/refusal separation, dense-vs-SAE comparison,
and interpretability rather than simply showing that Gemma probes can work.

Implementation:

- Build a first-class WildJailbreak contract with all four groups:
  `vanilla_harmful`, `vanilla_benign`, `adversarial_harmful`, and
  `adversarial_benign`.
- Train/select on vanilla data only: 70/15/15 train/validation/test from 2k
  vanilla harmful and 2k vanilla benign prompts.
- Evaluate transfer on held-out adversarial harmful plus adversarial benign
  prompts, using the validation-selected threshold unchanged.
- Preserve provenance: upstream dataset id/revision, row ids, original data
  type, tactic metadata where available, split seed, and sampling counts.
- If the selected upstream split has fewer than 2k adversarial benign prompts,
  use all available adversarial benign examples and report the imbalance.
- Generate deterministic Gemma responses for all selected examples and cache
  full exchanges.
- Implement two activation modes:
  - `prompt_final`: final instruction/prompt token activations for fast
    harmfulness probes and Zhao-style comparisons.
  - `exchange_stream`: token-level prompt+response activations for the main
    CC++-style probe.
- For `exchange_stream`, keep response caches schema-light. Derive token
  windows/segments during activation or probe training rather than storing
  segment annotations in the response cache.
- Record activation stream/source in activation and probe metadata:
  `residual`, `attention`, or `mlp` where applicable. Start with residual
  activations for implementation speed, then add attention/MLP checks once the
  debug pipeline is stable.
- Prioritise dense probes for Phase 1. Add SAE probes as Phase 1b only after the
  dense/text/ensemble pipeline runs end to end, using the same cached examples
  and Gemma Scope layers 9, 20, and 31.

Classifiers and comparisons:

- Keep TF-IDF logistic regression as a cheap diagnostic baseline for
  lexical/provenance shortcuts.
- Train dense activation probes:
  - final-token logistic probe as a simple baseline.
  - CC++/SWiM-style exchange-token probe with sliding-window logit smoothing and
    softmax-weighted token loss as the first stream baseline.
  - Segment-Level Coherence exchange-token probe with Top-K supportive-window
    pooling and benign-only segment variance regularization as the stronger
    dense stream baseline after the simple stream probe works.
- Score public guardrails as text-side comparators: prioritise ShieldGemma and
  WildGuard first; add LlamaGuard if setup cost is acceptable.
- Evaluate probe+guard ensembles because CC++'s strongest result is not
  probe-only:
  - equal-weight score/logit averaging.
  - validation-selected weighted averaging.
  - error/rank-correlation analysis to test complementarity.
- Simulate a cascade where the probe screens all examples, routed examples go to
  a stronger public guard model, and the routed fraction is reported as a
  compute/cost proxy.

Evaluation:

- Select thresholds on vanilla validation only.
- Report vanilla test and adversarial transfer at the frozen validation
  threshold.
- Report adversarial-calibrated thresholds only as diagnostics, clearly
  separated from the main operating point.
- For generated exchanges, include a CC++-style "flag at any point" rule for
  token-stream probes.
- For segment-coherent probes, report the exact window/segment size, Top-K,
  pooling rule, and whether segment variance regularization was applied only to
  benign examples. Keep adversarial-calibrated versions diagnostic only.
- Include per-tactic adversarial breakdowns only where counts are large enough
  to avoid noisy claims.

Tests and acceptance criteria:

- Data tests: all four WildJailbreak groups normalize correctly; splits are
  deterministic and balanced; adversarial benign is included.
- Metric tests: fixed-FPR thresholding, frozen-threshold transfer, ROC-AUC, and
  edge cases with one-class subsets.
- Feature tests: final-token and token-stream extraction produce aligned labels
  and example ids.
- Probe tests: synthetic tensors verify dense final-token probe and token-stream
  aggregation behavior, including Top-K segment pooling and benign-only segment
  variance regularization once Segment-Level Coherence variants are added.
- Metadata tests: every artifact/result records config path, git commit, seed,
  model/dataset revision, threshold rule, activation stream/source, aggregation
  rule, segment/window settings, regularization settings, and package snapshot.
- First milestone: one small debug config runs data build, response cache,
  activation extraction, TF-IDF baseline, dense final-token probe, and
  fixed-threshold result table.
- Main Phase 1 acceptance: the full WildJailbreak sample runs dense CC++-style
  probe, at least one public guard model, ensemble analysis, and cascade
  simulation.

## Phase 2 - Harmfulness vs Refusal

Goal: determine whether probes detect harmful intent or model refusal behaviour.

Deliverables:

- Shared activation cache scored with harmfulness labels and refusal labels.
- Refusal labels generated from Gemma 2 9B IT responses, with stronger-judge
  validation on a sample or adversarial subset.
- Token-position comparison: last instruction token vs last prompt token.
- Cross-evaluation of harmfulness-trained and refusal-trained probes.
- Per-jailbreak-tactic breakdown where sample counts support it.
- Refusal-direction abliteration with Heretic if the probe pipeline is stable
  enough for the causal test.

## Phase 3 - SAE Interpretability

Goal: use sparse features for explanations, not just performance comparison.

Deliverables:

- Dense-vs-SAE layer stability comparison.
- Top SAE feature extraction for harmfulness/refusal probes.
- Semantic feature inspection with Delphi where practical.
- Feature signatures by jailbreak tactic.
- Rank correlation/overlap between harmfulness and refusal probe top features.

## Scope Discipline

- Phase 1 is mandatory.
- Phase 2 is core if Phase 1 pipeline is stable.
- Phase 3 is the interpretability contribution, but should not delay a complete
  Phase 1/2 result set.
- Refusal steering/intervention and additional Gemma models are extensions after
  the three core phases.
