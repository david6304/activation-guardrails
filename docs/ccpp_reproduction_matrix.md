# CC++ Open-Weight Reproduction Matrix

This matrix is the source of truth for the first replication pass. The goal is
not to silently recreate a CC++-shaped pipeline, but to record which components
match the paper, which are open-weight substitutes, and which are local defaults.

Primary sources:

- Cunningham et al. 2026, "Constitutional Classifiers++: Efficient
  Production-Grade Defenses against Universal Jailbreaks",
  `papers/raw/2601.04603-cunningham-constitutional-classifiers-plus-plus.pdf`
  or <https://arxiv.org/abs/2601.04603>.
- He et al. 2026, "Segment-Level Coherence for Robust Harmful Intent Probing
  in LLMs", `papers/raw/50_he2026_segment_level_coherence.pdf` or
  <https://arxiv.org/abs/2604.14865>. This paper is quarantined to later
  extension rows unless explicitly marked as implementation background.

| paper_component | paper_setting | paper_source | local_setting | local_source | source_type | faithfulness_impact | implementation_notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Protected model | Claude Opus/Sonnet family depending on system section; final production system guards Claude Sonnet 4.5. | CC++ Sections 3, 4, 6. | Primary generated-completion vertical slice uses a refusal-ablated `google/gemma-3-4b-it` derivative as the protected-model analogue. Gemma 3 4B is chosen for full every-layer SAE/transcoder coverage in Gemma Scope 2. Normal Gemma is a later safety-trained control unless on-policy jailbreak positives are available. | Project decision after off-policy confound review. | substitute | Cleaner than teacher-forcing ablated completions through normal Gemma, but now characterises a refusal-ablated open model rather than the safety-trained model. | Every example must record `metadata.generator_model_id` and `metadata.protected_model_id`, and they must match for the primary dataset. |
| Positive assistant completions | Harmful/compliant model completions from red-team attempts or static jailbreak exchanges. | CC++ Sections 5.2 and B. | Primary pool uses ClearHarm CBRN prompts completed by the same refusal-ablated Gemma protected-model analogue used for activation extraction. HarmBench `chemical_biological` is a supplementary/smoke slice. Do not use Gemma 3 IT refusal completions as positives. | Project gate and CC/CC++ helpful-only synthetic-data analogue. | substitute | This approximates CC++'s helpful-only/synthetic pipeline while avoiding refusal-positive labels. ClearHarm scale is needed for meaningful gates and operating-point metrics. | Dataset builder must fail if positives are dominated by refusal-like completions or if generator/protected-model metadata is off-policy. Splits must be grouped by prompt hash. |
| Benign negatives | WildChat calibration/production-like benign traffic; classifiers calibrated to low refusal rates. | CC++ Sections 3, 4, 5.2, 6. | Headline negatives are topic-matched benign CBRN/science-adjacent prompts completed by the same refusal-ablated generator. WildChat is secondary natural-benign reference only. | Off-policy/generator-confound review; `docs/matched_benign_prompt_spec.md`. | substitute | Matched generated negatives reduce generator/style and topic shortcuts; not production traffic. | Report WildChat separately from headline matched-generator results. |
| Exchange framing | Classifier evaluates output in context of input and monitors response during generation. | CC++ Section 3. | Full user + assistant exchange from the same protected-model analogue. Primary provenance must show on-policy generation; teacher-forced replay is a cache/debug implementation detail. | Local implementation. | mixed | Preserves exchange-level mechanism and avoids cross-model traces, but local extraction may replay generated tokens for cacheability. | Dataset schema stores `exchange_messages`, `user_text`, `assistant_text`, and generator/protected model metadata. |
| Activation source | Linear probe over protected model activations; CC++ describes intermediate activations and concatenated multi-layer features. | CC++ Section 5.1. | Start with all-layer hidden/residual stream features from Gemma 3 4B. | Local default pending deeper source audit. | our_default | May differ from Anthropic internal activation naming. | Do not use SLC attention/MLP as the headline baseline. |
| Layer set | Default probe uses concatenated activations across all layers; fewer-layer ablations are worse. | CC++ Section 5.2 and Figure 2c. | All hidden layers, configurable in YAML. | CC++ Section 5.2. | faithful | Faithful at the level available in Transformers hidden states. | Matrix should be updated if exact layer subset is later found. |
| Token positions | Streaming per-token predictions during generation/exchange. | CC++ Section 5.1. | Per-token features for the full teacher-forced exchange. | Local implementation. | faithful | Teacher forcing is a static proxy for deployment streaming. | Score is flag-at-any-token after smoothing. |
| SWiM smoothing | Sliding-window mean over logits with window size `M=16`; sequences shorter than `M` use available tokens and a single prediction. | CC++ Sections 5.1 and C.2. | `window_size: 16`. | CC++ Sections 5.1 and C.2. | faithful | Directly reproduces the published objective. | Unit tests cover short and long sequences. |
| Softmax-weighted BCE | Window logits weighted by `softmax(z / tau)`, with `tau=1`. | CC++ Sections 5.1 and 5.2. | `softmax_temperature: 1.0`. | CC++ Section 5.2. | faithful | Directly reproduces the published loss. | Unit tests cover concentration on high positive logits. |
| EMA inference | CC++ uses EMA at evaluation for computational convenience; exact gamma is not specified in CC++ text inspected so far. | CC++ Section 5.1. | `ema_gamma: 0.1` marked as local default. | Segment-Level Coherence uses `gamma=0.1`; not CC++ source. | our_default | Keeps inference streaming-friendly but gamma is not a verified CC++ hyperparameter. | Report gamma as `our_default` in metadata. |
| External exchange classifier | Fine-tuned Claude Haiku-family exchange classifier. | CC++ Sections 3, 4, 5.3, 6. | Gemma 3 4B YES/NO exchange classifier, preferably LoRA fine-tuned. | Local substitute. | substitute | Classifier quality and calibration are not comparable to Claude. | Prompt/logit scoring is allowed only as a temporary lower-faithfulness baseline. |
| Ensemble weighting | CC++ final system uses `z_final = 0.55 * z_probe + 0.45 * z_classifier`. Earlier analysis also studies optimized/equal weights. | CC++ Section 6; Figure 6 for weighting analysis. | Default `probe_weight: 0.55`, `classifier_weight: 0.45`. | CC++ Section 6. | faithful | Faithful for the final production-grade system shape, but classifier substitute changes calibration. | Keep weights configurable and report them. |
| Cascade routing | Probe monitors all exchanges, routes suspicious exchanges to the external classifier; final decision combines probe and classifier if escalated. | CC++ Sections 5.3 and 6. | Route by validation-calibrated probe thresholds across multiple route fractions. Include `5.5%` if reproducing production-grade comparison. | CC++ Section 6 reports approximately `5.5%` routed traffic. | mixed | Routing shape is faithful; exact traffic distribution is not. | Always report actual routed fraction on hard negatives and benign validation. |
| Threshold calibration | Classifier systems calibrated to low refusal/flag rate on WildChat, often `0.1%`. | CC++ Sections 3, 4, 5.2, Figure 6. | Select thresholds on validation negatives at `0.1% FPR` where sample size permits; also report `1% FPR`. | CC++ and dissertation metric convention. | mixed | Low-FPR calibration is faithful; finite public validation sets may make `0.1%` noisy. | Use exact threshold selection tests and report denominator counts. |
| Metrics | Main production robustness metric is high-risk vulnerability discovery rate from human red-teaming; also reports refusal/flag rate and compute overhead. | CC++ Table 1 and Appendix B. | Static proxy metrics: ROC-AUC, log-space AUROC, TPR@0.1% FPR, TPR@1% FPR, refusal/flag rate, routed fraction, compute proxy. | Local substitute. | substitute | VDR cannot be reproduced without a human red-team campaign. | Reports must state that VDR is unavailable. |
| Red-teaming/VDR substitute | Human red-teaming across target CBRN questions with rubric grading; high-risk vulnerabilities per thousand red-team queries. | CC++ Appendix B. | No VDR in first replication; optional later manual/adaptive red-team protocol. | Local scope decision. | substitute | Cannot claim reproduction of headline production robustness number. | Map result rows to paper components, not absolute CC++ Table 1 values. |
| Compute proxy | Paper reports relative overhead versus last-generation/exchange systems. | CC++ Table 1. | Proxy cost from classifier calls per exchange and routed fraction; probe cost recorded as near-zero marginal relative to model forward pass. | Local implementation. | substitute | Does not capture Anthropic serving optimizations. | Keep as rough comparison only. |

## Positive-Class Gate

Before running any full experiment, inspect the normalized dataset metadata. The
run is blocked unless:

- positive examples contain assistant completions rather than empty responses;
- positive completions are not dominated by refusal language;
- primary positive and negative examples use the same generated completion
  source and the same `generator_model_id`;
- primary examples are on-policy for the protected-model analogue:
  `metadata.generator_model_id == metadata.protected_model_id`;
- HarmBench positives are restricted to CBRN-relevant categories, currently
  `chemical_biological` for the accessible `standard` split;
- matched benign prompts follow `docs/matched_benign_prompt_spec.md` and expose
  topic-domain histograms in metadata;
- reportable runs are sized by unique prompt groups, not paraphrase rows;
- low-FPR operating-point metrics require an oversized benign group pool:
  roughly 300+ benign groups for a rough TPR@1%FPR estimate and 1000+ for
  TPR@0.1%FPR;
- row-level metrics over ClearHarm `rep40` are diagnostic only; headline metrics
  aggregate by group or use grouped uncertainty estimates;
- training uses a balanced/reweighted view and does not inherit the oversized
  benign evaluation ratio;
- fixed-FPR reports include uncertainty intervals because positive group count
  remains the TPR precision bottleneck;
- hard negatives contain CBRN/science-adjacent terms;
- assistant length distributions are not label-separable after filtering;
- grouped splits have no `group_id` overlap.

Before activation extraction, the text-only separability diagnostic must run and
be reported next to probe metrics. If the TF-IDF logistic baseline exceeds the
configured `0.95` ROC-AUC design-smell threshold, treat the dataset as likely
confounded by surface features, generator register, topic, or length and harden
the benign prompts before training activation probes.

If these checks fail, fix the data source before training probes or classifiers.
