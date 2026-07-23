# Four-Week Dissertation Finish Plan

Decision date: 2026-07-22. The dissertation has approximately four weeks
remaining. Reportable experiments should freeze after at most ten further days,
leaving the remainder for analysis, writing, revision, and reproducibility.

## Intended narrative

The dissertation will retain the proposal's three-part structure, but reduce
each phase to the shortest experiment capable of supporting a defensible claim:

1. **Practical performance:** activation probes add information to text guards
   under a surface-form shift, with an activation--text ensemble as the intended
   primary positive result.
2. **Representation diagnosis:** token position explains what the detector is
   measuring, while score-rank and threshold analyses distinguish calibration
   shift from loss of linear harmfulness separation.
3. **SAE interpretation:** sparse features test how much of the transferable
   signal can be retained and interpreted, without requiring them to outperform
   dense probes.

One practical positive result, one explanatory result, and one bounded
interpretability result are sufficient. The project will not require all three
phases to produce benchmark wins.

## Frozen scope

- Backbone: `google/gemma-3-27b-it` only for new reportable experiments.
- Conditions: plain control, Swahili primary shift, and reverse as a hard
  negative control.
- Primary activation position: `t_inst`, the final token of the actual user
  instruction.
- Control position: `t_post-inst`, the final token of the complete templated
  prompt before generation, which is the position used by the current runs.
- Primary detector: all-layer logistic regression at `t_inst`.
- Named latent baseline: Zhao-style layer-averaged cosine-centroid score.
- Primary metric: TPR at 1% FPR. AUROC is secondary. Report paired bootstrap
  confidence intervals for the main matched comparisons.
- Existing 12B and current-position results are supplementary scale and
  representation controls; they do not gate the new work.

Swahili is selected independently of probe test performance as the primary
shift because Gemma 3 27B retains 99% benign QA capability while the existing
WildGuard screen detects 0% of harmful prompts. Reverse remains useful even if
negative because it tests a stronger transformation under which the current
probe loses discrimination.

## Phase 1: matched activation-versus-text guardrail comparison

### Question

Do instruction-position activations provide a useful signal beyond text-only
guards under a meaning-preserving surface transformation?

### Minimum experiment

1. Extract `t_inst` and `t_post-inst` activations in the same 27B forward pass.
2. Fit the all-layer logistic probe and Zhao centroid using the frozen plain
   training split and operational-harm labels.
3. Evaluate exactly the same plain, Swahili, and reverse test examples with:
   - the corrected activation probe;
   - Zhao centroid;
   - character n-gram TF--IDF;
   - WildGuard;
   - ShieldGemma's continuous score; and
   - a simple activation--ShieldGemma ensemble.
4. For continuous detectors, report both:
   - **strict transfer:** the threshold fixed on plain tune negatives at 1% FPR;
   - **condition-matched calibration:** a threshold fixed on translated tune
     negatives at 1% FPR, without using transformed harmful examples.
5. Keep the existing WildChat calibration as a separate deployment-style
   background alert-rate analysis. Do not call its unlabelled alert rate an FPR.
6. Select no threshold, layer, detector, or condition using test outcomes.

The ensemble should use a fixed, simple rule: map activation and ShieldGemma
scores to percentiles on the relevant tune-negative background and take their
maximum, then calibrate that final score on the same tune-negative background.

### Intended result and permissible claim

The target result is that the activation probe exceeds the strongest text-only
baseline on Swahili at matched FPR, and that the activation--text ensemble is
best overall or shows materially complementary errors. The strongest defensible
claim would be:

> Harmfulness separability in instruction-position activations transfers better
> than text-guard detection under Swahili surface shift. Condition-matched
> calibration uses no shifted harmful examples, and combining activation and
> text scores improves robustness further.

The frozen-English-threshold result remains reportable even if negative. A
condition-matched positive result does not establish a calibration-invariant
guardrail.

### Stop rule

Once the exact matched table, confidence intervals, and error-overlap analysis
exist, Phase 1 is frozen. French, Hindi, Zulu, more models, and additional
ciphers do not enter the critical path.

## Phase 2: token-position and failure-mode decomposition

### Question

Are current failures caused by measuring a refusal-skewed prompt position,
condition-dependent score calibration, or genuine loss of linearly accessible
harmfulness information?

### Minimum experiment

- Compare all-layer logistic and Zhao centroid at `t_inst` and `t_post-inst`.
- Use the same plain, Swahili, and reverse data from Phase 1.
- Report AUROC, harmful and benign score distributions, strict-transfer TPR/FPR,
  and condition-matched TPR/FPR.
- Interpret high AUROC with a failed frozen threshold as calibration shift;
  interpret AUROC collapse as loss of discrimination.
- Use existing response/refusal labels only if they can be joined reliably with
  no substantial new generation run. Their analysis is optional, not a gate.

This phase is geometric and behavioural, not causal. Without an intervention,
the dissertation will not claim that token-position results causally separate
harmfulness and refusal. The originally proposed refusal abliteration is
deferred.

### Intended result

The expected explanatory pattern is that `t_inst` better isolates transferable
harmfulness, Swahili primarily shifts score calibration while preserving rank,
and reverse causes a larger loss of discrimination. This explains both the
positive Phase 1 result and the reverse failure without requiring every
transformation to work.

### Stop rule

One token-position figure and one table across the three frozen conditions are
sufficient. No MLP-hook, generation-trajectory, or broad layer sweep follows.

## Phase 3: minimal dense-versus-SAE interpretation

### Question

How much of the transferable harmfulness signal is retained in sparse features,
and are the important features stable between plain and Swahili?

### Minimum experiment

- Use the available `google/gemma-scope-2-27b-it` `resid_post` SAEs at blocks 31
  and 40 only.
- Fix one standard checkpoint per layer before evaluation, matching the source
  hook and choosing a practical medium-sparsity variant.
- Use `t_inst` only and the same plain, Swahili, and reverse split.
- Compare a dense single-layer logistic probe with an SAE-feature logistic
  probe.
- Report AUROC, TPR at 1% FPR, and the fraction of dense above-chance AUROC
  retained by the SAE:

  \[
  R_{\mathrm{SAE}} =
  \frac{\mathrm{AUROC}_{\mathrm{SAE}} - 0.5}
       {\mathrm{AUROC}_{\mathrm{dense}} - 0.5}.
  \]

- Compare the highest-weighted SAE features across plain and Swahili and inspect
  only the top 10--20 features using existing descriptions where available.

The SAE need not beat the dense probe. Retaining useful signal while providing
interpretable or cross-condition-stable features is a valid result; weak SAE
transfer is also reportable as a performance--interpretability limitation.

### Stop rule

Two layers and one SAE checkpoint per layer are enough. No width sweep,
all-layer SAE sweep, automated Delphi pipeline, activation steering, or feature
intervention is required.

## Fallback and hard exclusions

If the matched Swahili result is negative for both the probe and ensemble, one
preselected vowel-removal experiment is the only permitted transformation
fallback. It is justified by the existing 61% capability and weak guard screen.
Do not search transformations until a positive outcome appears.

The following are excluded from the four-week critical path:

- new corrected-position 12B experiments;
- further language or cipher search;
- MLP probes and broad per-layer selection;
- refusal abliteration or Heretic studies;
- large response-generation studies;
- full SAE layer/width sweeps;
- large-scale Delphi interpretation or steering; and
- additional backbone models.

## Schedule

### Days 1--4

- Implement dual-position extraction and run the smallest real-model smoke.
- Run the corrected 27B probe and Zhao centroid.
- Freeze the primary activation detector.
- Draft dataset, label, token-position, and calibration methods concurrently.

### Days 5--7

- Complete the exact matched text/activation comparison and ensemble.
- Produce the primary table, confidence intervals, and error-overlap figure.
- Freeze Phase 1.

### Days 8--10

- Complete the token-position analysis.
- Run the two-layer SAE comparison and inspect the small feature set.
- Freeze all reportable experiments. Only bug-fix reruns are permitted after
  this point.

### Remaining time

- Week 3: write introduction, literature review, methods, and Phase 1 results;
  finalise Phase 2/3 analysis.
- Week 4: write discussion and conclusion, revise claims and limitations,
  complete figures/appendices/reproducibility details, obtain feedback, and
  proofread.

## Dissertation-level success criterion

The desired three-part contribution is:

1. a practical activation or activation--text advantage on the frozen Swahili
   comparison;
2. a token-position analysis that distinguishes calibration shift from
   representation loss; and
3. a bounded SAE result describing the accuracy--interpretability trade-off.

This is a narrowing of the proposal rather than a change of topic: it preserves
the open-model guardrail comparison, harmfulness/refusal motivation, and SAE
interpretability questions while dropping interventions and exhaustive sweeps
that are not necessary for a distinction-quality dissertation.
