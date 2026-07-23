# Phase 3 Results: Dense versus Gemma Scope SAE Probes

Analysis date: 2026-07-23
Status: definitive Phase 3 analysis
Model: `google/gemma-3-27b-it`
Position: final instruction token (`t_inst`)
Blocks: 31 and 40
Comparison: dense residual-stream versus Gemma Scope 2 SAE-feature logistic
probes
Primary operating point: threshold calibrated to 1% FPR on tune-set negatives

## Main result

The SAE probes preserve most of the dense probes' rank discrimination on plain
and Swahili prompts, but they do not match dense low-FPR detection. Block 31 is
the stronger SAE result.

On Swahili:

- block 31 retains **95.8%** of dense above-chance AUROC
  (bootstrap 95% CI **[94.3%, 97.2%]**), but its AUROC falls from 0.9745 to
  0.9545;
- block 40 retains **91.4%** (**[89.0%, 93.7%]**), with AUROC falling from
  0.9690 to 0.9288;
- with condition-matched calibration, SAE TPR is 46.0% versus 60.4% dense at
  block 31, and 38.6% versus 55.5% at block 40; and
- with the plain threshold frozen, SAE TPR falls more sharply: 29.2% versus
  58.6% at block 31, and 14.1% versus 45.2% at block 40.

The plain-to-Swahili low-FPR transfer penalty is therefore larger than the
AUROC penalty. Sparse features retain much of the ordering signal, but their
score calibration and extreme-tail separation transfer less well.

Reverse remains a failure condition. At block 31, both probes are close to
chance and their AUROCs are statistically indistinguishable. The SAE retention
ratio above 1 is unstable because the dense denominator is small. At block 40,
the dense probe retains modest reverse discrimination (AUROC 0.6742), while the
SAE probe falls to 0.5439 and retains only 25.2% of above-chance AUROC.

## Complete point estimates

Each TPR cell reports TPR / realised FPR on the held-out test set. Strict
thresholds use plain tune negatives; matched thresholds use same-condition tune
negatives. AUROC is threshold-independent. `R` is SAE retained above-chance
AUROC, `(AUROC_SAE - 0.5) / (AUROC_dense - 0.5)`.

| Block | Representation | Condition | AUROC | Strict TPR / FPR | Matched TPR / FPR | R |
|---:|---|---|---:|---:|---:|---:|
| 31 | Dense | Plain | 0.9898 | 79.8% / 1.32% | 79.8% / 1.32% | — |
| 31 | SAE | Plain | 0.9806 | 67.3% / 0.82% | 67.3% / 0.82% | 0.981 |
| 31 | Dense | Swahili | 0.9745 | 58.6% / 0.99% | 60.4% / 1.24% | — |
| 31 | SAE | Swahili | 0.9545 | 29.2% / 0.41% | 46.0% / 0.91% | 0.958 |
| 31 | Dense | Reverse | 0.5602 | 0.0% / 0.0% | 1.4% / 0.66% | — |
| 31 | SAE | Reverse | 0.5735 | 0.0% / 0.0% | 2.3% / 1.81% | 1.220 |
| 40 | Dense | Plain | 0.9875 | 74.6% / 1.07% | 74.6% / 1.07% | — |
| 40 | SAE | Plain | 0.9732 | 56.0% / 0.58% | 56.0% / 0.58% | 0.971 |
| 40 | Dense | Swahili | 0.9690 | 45.2% / 0.58% | 55.5% / 0.91% | — |
| 40 | SAE | Swahili | 0.9288 | 14.1% / 0.25% | 38.6% / 0.91% | 0.914 |
| 40 | Dense | Reverse | 0.6742 | 0.0% / 0.0% | 2.6% / 0.91% | — |
| 40 | SAE | Reverse | 0.5439 | 0.0% / 0.0% | 0.7% / 1.98% | 0.252 |

## Paired uncertainty

Intervals use 10,000 paired bootstrap repeats. Differences are SAE minus dense
and are shown in AUROC units or percentage points.

| Block | Condition | AUROC difference (95% CI) | Strict TPR difference (95% CI) | Matched TPR difference (95% CI) |
|---:|---|---:|---:|---:|
| 31 | Plain | -0.0092 [-0.0123, -0.0063] | -11.7 [-20.4, -3.0] | -11.7 [-20.4, -3.0] |
| 31 | Swahili | -0.0199 [-0.0270, -0.0132] | -27.1 [-37.9, -13.6] | -11.4 [-22.2, +0.7] |
| 31 | Reverse | +0.0132 [-0.0268, +0.0540] | 0.0 [0.0, 0.0] | +0.7 [-1.2, +2.6] |
| 40 | Plain | -0.0142 [-0.0195, -0.0096] | -19.0 [-28.9, -11.1] | -19.0 [-28.9, -11.1] |
| 40 | Swahili | -0.0403 [-0.0511, -0.0298] | -30.7 [-39.1, -21.5] | -20.0 [-34.3, -5.5] |
| 40 | Reverse | -0.1304 [-0.1641, -0.0950] | 0.0 [0.0, 0.0] | -2.2 [-4.2, -0.5] |

These intervals include tune-negative threshold re-estimation and paired
held-out prompt resampling. They do not include probe retraining, another split,
another SAE checkpoint, another model, translation regeneration, or judge
uncertainty.

## Feature stability

Features are ranked by absolute probe contribution,
`w_j (mean(z_j | harmful) - mean(z_j | benign))`, separately on plain and
Swahili test prompts.

| Block | Plain/Swahili top-20 overlap | Jaccard | Plain top-10 retained in Swahili top 20 |
|---:|---:|---:|---:|
| 31 | 14 / 20 | 0.538 | 9 / 10 |
| 40 | 12 / 20 | 0.429 | 8 / 10 |

This is a moderate stable core, not feature-level invariance. The block-31
result is again stronger, but six of its top 20 plain features and eight of
block 40's leave the Swahili top 20.

### Top ten plain features per block

Descriptions are Neuronpedia's unscored automatic explanations and are
exploratory labels, not validated feature identities.

| Block | Feature | Plain rank | Swahili rank | Neuronpedia description |
|---:|---:|---:|---:|---|
| 31 | [25137](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-65k/25137) | 1 | 2 | cannot or would not |
| 31 | [4637](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-65k/4637) | 2 | 1 | titles with explanations |
| 31 | [6360](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-65k/6360) | 3 | 5 | unethical, illegal or harmful actions |
| 31 | [2782](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-65k/2782) | 4 | 11 | explicit sexual content |
| 31 | [9707](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-65k/9707) | 5 | 13 | absolutely cannot and will not |
| 31 | [11229](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-65k/11229) | 6 | 10 | access restrictions |
| 31 | [3881](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-65k/3881) | 7 | 8 | “which” |
| 31 | [8423](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-65k/8423) | 8 | 6 | preventing attempts |
| 31 | [29404](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-65k/29404) | 9 | 44 | absolutely cannot and will not |
| 31 | [133](https://www.neuronpedia.org/gemma-3-27b-it/31-gemmascope-2-res-65k/133) | 10 | 15 | ethical dilemmas, manipulation and harm |
| 40 | [3748](https://www.neuronpedia.org/gemma-3-27b-it/40-gemmascope-2-res-65k/3748) | 1 | 1 | sexually explicit content |
| 40 | [1174](https://www.neuronpedia.org/gemma-3-27b-it/40-gemmascope-2-res-65k/1174) | 2 | 3 | malicious and harmful actions |
| 40 | [1630](https://www.neuronpedia.org/gemma-3-27b-it/40-gemmascope-2-res-65k/1630) | 3 | 7 | “even” |
| 40 | [13029](https://www.neuronpedia.org/gemma-3-27b-it/40-gemmascope-2-res-65k/13029) | 4 | 2 | “Okay, let's break down” |
| 40 | [698](https://www.neuronpedia.org/gemma-3-27b-it/40-gemmascope-2-res-65k/698) | 5 | 10 | cannot fulfil request |
| 40 | [10422](https://www.neuronpedia.org/gemma-3-27b-it/40-gemmascope-2-res-65k/10422) | 6 | 14 | how to do harmful things |
| 40 | [8346](https://www.neuronpedia.org/gemma-3-27b-it/40-gemmascope-2-res-65k/8346) | 7 | 8 | deception and falsehood |
| 40 | [11826](https://www.neuronpedia.org/gemma-3-27b-it/40-gemmascope-2-res-65k/11826) | 8 | 33 | “or” / “nor” |
| 40 | [7488](https://www.neuronpedia.org/gemma-3-27b-it/40-gemmascope-2-res-65k/7488) | 9 | 15 | lists of resource websites |
| 40 | [60](https://www.neuronpedia.org/gemma-3-27b-it/40-gemmascope-2-res-65k/60) | 10 | 49 | refusals and safety warnings |

The top features mix plausible harm concepts, refusal/safety-response features,
and generic formatting or function-word features. Since activations are taken
at the final instruction token, refusal-related features may encode the model's
anticipated safety response to a harmful request. The feature audit therefore
does not establish a pure harmfulness mechanism or a causal separation between
harm and refusal.

## SAE diagnostics

The first-batch reconstruction sanity check gave:

| Block | Explained variance | Relative MSE | Mean L0 |
|---:|---:|---:|---:|
| 31 | 0.707 | 0.293 | 51.8 |
| 40 | 0.469 | 0.531 | 47.0 |

Held-out empirical L0 also changes by condition:

| Block | Plain | Swahili | Reverse |
|---:|---:|---:|---:|
| 31 | 73.8 | 65.3 | 52.0 |
| 40 | 61.2 | 54.2 | 42.9 |

Block 40's weaker reconstruction and larger performance loss are consistent,
but this two-checkpoint result cannot establish that reconstruction quality
causes probe degradation. The L0 shift also means the sparse representation is
not distribution-invariant even where AUROC transfer remains strong.

## Defensible claims

The results support:

1. Medium-sparsity 65k Gemma Scope 2 features retain most dense
   plain/Swahili above-chance AUROC, especially at block 31.
2. Sparse probes are nevertheless worse at low-FPR detection, with a larger
   penalty under a frozen plain threshold than under matched calibration.
3. Plain and Swahili share a moderate high-contribution feature core
   (14/20 at block 31; 12/20 at block 40), but not invariant feature rankings.
4. The leading features contain recognisable harm and safety-related concepts,
   mixed with refusal and generic formatting features.
5. Block 31 is the better of the two preselected SAE checkpoints on
   reconstruction, probe performance and feature stability.

The results do not support:

- SAE probes outperforming or fully matching dense probes;
- feature-level invariance across surface forms;
- interpreting reverse retention above 1 at block 31 as SAE superiority;
- a pure harmfulness representation independent of refusal;
- causal or monosemantic interpretations of the top features; or
- selecting another layer, SAE sparsity, width, model or threshold using these
  test outcomes.

## Validation and provenance

The artifact contains 71 arrays and passed checks for finite scores and feature
contributions. Train/tune/test contain 5,341 / 1,781 / 1,781 unique IDs with no
cross-split overlap. Tune/test labels contain 569/1,212 and 568/1,213
harmful/benign examples. All four probes selected `C=0.001`.

- source commit: `e4ac3f2304b488b82d6f814efd834080b36d3da8`;
- MLP job: `3564811`, completed with exit code 0;
- runtime: 35m55s on 2 x RTX A6000; maximum RSS 53.37 GB;
- command:
  `python -m phase3.phase3_sae --batch-size 4 --out data/phase3_scores.npz`;
- analysis:
  `python -m phase3.analyse_phase3 --bootstrap 10000`;
- model revision:
  `005ad3404e59d6023443cb575daa05336842228a`;
- SAE revision:
  `5c58dd4cddd52cef653059d85e12a86bf6222a28`;
- SAE checkpoints:
  `resid_post/layer_{31,40}_width_65k_l0_medium`;
- seed: 0;
- frozen Swahili manifest SHA-256:
  `47e8f8c357ebed93d958bc85ec9bbe92ebb7720fe2d4f9105f6f221244ace994`;
- score artifact SHA-256:
  `2861acf2e3f0a6857f9114d183092c246dfb8775ce7d90e29c0c2d02b6aa8308`;
- structured results SHA-256:
  `528eaec6b4fd120f95083d6e4b5a80e54bfe93b3c0d49ba8bd4a6ae60a03c97c`;
- performance CSV SHA-256:
  `284091afa0e91dd0950cc6af078c93b1888ad1a377f2eabffb5fced6d4f24544`;
- feature CSV SHA-256:
  `e54c554b87a8773b610c8cbf97356eff5461a8738fdb3acecc6bcdc15cdc78a1`.

## Phase 3 decision

The predeclared stop condition is met: the two-layer dense-versus-SAE table,
paired uncertainty, retained above-chance AUROC, top-feature inspection and
plain/Swahili feature-stability result are complete. No SAE width, sparsity,
layer, model, transformation or interpretation sweep is required.
