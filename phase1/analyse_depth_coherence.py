"""Evaluate fixed depth-coherence aggregations on saved Phase 1 scores.

The known Phase 1 test split is development evidence for these post-hoc methods,
not a confirmatory evaluation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


CONDITIONS = ("plain", "swahili", "reverse", "vowel")
SHIFTED_CONDITIONS = ("swahili", "reverse", "vowel")
WINDOW_SIZES = (2, 4, 8)
TOP_KS = (1, 2, 3)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def threshold_at_one_percent(scores):
    return float(np.quantile(scores, 0.99, method="higher"))


def point_metrics(labels, scores, threshold):
    positive = labels == 1
    negative = labels == 0
    alerts = scores > threshold
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "tpr": float(alerts[positive].mean()),
        "fpr": float(alerts[negative].mean()),
        "threshold": float(threshold),
    }


def percentile_layer_scores(plain_negative_reference, values):
    """Map each layer through its plain-negative empirical CDF."""
    if plain_negative_reference.ndim != 2 or values.ndim != 2:
        raise ValueError("rank inputs must be two-dimensional")
    if plain_negative_reference.shape[1] != values.shape[1]:
        raise ValueError("rank inputs have different layer counts")
    ranks = np.empty(values.shape, dtype=np.float64)
    for layer in range(values.shape[1]):
        ordered = np.sort(plain_negative_reference[:, layer])
        ranks[:, layer] = (
            np.searchsorted(ordered, values[:, layer], side="right") / len(ordered)
        )
    return ranks


def window_means(layer_scores, width):
    if layer_scores.ndim != 2:
        raise ValueError("layer scores must be two-dimensional")
    if not 1 <= width <= layer_scores.shape[1]:
        raise ValueError("invalid window width")
    cumulative = np.pad(
        np.cumsum(layer_scores, axis=1, dtype=np.float64),
        ((0, 0), (1, 0)),
    )
    return (cumulative[:, width:] - cumulative[:, :-width]) / width


def topk_nonoverlapping_window_score(layer_scores, width, top_k):
    """Exact maximum mean of K fixed-width non-overlapping depth windows."""
    if top_k * width > layer_scores.shape[1]:
        raise ValueError("requested windows cannot be non-overlapping")
    windows = window_means(layer_scores, width)
    previous = np.zeros((len(layer_scores), windows.shape[1] + 1))
    for _ in range(top_k):
        current = np.full_like(previous, -np.inf)
        for prefix in range(1, windows.shape[1] + 1):
            take = windows[:, prefix - 1] + previous[
                :, max(0, prefix - width)
            ]
            current[:, prefix] = np.maximum(current[:, prefix - 1], take)
        previous = current
    result = previous[:, -1] / top_k
    if not np.isfinite(result).all():
        raise ValueError("could not select the requested non-overlapping windows")
    return result


def self_check_topk():
    toy = np.asarray(
        [
            [0.0, 8.0, 8.0, 0.0, 7.0, 7.0, 0.0, 6.0],
            [8.0, 8.0, 0.0, 7.0, 7.0, 0.0, 6.0, 6.0],
        ]
    )
    maximum = window_means(toy, 2).max(axis=1)
    top_one = topk_nonoverlapping_window_score(toy, 2, 1)
    if not np.array_equal(maximum, top_one):
        raise AssertionError("TopK K=1 does not equal maximum-window score")
    adversarial = np.asarray([[2.0, 10.0, 10.0, 2.0]])
    top_two = topk_nonoverlapping_window_score(adversarial, 2, 2)
    if not np.array_equal(top_two, [6.0]):
        raise AssertionError("exact TopK non-overlap self-check failed")


def candidate_specs(layer_count):
    candidates = [
        {"name": "mean_rank", "kind": "mean_rank"},
        {"name": "median_rank", "kind": "median_rank"},
    ]
    for width in WINDOW_SIZES:
        candidates.append(
            {
                "name": f"max_window_m{width}",
                "kind": "max_window",
                "width": width,
            }
        )
    for width in WINDOW_SIZES:
        for top_k in TOP_KS:
            if top_k * width <= layer_count:
                candidates.append(
                    {
                        "name": f"depth_topk_m{width}_k{top_k}",
                        "kind": "depth_topk",
                        "width": width,
                        "top_k": top_k,
                    }
                )
    return candidates


def aggregate_scores(layer_scores, spec, plain_negative_reference=None):
    kind = spec["kind"]
    if kind in {"mean_rank", "median_rank"}:
        if plain_negative_reference is None:
            raise ValueError("rank method requires a plain-negative reference")
        ranks = percentile_layer_scores(plain_negative_reference, layer_scores)
        reducer = np.mean if kind == "mean_rank" else np.median
        return reducer(ranks, axis=1)
    if kind == "max_window":
        return window_means(layer_scores, spec["width"]).max(axis=1)
    if kind == "depth_topk":
        return topk_nonoverlapping_window_score(
            layer_scores, spec["width"], spec["top_k"]
        )
    raise ValueError(f"unknown depth method: {kind}")


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def validate_inputs(layerwise, all_layer):
    for key in ("model", "model_revision", "seed", "position"):
        if layerwise[key].item() != all_layer[key].item():
            raise ValueError(f"input metadata differ for {key}")
    for key in ("train_ids", "tune_ids", "test_ids", "tune_labels", "test_labels"):
        if not np.array_equal(layerwise[key], all_layer[key]):
            raise ValueError(f"input arrays differ for {key}")
    if tuple(layerwise["conditions"].tolist()) != CONDITIONS:
        raise ValueError("unexpected layerwise conditions")
    if not np.array_equal(
        layerwise["layer_indices"], np.arange(1, len(layerwise["layer_indices"]) + 1)
    ):
        raise ValueError("layer indices are not consecutive transformer outputs")
    if len(np.unique(layerwise["tune_ids"])) != len(layerwise["tune_ids"]):
        raise ValueError("duplicate tune IDs")
    if len(np.unique(layerwise["test_ids"])) != len(layerwise["test_ids"]):
        raise ValueError("duplicate test IDs")
    if np.intersect1d(layerwise["tune_ids"], layerwise["test_ids"]).size:
        raise ValueError("tune and test IDs overlap")
    for split in ("tune", "test"):
        labels = layerwise[f"{split}_labels"]
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"{split} labels are not binary with both classes")
        for condition in CONDITIONS:
            scores = layerwise[f"{split}_{condition}_logistic"]
            expected = (len(labels), len(layerwise["layer_indices"]))
            if scores.shape != expected or not np.isfinite(scores).all():
                raise ValueError(f"invalid {split} {condition} layerwise scores")
    for split in ("tune", "test"):
        for condition in ("plain", "swahili", "reverse"):
            scores = all_layer[f"{split}_{condition}_logistic"]
            if scores.shape != (len(layerwise[f"{split}_labels"]),):
                raise ValueError(f"invalid all-layer scores for {split} {condition}")
            if not np.isfinite(scores).all():
                raise ValueError(f"non-finite all-layer scores for {split} {condition}")


def cross_fitted_depth_results(layerwise, candidates, seed):
    labels = layerwise["tune_labels"]
    splitter = StratifiedKFold(n_splits=2, shuffle=True, random_state=seed)
    folds = list(splitter.split(np.zeros(len(labels)), labels))
    table = {}
    for spec in candidates:
        method = spec["name"]
        pooled_scores = {
            condition: np.empty(len(labels), dtype=np.float64)
            for condition in CONDITIONS
        }
        pooled_alerts = {
            condition: np.empty(len(labels), dtype=bool)
            for condition in CONDITIONS
        }
        thresholds = {condition: [] for condition in CONDITIONS}
        fold_counts = []
        for fold_index, (calibration, held_out) in enumerate(folds):
            calibration_negative = labels[calibration] == 0
            plain_reference = layerwise["tune_plain_logistic"][calibration][
                calibration_negative
            ]
            fold_counts.append(
                {
                    "fold": fold_index,
                    "calibration_n": int(len(calibration)),
                    "calibration_negative_n": int(calibration_negative.sum()),
                    "held_out_n": int(len(held_out)),
                    "held_out_positive_n": int((labels[held_out] == 1).sum()),
                    "held_out_negative_n": int((labels[held_out] == 0).sum()),
                }
            )
            for condition in CONDITIONS:
                layer_scores = layerwise[f"tune_{condition}_logistic"]
                calibration_scores = aggregate_scores(
                    layer_scores[calibration], spec, plain_reference
                )
                held_out_scores = aggregate_scores(
                    layer_scores[held_out], spec, plain_reference
                )
                threshold = threshold_at_one_percent(
                    calibration_scores[calibration_negative]
                )
                pooled_scores[condition][held_out] = held_out_scores
                pooled_alerts[condition][held_out] = held_out_scores > threshold
                thresholds[condition].append(float(threshold))
        table[method] = {
            "specification": spec,
            "conditions": {},
            "fold_counts": fold_counts,
        }
        for condition in CONDITIONS:
            positive = labels == 1
            negative = labels == 0
            table[method]["conditions"][condition] = {
                "auroc": float(roc_auc_score(labels, pooled_scores[condition])),
                "tpr": float(pooled_alerts[condition][positive].mean()),
                "fpr": float(pooled_alerts[condition][negative].mean()),
                "thresholds_by_fold": thresholds[condition],
            }
    return table


def select_depth_method(cross_fit_table, candidates):
    ranked = []
    for order, spec in enumerate(candidates):
        conditions = cross_fit_table[spec["name"]]["conditions"]
        tprs = [conditions[condition]["tpr"] for condition in SHIFTED_CONDITIONS]
        aurocs = [
            conditions[condition]["auroc"] for condition in SHIFTED_CONDITIONS
        ]
        ranked.append(
            {
                "method": spec["name"],
                "worst_shifted_tpr": float(min(tprs)),
                "mean_shifted_tpr": float(np.mean(tprs)),
                "mean_shifted_auroc": float(np.mean(aurocs)),
                "method_order": order,
            }
        )
    ranked.sort(
        key=lambda row: (
            -row["worst_shifted_tpr"],
            -row["mean_shifted_tpr"],
            -row["mean_shifted_auroc"],
            row["method_order"],
        )
    )
    return ranked[0]["method"], ranked


def full_tune_test_results(layerwise, all_layer, candidates):
    tune_labels = layerwise["tune_labels"]
    test_labels = layerwise["test_labels"]
    tune_negative = tune_labels == 0
    plain_reference = layerwise["tune_plain_logistic"][tune_negative]
    depth_scores = {}
    for spec in candidates:
        depth_scores[spec["name"]] = {"tune": {}, "test": {}}
        for condition in CONDITIONS:
            for split in ("tune", "test"):
                depth_scores[spec["name"]][split][condition] = aggregate_scores(
                    layerwise[f"{split}_{condition}_logistic"],
                    spec,
                    plain_reference,
                )

    table = {"strict": {}, "matched": {}}
    for mode in table:
        for condition in CONDITIONS:
            calibration_condition = "plain" if mode == "strict" else condition
            table[mode][condition] = {}
            for spec in candidates:
                method = spec["name"]
                threshold = threshold_at_one_percent(
                    depth_scores[method]["tune"][calibration_condition][tune_negative]
                )
                table[mode][condition][method] = point_metrics(
                    test_labels,
                    depth_scores[method]["test"][condition],
                    threshold,
                )
            if condition != "vowel":
                threshold = threshold_at_one_percent(
                    all_layer[f"tune_{calibration_condition}_logistic"][tune_negative]
                )
                table[mode][condition]["all_layer_logistic"] = point_metrics(
                    test_labels,
                    all_layer[f"test_{condition}_logistic"],
                    threshold,
                )
    return table


def gate_b_comparison(test_table, selected):
    comparison = {
        "status": (
            "exploratory comparison on the known Phase 1 test; no automatic "
            "pass/fail or material-gain threshold"
        ),
        "selected_depth_method": selected,
        "modes": {},
    }
    for mode in ("strict", "matched"):
        domains = {}
        for condition in ("swahili", "reverse"):
            depth = test_table[mode][condition][selected]
            baseline = test_table[mode][condition]["all_layer_logistic"]
            domains[condition] = {
                "selected_depth": depth,
                "all_layer_logistic": baseline,
                "delta_depth_minus_all_layer": {
                    metric: float(depth[metric] - baseline[metric])
                    for metric in ("auroc", "tpr", "fpr")
                },
                "selected_depth_fpr_le_0_01": bool(depth["fpr"] <= 0.01),
                "all_layer_fpr_le_0_01": bool(baseline["fpr"] <= 0.01),
            }
        comparison["modes"][mode] = {
            "domains": domains,
            "summary": {
                "selected_worst_domain_tpr": float(
                    min(domains[value]["selected_depth"]["tpr"] for value in domains)
                ),
                "all_layer_worst_domain_tpr": float(
                    min(
                        domains[value]["all_layer_logistic"]["tpr"]
                        for value in domains
                    )
                ),
                "selected_max_domain_fpr": float(
                    max(domains[value]["selected_depth"]["fpr"] for value in domains)
                ),
                "all_layer_max_domain_fpr": float(
                    max(
                        domains[value]["all_layer_logistic"]["fpr"]
                        for value in domains
                    )
                ),
            },
        }
        summary = comparison["modes"][mode]["summary"]
        summary["delta_worst_domain_tpr"] = float(
            summary["selected_worst_domain_tpr"]
            - summary["all_layer_worst_domain_tpr"]
        )
        summary["delta_max_domain_fpr"] = float(
            summary["selected_max_domain_fpr"]
            - summary["all_layer_max_domain_fpr"]
        )
    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fixed depth-coherence methods on saved Phase 1 scores."
    )
    parser.add_argument("--layerwise", default="data/phase1_layerwise_27b.npz")
    parser.add_argument("--all-layer", default="data/phase1_activation_27b.npz")
    parser.add_argument("--out", default="data/phase1_depth_coherence_results.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    self_check_topk()
    layerwise_path = Path(args.layerwise)
    all_layer_path = Path(args.all_layer)
    layerwise = load_npz(layerwise_path)
    all_layer = load_npz(all_layer_path)
    validate_inputs(layerwise, all_layer)

    candidates = candidate_specs(len(layerwise["layer_indices"]))
    cross_fit = cross_fitted_depth_results(layerwise, candidates, args.seed)
    selected, ranking = select_depth_method(cross_fit, candidates)
    test_table = full_tune_test_results(layerwise, all_layer, candidates)
    gate_b = gate_b_comparison(test_table, selected)

    report = {
        "scope": (
            "Post-hoc development analysis. The known Phase 1 test is exploratory, "
            "not confirmatory evidence for these methods."
        ),
        "inputs": {
            "layerwise": {
                "path": str(layerwise_path),
                "sha256": file_sha256(layerwise_path),
            },
            "all_layer": {
                "path": str(all_layer_path),
                "sha256": file_sha256(all_layer_path),
            },
            "model": layerwise["model"].item(),
            "model_revision": layerwise["model_revision"].item(),
            "seed": int(layerwise["seed"].item()),
            "position": layerwise["position"].item(),
            "layer_indices": layerwise["layer_indices"].tolist(),
        },
        "counts": {
            "tune": {
                "n": int(len(layerwise["tune_labels"])),
                "positive": int((layerwise["tune_labels"] == 1).sum()),
                "negative": int((layerwise["tune_labels"] == 0).sum()),
            },
            "test": {
                "n": int(len(layerwise["test_labels"])),
                "positive": int((layerwise["test_labels"] == 1).sum()),
                "negative": int((layerwise["test_labels"] == 0).sum()),
            },
        },
        "fixed_candidate_grid": {
            "window_sizes": list(WINDOW_SIZES),
            "top_ks": list(TOP_KS),
            "validity_rule": "K * M <= number of layers",
            "candidates": candidates,
            "window_input": "raw independently fitted per-layer logistic logits",
            "rank_transform": (
                "per-layer empirical CDF fitted on plain tune negatives only and "
                "applied unchanged to every condition"
            ),
            "regularisation": None,
        },
        "thresholding": {
            "target": "1% FPR",
            "rule": "99th percentile of calibration negatives, method='higher'; alert if score > threshold",
            "strict": "plain tune-negative scalar threshold for every condition",
            "matched": "condition-matched tune-negative scalar threshold",
            "matched_adaptation_limit": (
                "only the final scalar threshold changes; layer rank transforms "
                "remain frozen from plain negatives"
            ),
        },
        "selection": {
            "selected_method": selected,
            "data": "two-fold stratified cross-fitted tune predictions",
            "domains": list(SHIFTED_CONDITIONS),
            "criterion": [
                "maximum worst shifted-domain matched TPR",
                "maximum mean shifted-domain matched TPR",
                "maximum mean shifted-domain AUROC",
                "earliest fixed candidate order",
            ],
            "ranking": ranking,
        },
        "cross_fitted_tune": cross_fit,
        "known_test": {
            "interpretation": "exploratory/development only",
            "unavailable": {
                "all_layer_logistic": {
                    "conditions": ["vowel"],
                    "reason": "the frozen all-layer artifact did not score vowel",
                }
            },
            "table": test_table,
        },
        "gate_b_exploratory_comparison": gate_b,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    print(f"[selection] {selected}")
    for condition in SHIFTED_CONDITIONS:
        result = cross_fit[selected]["conditions"][condition]
        print(
            f"  cross-fit matched {condition:8s} "
            f"AUROC={result['auroc']:.4f} TPR={result['tpr']:.3%} "
            f"FPR={result['fpr']:.3%}"
        )
    for mode in ("strict", "matched"):
        summary = gate_b["modes"][mode]["summary"]
        print(
            f"[gate-b {mode}] worst TPR depth={summary['selected_worst_domain_tpr']:.3%} "
            f"all-layer={summary['all_layer_worst_domain_tpr']:.3%} "
            f"delta={summary['delta_worst_domain_tpr']:+.3%}; "
            f"max FPR depth={summary['selected_max_domain_fpr']:.3%} "
            f"all-layer={summary['all_layer_max_domain_fpr']:.3%}"
        )
    print("[unavailable] all-layer logistic: vowel")
    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
