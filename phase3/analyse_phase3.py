"""Analyse the frozen two-layer dense-versus-SAE Phase 3 artefact."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


BLOCKS = (31, 40)
REPRESENTATIONS = ("dense", "sae")
CONDITIONS = ("plain", "swahili", "reverse")


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def threshold(scores):
    return float(np.quantile(scores, 0.99, method="higher"))


def metrics(labels, scores, cutoff):
    positive = labels == 1
    negative = labels == 0
    alerts = scores > cutoff
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "tpr": float(alerts[positive].mean()),
        "fpr": float(alerts[negative].mean()),
        "threshold": cutoff,
    }


def validate(source):
    if str(source["position"]) != "t_inst":
        raise ValueError("Phase 3 artefact is not at t_inst")
    for block in BLOCKS:
        for representation in REPRESENTATIONS:
            for split in ("tune", "test"):
                for condition in CONDITIONS:
                    key = f"{split}_{condition}_{representation}_block{block}"
                    if key not in source or not np.isfinite(source[key]).all():
                        raise ValueError(f"missing or non-finite {key}")
        for condition in ("plain", "swahili"):
            values = source[f"contribution_{condition}_block{block}"]
            if values.shape != (65536,) or not np.isfinite(values).all():
                raise ValueError(f"invalid block {block} {condition} contributions")


def point_results(source):
    labels = source["test_labels"]
    tune_labels = source["tune_labels"]
    tune_negative = tune_labels == 0
    results = {}
    for block in BLOCKS:
        results[str(block)] = {}
        dense_aurocs = {}
        for representation in REPRESENTATIONS:
            results[str(block)][representation] = {}
            for condition in CONDITIONS:
                strict_reference = source[
                    f"tune_plain_{representation}_block{block}"
                ][tune_negative]
                matched_reference = source[
                    f"tune_{condition}_{representation}_block{block}"
                ][tune_negative]
                scores = source[f"test_{condition}_{representation}_block{block}"]
                strict = metrics(labels, scores, threshold(strict_reference))
                matched = metrics(labels, scores, threshold(matched_reference))
                results[str(block)][representation][condition] = {
                    "strict": strict,
                    "matched": matched,
                }
                if representation == "dense":
                    dense_aurocs[condition] = strict["auroc"]
        for condition in CONDITIONS:
            sae = results[str(block)]["sae"][condition]["strict"]["auroc"]
            denominator = dense_aurocs[condition] - 0.5
            retained = (sae - 0.5) / denominator if denominator != 0 else None
            results[str(block)]["sae"][condition]["retained_above_chance_auroc"] = (
                retained
            )
            results[str(block)]["sae"][condition]["retention_well_conditioned"] = (
                abs(denominator) >= 0.02
            )
    return results


def bootstrap(source, block, condition, repeats, seed):
    labels = source["test_labels"]
    tune_labels = source["tune_labels"]
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    tune_negative = np.flatnonzero(tune_labels == 0)
    rng = np.random.default_rng(seed)
    values = {
        "auroc_sae_minus_dense": np.empty(repeats),
        "strict_tpr_sae_minus_dense": np.empty(repeats),
        "matched_tpr_sae_minus_dense": np.empty(repeats),
        "retained_above_chance_auroc": np.empty(repeats),
    }
    for repeat in range(repeats):
        calibration = rng.choice(tune_negative, len(tune_negative), replace=True)
        positive_sample = rng.choice(positive, len(positive), replace=True)
        negative_sample = rng.choice(negative, len(negative), replace=True)
        sample = np.concatenate((positive_sample, negative_sample))
        sample_labels = labels[sample]
        metrics_by_representation = {}
        for representation in REPRESENTATIONS:
            test_scores = source[
                f"test_{condition}_{representation}_block{block}"
            ]
            strict_cutoff = threshold(
                source[f"tune_plain_{representation}_block{block}"][calibration]
            )
            matched_cutoff = threshold(
                source[f"tune_{condition}_{representation}_block{block}"][
                    calibration
                ]
            )
            metrics_by_representation[representation] = {
                "auroc": roc_auc_score(sample_labels, test_scores[sample]),
                "strict_tpr": np.mean(
                    test_scores[positive_sample] > strict_cutoff
                ),
                "matched_tpr": np.mean(
                    test_scores[positive_sample] > matched_cutoff
                ),
            }
        dense = metrics_by_representation["dense"]
        sae = metrics_by_representation["sae"]
        values["auroc_sae_minus_dense"][repeat] = sae["auroc"] - dense["auroc"]
        values["strict_tpr_sae_minus_dense"][repeat] = (
            sae["strict_tpr"] - dense["strict_tpr"]
        )
        values["matched_tpr_sae_minus_dense"][repeat] = (
            sae["matched_tpr"] - dense["matched_tpr"]
        )
        denominator = dense["auroc"] - 0.5
        values["retained_above_chance_auroc"][repeat] = (
            (sae["auroc"] - 0.5) / denominator if denominator != 0 else np.nan
        )
    return {
        metric: {
            "mean": float(np.nanmean(samples)),
            "95ci": np.nanquantile(samples, [0.025, 0.975]).tolist(),
        }
        for metric, samples in values.items()
    }


def feature_stability(source):
    examples = json.loads(str(source["top_examples_json"]))
    results = {}
    feature_rows = []
    for block in BLOCKS:
        plain = source[f"contribution_plain_block{block}"]
        swahili = source[f"contribution_swahili_block{block}"]
        plain_order = np.argsort(np.abs(plain))[::-1]
        swahili_order = np.argsort(np.abs(swahili))[::-1]
        plain_top = set(map(int, plain_order[:20]))
        swahili_top = set(map(int, swahili_order[:20]))
        overlap = plain_top & swahili_top
        results[str(block)] = {
            "top_k": 20,
            "overlap_n": len(overlap),
            "jaccard": len(overlap) / len(plain_top | swahili_top),
            "overlap_features": sorted(overlap),
        }
        swahili_ranks = np.empty(len(swahili_order), dtype=np.int32)
        swahili_ranks[swahili_order] = np.arange(1, len(swahili_order) + 1)
        weight = source[f"sae_weight_block{block}"]
        for plain_rank, feature in enumerate(plain_order[:10], start=1):
            feature = int(feature)
            feature_rows.append(
                {
                    "block": block,
                    "feature": feature,
                    "plain_rank": plain_rank,
                    "swahili_rank": int(swahili_ranks[feature]),
                    "in_swahili_top20": feature in swahili_top,
                    "weight": float(weight[feature]),
                    "plain_contribution": float(plain[feature]),
                    "swahili_contribution": float(swahili[feature]),
                    "neuronpedia": (
                        f"https://www.neuronpedia.org/gemma-3-27b-it/"
                        f"{block}-gemmascope-2-res-65k/{feature}"
                    ),
                    "top_examples": examples[str(block)][str(feature)],
                }
            )
    return results, feature_rows


def write_performance_csv(path, results):
    fields = (
        "block",
        "representation",
        "condition",
        "auroc",
        "strict_tpr",
        "strict_fpr",
        "matched_tpr",
        "matched_fpr",
        "retained_above_chance_auroc",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for block in BLOCKS:
            for representation in REPRESENTATIONS:
                for condition in CONDITIONS:
                    row = results[str(block)][representation][condition]
                    writer.writerow(
                        {
                            "block": block,
                            "representation": representation,
                            "condition": condition,
                            "auroc": row["strict"]["auroc"],
                            "strict_tpr": row["strict"]["tpr"],
                            "strict_fpr": row["strict"]["fpr"],
                            "matched_tpr": row["matched"]["tpr"],
                            "matched_fpr": row["matched"]["fpr"],
                            "retained_above_chance_auroc": row.get(
                                "retained_above_chance_auroc"
                            ),
                        }
                    )


def write_features_csv(path, rows):
    fields = (
        "block",
        "feature",
        "plain_rank",
        "swahili_rank",
        "in_swahili_top20",
        "weight",
        "plain_contribution",
        "swahili_contribution",
        "neuronpedia",
        "top_examples",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "top_examples": json.dumps(row["top_examples"])})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="data/phase3_scores.npz")
    parser.add_argument("--out", default="data/phase3_results.json")
    parser.add_argument("--csv", default="data/phase3_results.csv")
    parser.add_argument("--features-csv", default="data/phase3_features.csv")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    source = load_npz(args.scores)
    validate(source)
    results = point_results(source)
    intervals = {}
    for block_index, block in enumerate(BLOCKS):
        intervals[str(block)] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            intervals[str(block)][condition] = bootstrap(
                source,
                block,
                condition,
                args.bootstrap,
                args.seed + 10 * block_index + condition_index,
            )
    stability, feature_rows = feature_stability(source)
    report = {
        "model": str(source["model"]),
        "model_revision": str(source["model_revision"]),
        "sae_repo": str(source["sae_repo"]),
        "sae_revision": str(source["sae_revision"]),
        "position": str(source["position"]),
        "operating_point": "threshold from tune negatives at 1% FPR",
        "bootstrap_repeats": args.bootstrap,
        "results": results,
        "paired_sae_minus_dense": intervals,
        "feature_stability": stability,
        "sae_validation": json.loads(str(source["sae_validation_json"])),
        "empirical_l0": json.loads(str(source["l0_json"])),
        "top_features": feature_rows,
    }
    output_path = Path(args.out)
    csv_path = Path(args.csv)
    features_path = Path(args.features_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    write_performance_csv(csv_path, results)
    write_features_csv(features_path, feature_rows)

    for block in BLOCKS:
        print(f"\n[block {block}]")
        for condition in CONDITIONS:
            dense = results[str(block)]["dense"][condition]
            sae = results[str(block)]["sae"][condition]
            print(
                f"  {condition:7s} dense AUROC={dense['strict']['auroc']:.4f} "
                f"matched={dense['matched']['tpr']:.3%}/{dense['matched']['fpr']:.3%}; "
                f"SAE AUROC={sae['strict']['auroc']:.4f} "
                f"matched={sae['matched']['tpr']:.3%}/{sae['matched']['fpr']:.3%} "
                f"R={sae['retained_above_chance_auroc']:.3f}"
            )
        feature_result = stability[str(block)]
        print(
            f"  feature top-20 overlap={feature_result['overlap_n']}/20 "
            f"Jaccard={feature_result['jaccard']:.3f}"
        )
    print(f"[done] {output_path}")
    print(f"[done] {csv_path}")
    print(f"[done] {features_path}")


if __name__ == "__main__":
    main()
