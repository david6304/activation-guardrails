"""Create the frozen Phase 1 table and bootstrap intervals from the repo root."""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


CONDITIONS = ("plain", "swahili", "reverse")
DETECTORS = ("logistic", "centroid", "tfidf", "shieldgemma")


def threshold_at_one_percent(scores):
    return float(np.quantile(scores, 0.99, method="higher"))


def metrics(labels, scores, threshold):
    positive = labels == 1
    negative = labels == 0
    alerts = scores > threshold
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "tpr": float(alerts[positive].mean()),
        "fpr": float(alerts[negative].mean()),
        "threshold": float(threshold),
    }


def percentile_scores(reference, values):
    ordered = np.sort(reference)
    return np.searchsorted(ordered, values, side="right") / len(ordered)


def ensemble_scores(activation_reference, shield_reference, activation, shield):
    return np.maximum(
        percentile_scores(activation_reference, activation),
        percentile_scores(shield_reference, shield),
    )


def point_results(labels, tune_labels, activation, baselines):
    results = {"strict": {}, "matched": {}}
    point_scores = {"strict": {}, "matched": {}}
    point_thresholds = {"strict": {}, "matched": {}}
    tune_negative = tune_labels == 0
    for mode in ("strict", "matched"):
        for condition in CONDITIONS:
            calibration_condition = "plain" if mode == "strict" else condition
            results[mode][condition] = {}
            point_scores[mode][condition] = {}
            point_thresholds[mode][condition] = {}
            for detector in DETECTORS:
                source = activation if detector in {"logistic", "centroid"} else baselines
                reference = source[
                    f"tune_{calibration_condition}_{detector}"
                ][tune_negative]
                test_scores = source[f"test_{condition}_{detector}"]
                threshold = threshold_at_one_percent(reference)
                results[mode][condition][detector] = metrics(
                    labels, test_scores, threshold
                )
                point_scores[mode][condition][detector] = test_scores
                point_thresholds[mode][condition][detector] = threshold

            activation_reference = activation[
                f"tune_{calibration_condition}_logistic"
            ][tune_negative]
            shield_reference = baselines[
                f"tune_{calibration_condition}_shieldgemma"
            ][tune_negative]
            reference_ensemble = ensemble_scores(
                activation_reference,
                shield_reference,
                activation_reference,
                shield_reference,
            )
            test_ensemble = ensemble_scores(
                activation_reference,
                shield_reference,
                activation[f"test_{condition}_logistic"],
                baselines[f"test_{condition}_shieldgemma"],
            )
            threshold = threshold_at_one_percent(reference_ensemble)
            results[mode][condition]["ensemble"] = metrics(
                labels, test_ensemble, threshold
            )
            point_scores[mode][condition]["ensemble"] = test_ensemble
            point_thresholds[mode][condition]["ensemble"] = threshold
    return results, point_scores, point_thresholds


def bootstrap_primary(
    labels,
    tune_labels,
    activation,
    baselines,
    condition,
    mode,
    repeats,
    seed,
):
    """Paired resampling of calibration negatives and matched test examples."""
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    tune_negative = np.flatnonzero(tune_labels == 0)
    calibration_condition = "plain" if mode == "strict" else condition
    detector_names = (*DETECTORS, "ensemble")
    tprs = {name: np.empty(repeats) for name in detector_names}
    fprs = {name: np.empty(repeats) for name in detector_names}

    for repeat in range(repeats):
        calibration_sample = rng.choice(
            tune_negative, size=len(tune_negative), replace=True
        )
        positive_sample = rng.choice(positive, size=len(positive), replace=True)
        negative_sample = rng.choice(negative, size=len(negative), replace=True)
        for detector in DETECTORS:
            source = activation if detector in {"logistic", "centroid"} else baselines
            reference = source[f"tune_{calibration_condition}_{detector}"][
                calibration_sample
            ]
            threshold = threshold_at_one_percent(reference)
            test_scores = source[f"test_{condition}_{detector}"]
            tprs[detector][repeat] = np.mean(test_scores[positive_sample] > threshold)
            fprs[detector][repeat] = np.mean(test_scores[negative_sample] > threshold)

        activation_reference = activation[
            f"tune_{calibration_condition}_logistic"
        ][calibration_sample]
        shield_reference = baselines[
            f"tune_{calibration_condition}_shieldgemma"
        ][calibration_sample]
        reference_ensemble = ensemble_scores(
            activation_reference,
            shield_reference,
            activation_reference,
            shield_reference,
        )
        threshold = threshold_at_one_percent(reference_ensemble)
        test_ensemble = ensemble_scores(
            activation_reference,
            shield_reference,
            activation[f"test_{condition}_logistic"],
            baselines[f"test_{condition}_shieldgemma"],
        )
        tprs["ensemble"][repeat] = np.mean(
            test_ensemble[positive_sample] > threshold
        )
        fprs["ensemble"][repeat] = np.mean(
            test_ensemble[negative_sample] > threshold
        )

    intervals = {}
    for detector in detector_names:
        intervals[detector] = {
            "tpr_95ci": np.quantile(tprs[detector], [0.025, 0.975]).tolist(),
            "fpr_95ci": np.quantile(fprs[detector], [0.025, 0.975]).tolist(),
        }
    differences = {}
    for left, right in (
        ("logistic", "shieldgemma"),
        ("ensemble", "logistic"),
        ("ensemble", "shieldgemma"),
    ):
        delta = tprs[left] - tprs[right]
        differences[f"{left}_minus_{right}"] = {
            "mean": float(delta.mean()),
            "95ci": np.quantile(delta, [0.025, 0.975]).tolist(),
        }
    return intervals, differences


def error_overlap(labels, scores, thresholds):
    positive = labels == 1
    alerts = {
        detector: detector_scores > thresholds[detector]
        for detector, detector_scores in scores.items()
    }
    activation = alerts["logistic"][positive]
    shield = alerts["shieldgemma"][positive]
    ensemble = alerts["ensemble"][positive]
    return {
        "harmful_n": int(positive.sum()),
        "both_activation_and_shield": int(np.sum(activation & shield)),
        "activation_only": int(np.sum(activation & ~shield)),
        "shield_only": int(np.sum(~activation & shield)),
        "neither": int(np.sum(~activation & ~shield)),
        "ensemble_detected": int(ensemble.sum()),
        "ensemble_rescued_over_activation": int(np.sum(ensemble & ~activation)),
        "ensemble_rescued_over_shield": int(np.sum(ensemble & ~shield)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", default="data/phase1_activation_27b.npz")
    parser.add_argument("--baselines", default="data/phase1_baselines.npz")
    parser.add_argument("--out", default="data/phase1_results.json")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with np.load(args.activation, allow_pickle=False) as activation_file:
        activation = {key: activation_file[key] for key in activation_file.files}
    with np.load(args.baselines, allow_pickle=False) as baseline_file:
        baselines = {key: baseline_file[key] for key in baseline_file.files}
    if not np.array_equal(activation["tune_ids"], baselines["tune_ids"]):
        raise ValueError("tune IDs differ")
    if not np.array_equal(activation["test_ids"], baselines["test_ids"]):
        raise ValueError("test IDs differ")

    labels = activation["test_labels"]
    tune_labels = activation["tune_labels"]
    results, point_scores, point_thresholds = point_results(
        labels, tune_labels, activation, baselines
    )
    bootstrap = {}
    differences = {}
    overlap = {}
    for mode in ("strict", "matched"):
        bootstrap[mode] = {}
        differences[mode] = {}
        overlap[mode] = {}
        for condition in ("swahili", "reverse"):
            bootstrap[mode][condition], differences[mode][condition] = bootstrap_primary(
                labels,
                tune_labels,
                activation,
                baselines,
                condition,
                mode,
                args.bootstrap,
                args.seed,
            )
            overlap[mode][condition] = error_overlap(
                labels,
                point_scores[mode][condition],
                point_thresholds[mode][condition],
            )

    wildguard = {}
    for condition in CONDITIONS:
        flags = baselines[f"test_{condition}_wildguard"]
        wildguard[condition] = {
            "tpr": float(flags[labels == 1].mean()),
            "fpr": float(flags[labels == 0].mean()),
        }

    report = {
        "operating_point": "threshold from tune negatives at 1% FPR",
        "bootstrap_repeats": args.bootstrap,
        "results": results,
        "bootstrap": bootstrap,
        "paired_tpr_differences": differences,
        "error_overlap": overlap,
        "wildguard_fixed_decision": wildguard,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")

    for mode in ("strict", "matched"):
        print(f"\n[{mode}]")
        for condition in CONDITIONS:
            print(f"  {condition}")
            for detector, result in results[mode][condition].items():
                print(
                    f"    {detector:12s} AUROC={result['auroc']:.4f} "
                    f"TPR={result['tpr']:.3%} FPR={result['fpr']:.3%}"
                )
    print("\n[WildGuard fixed decision]")
    for condition, result in wildguard.items():
        print(
            f"  {condition:8s} TPR={result['tpr']:.3%} FPR={result['fpr']:.3%}"
        )
    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
