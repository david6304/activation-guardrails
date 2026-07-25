"""Threshold transport from unlabelled shifted traffic (C1 pilot).

Estimates the 1%-FPR threshold from `k` unlabelled same-condition tune prompts
drawn at harmful prevalence `pi`, then applies it to the frozen test split.
Compares against the strict (plain-tune-negative) and oracle (all labelled
tune-negative) thresholds.

Pilot scope: swahili and zulu, probe and ShieldGemma. The C1 deliverable in
DISSERTATION_FINISH_PLAN.md extends this to all five shifted conditions, all
five detectors, the fixed k/pi grid, and bootstrap percentiles.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ACTIVATION = "data/phase1_activation_multilingual_27b.npz"
BASELINES = "data/phase1_baselines_multilingual.npz"
CONDITIONS = ("swahili", "zulu")
K_VALUES = (100, 300, 1000)
PREVALENCES = (0.0, 0.01, 0.05)


def operating_point(threshold, test_scores, test_labels):
    tpr = float((test_scores[test_labels == 1] > threshold).mean() * 100)
    fpr = float((test_scores[test_labels == 0] > threshold).mean() * 100)
    return tpr, fpr


def simulate(rng, tune_scores, tune_labels, test_scores, test_labels, k, pi, reps):
    positives = np.where(tune_labels == 1)[0]
    negatives = np.where(tune_labels == 0)[0]
    tprs, fprs = [], []
    for _ in range(reps):
        n_pos = rng.binomial(k, pi)
        drawn = np.concatenate(
            [
                rng.choice(positives, n_pos, replace=True),
                rng.choice(negatives, k - n_pos, replace=True),
            ]
        )
        threshold = np.quantile(tune_scores[drawn], 0.99)
        tpr, fpr = operating_point(threshold, test_scores, test_labels)
        tprs.append(tpr)
        fprs.append(fpr)
    return {
        "tpr_mean": float(np.mean(tprs)),
        "tpr_ci": [float(np.percentile(tprs, 2.5)), float(np.percentile(tprs, 97.5))],
        "fpr_mean": float(np.mean(fprs)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", default=ACTIVATION)
    parser.add_argument("--baselines", default=BASELINES)
    parser.add_argument("--reps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/c1_unlabelled_calibration_pilot.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    activation = np.load(args.activation, allow_pickle=True)
    baselines = np.load(args.baselines, allow_pickle=True)
    tune_labels = activation["tune_labels"]
    test_labels = activation["test_labels"]

    results = {"seed": args.seed, "reps": args.reps, "conditions": {}}
    for condition in CONDITIONS:
        detectors = {
            "probe": (
                activation[f"tune_{condition}_logistic"],
                activation[f"test_{condition}_logistic"],
            ),
            "shieldgemma": (
                baselines[f"tune_{condition}_shieldgemma"],
                baselines[f"test_{condition}_shieldgemma"],
            ),
        }
        condition_result = {}
        for name, (tune_scores, test_scores) in detectors.items():
            oracle = np.quantile(tune_scores[tune_labels == 0], 0.99)
            oracle_tpr, oracle_fpr = operating_point(oracle, test_scores, test_labels)
            strict_source = (
                activation["tune_plain_logistic"]
                if name == "probe"
                else baselines["tune_plain_shieldgemma"]
            )
            strict = np.quantile(strict_source[tune_labels == 0], 0.99)
            strict_tpr, strict_fpr = operating_point(strict, test_scores, test_labels)
            entry = {
                "oracle": {"tpr": oracle_tpr, "fpr": oracle_fpr},
                "strict": {"tpr": strict_tpr, "fpr": strict_fpr},
                "unlabelled": {},
            }
            for pi in PREVALENCES:
                for k in K_VALUES:
                    entry["unlabelled"][f"k{k}_pi{pi}"] = simulate(
                        rng,
                        tune_scores,
                        tune_labels,
                        test_scores,
                        test_labels,
                        k,
                        pi,
                        args.reps,
                    )
            condition_result[name] = entry
        results["conditions"][condition] = condition_result

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    for condition, detectors in results["conditions"].items():
        for name, entry in detectors.items():
            reference = entry["oracle"]
            recovered = entry["unlabelled"]["k300_pi0.01"]
            print(
                f"[{condition}/{name}] strict {entry['strict']['tpr']:.1f} | "
                f"unlabelled k=300 pi=0.01 {recovered['tpr_mean']:.1f}"
                f"/{recovered['fpr_mean']:.2f} | oracle {reference['tpr']:.1f}"
                f"/{reference['fpr']:.2f}"
            )
    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
