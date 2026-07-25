"""Threshold transport from unlabelled shifted traffic (C1).

Estimates the 1%-FPR threshold from `k` unlabelled same-condition tune prompts
drawn at harmful prevalence `pi`, then applies it to the frozen test split.
Compares against the strict (plain-tune-negative) and oracle (all labelled
tune-negative) thresholds.

Covers the five shifted conditions, six detectors, the fixed k/pi grid and
bootstrap percentiles. Qwen3Guard is carried alongside ShieldGemma because C4
found the strongest text guard is condition-dependent.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from phase1.analyse_phase1 import threshold_at_one_percent

ACTIVATION = "data/phase1_activation_multilingual_27b.npz"
BASELINES = "data/phase1_baselines_multilingual.npz"
MODERN_GUARDS = "data/c4_modern_guards.npz"
SMALL_GUARD = "data/phase1_small_guard.npz"
E5 = "data/phase1_multilingual_e5.npz"

CONDITIONS = ("french", "hindi", "swahili", "zulu", "reverse")
K_VALUES = (100, 300, 1000, 3000)
PREVALENCES = (0.00, 0.01, 0.02, 0.05, 0.10)
HEADLINE_CELL = "k300_pi0.01"


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
        # Plain interpolated quantile, not the repo's order-statistic helper: on a
        # 100-prompt draw the "higher" method is markedly conservative. The helper
        # is kept for the strict/oracle reference rows so they match the frozen tables.
        threshold = np.quantile(tune_scores[drawn], 0.99)
        tpr, fpr = operating_point(threshold, test_scores, test_labels)
        tprs.append(tpr)
        fprs.append(fpr)
    return {
        "tpr_mean": float(np.mean(tprs)),
        "tpr_ci": [float(np.percentile(tprs, 2.5)), float(np.percentile(tprs, 97.5))],
        "fpr_mean": float(np.mean(fprs)),
        "fpr_ci": [float(np.percentile(fprs, 2.5)), float(np.percentile(fprs, 97.5))],
    }


def load_detectors(args):
    """Return {detector: {split_condition: scores}} for every condition needed."""
    activation = np.load(args.activation, allow_pickle=True)
    baselines = np.load(args.baselines, allow_pickle=True)
    modern = np.load(args.modern_guards, allow_pickle=True)
    small = np.load(args.small_guard, allow_pickle=True)
    e5 = np.load(args.e5, allow_pickle=True)

    for other in (baselines, modern, small, e5):
        assert np.array_equal(activation["tune_ids"], other["tune_ids"])
        assert np.array_equal(activation["test_ids"], other["test_ids"])

    sources = {
        "probe": (activation, "{split}_{condition}_logistic"),
        "centroid": (activation, "{split}_{condition}_centroid"),
        "shieldgemma": (baselines, "{split}_{condition}_shieldgemma"),
        "qwen3guard": (modern, "{split}_{condition}_qwen3guard"),
        "multilingual_e5": (e5, "{split}_{condition}_scores"),
        "deberta_guard": (small, "{split}_{condition}_scores"),
        "tfidf": (baselines, "{split}_{condition}_tfidf"),
    }
    detectors = {}
    for name, (data, template) in sources.items():
        detectors[name] = {
            f"{split}_{condition}": data[
                template.format(split=split, condition=condition)
            ]
            for split in ("tune", "test")
            for condition in ("plain",) + CONDITIONS
        }
    return activation["tune_labels"], activation["test_labels"], detectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation", default=ACTIVATION)
    parser.add_argument("--baselines", default=BASELINES)
    parser.add_argument("--modern-guards", default=MODERN_GUARDS)
    parser.add_argument("--small-guard", default=SMALL_GUARD)
    parser.add_argument("--e5", default=E5)
    parser.add_argument("--reps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/c1_unlabelled_calibration.json")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    tune_labels, test_labels, detectors = load_detectors(args)

    results = {
        "seed": args.seed,
        "reps": args.reps,
        "k_values": list(K_VALUES),
        "prevalences": list(PREVALENCES),
        "conditions": {},
    }
    for condition in CONDITIONS:
        condition_result = {}
        for name, scores in detectors.items():
            tune_scores = scores[f"tune_{condition}"]
            test_scores = scores[f"test_{condition}"]
            oracle = threshold_at_one_percent(tune_scores[tune_labels == 0])
            strict = threshold_at_one_percent(
                scores["tune_plain"][tune_labels == 0]
            )
            oracle_tpr, oracle_fpr = operating_point(oracle, test_scores, test_labels)
            strict_tpr, strict_fpr = operating_point(strict, test_scores, test_labels)
            entry = {
                "oracle": {"tpr": oracle_tpr, "fpr": oracle_fpr},
                "strict": {"tpr": strict_tpr, "fpr": strict_fpr},
                "unlabelled": {},
            }
            for pi in PREVALENCES:
                for k in K_VALUES:
                    cell = simulate(
                        rng,
                        tune_scores,
                        tune_labels,
                        test_scores,
                        test_labels,
                        k,
                        pi,
                        args.reps,
                    )
                    cell["recovery"] = (
                        cell["tpr_mean"] / oracle_tpr if oracle_tpr > 0 else float("nan")
                    )
                    entry["unlabelled"][f"k{k}_pi{pi}"] = cell
            condition_result[name] = entry
        results["conditions"][condition] = condition_result

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    header = f"{'condition/detector':28s} {'strict':>14s} {'k=300 pi=0.01':>22s} {'oracle':>14s} {'rec':>6s}"
    print(header)
    for condition, entries in results["conditions"].items():
        for name, entry in entries.items():
            cell = entry["unlabelled"][HEADLINE_CELL]
            print(
                f"{condition + '/' + name:28s} "
                f"{entry['strict']['tpr']:8.1f}/{entry['strict']['fpr']:5.2f} "
                f"{cell['tpr_mean']:10.1f} [{cell['tpr_ci'][0]:.1f},{cell['tpr_ci'][1]:.1f}]"
                f"/{cell['fpr_mean']:5.2f} "
                f"{entry['oracle']['tpr']:8.1f}/{entry['oracle']['fpr']:5.2f} "
                f"{cell['recovery'] * 100:5.0f}%"
            )
    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
