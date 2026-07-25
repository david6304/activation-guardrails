"""C3: operating points calibrated on the large WildChat background pool.

The Phase 1 thresholds are set on ~1,213 tune negatives, so the 1% point rests
on about twelve prompts and 0.1% is not estimable at all. This sets thresholds
on 50,000 unlabelled WildChat prompts instead, at background alert rates of 1%
and 0.1%, and reads the frozen Phase 1 test TPR at those thresholds.

WildChat is unlabelled, so the quantity fixed by the threshold is a background
alert rate, never an FPR. The labelled FPR on the Phase 1 test negatives is
reported beside it.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ACTIVATION = "data/phase1_activation_multilingual_27b.npz"
BASELINES = "data/phase1_baselines_multilingual.npz"
MODERN_GUARDS = "data/c4_modern_guards.npz"
CONDITIONS = ("plain", "french", "hindi", "swahili", "zulu", "reverse")
RATES = (0.01, 0.001)


def phase1_scores(activation, baselines, modern, detector, condition):
    if detector == "probe":
        return activation[f"test_{condition}_logistic"]
    if detector == "centroid":
        return activation[f"test_{condition}_centroid"]
    if detector == "shieldgemma":
        return baselines[f"test_{condition}_shieldgemma"]
    if detector == "qwen3guard":
        return modern[f"test_{condition}_qwen3guard"]
    raise ValueError(detector)


def pool_key(detector):
    return {"probe": "logistic", "centroid": "centroid"}.get(detector, detector)


def operating_point(threshold, scores, labels):
    return {
        "tpr": float((scores[labels == 1] > threshold).mean() * 100),
        "fpr": float((scores[labels == 0] > threshold).mean() * 100),
        "threshold": float(threshold),
    }


def bootstrap_tpr(pool, rate, scores, labels, repeats, seed):
    rng = np.random.default_rng(seed)
    positive = scores[labels == 1]
    draws = np.empty(repeats)
    for repeat in range(repeats):
        sample = pool[rng.integers(0, len(pool), len(pool))]
        threshold = np.quantile(sample, 1 - rate)
        draws[repeat] = (positive > threshold).mean() * 100
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool",
        nargs="+",
        required=True,
        help="C3 pool score npz files, one per detector and condition",
    )
    parser.add_argument("--activation", default=ACTIVATION)
    parser.add_argument("--baselines", default=BASELINES)
    parser.add_argument("--modern-guards", default=MODERN_GUARDS)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/c3_pool_results.json")
    args = parser.parse_args()

    activation = np.load(args.activation, allow_pickle=True)
    baselines = np.load(args.baselines, allow_pickle=True)
    modern = np.load(args.modern_guards, allow_pickle=True)
    test_labels = activation["test_labels"]

    # {(detector, pool condition): pool scores}
    pools = {}
    metadata = {}
    for path in args.pool:
        with np.load(path, allow_pickle=True) as saved:
            info = json.loads(str(saved["metadata_json"]))
            pool_condition = info["condition"]
            metadata[Path(path).name] = info
            for key in saved.files:
                if not key.startswith(f"{pool_condition}_"):
                    continue
                name = key[len(pool_condition) + 1 :]
                detector = {"logistic": "probe", "centroid": "centroid"}.get(name, name)
                if detector in ("probe", "centroid", "shieldgemma", "qwen3guard"):
                    pools[(detector, pool_condition)] = saved[key]

    results = {
        "pool_files": metadata,
        "rates": list(RATES),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "cells": {},
    }
    for (detector, pool_condition), pool in sorted(pools.items()):
        for rate in RATES:
            threshold = float(np.quantile(pool, 1 - rate))
            for condition in CONDITIONS:
                scores = phase1_scores(
                    activation, baselines, modern, detector, condition
                )
                cell = operating_point(threshold, scores, test_labels)
                cell["tpr_ci"] = bootstrap_tpr(
                    pool, rate, scores, test_labels, args.bootstrap, args.seed
                )
                cell["pool_n"] = int(len(pool))
                key = f"{detector}|pool={pool_condition}|rate={rate}|{condition}"
                results["cells"][key] = cell

    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    for key, cell in results["cells"].items():
        print(
            f"{key:56s} tpr {cell['tpr']:5.1f} "
            f"[{cell['tpr_ci'][0]:.1f},{cell['tpr_ci'][1]:.1f}] "
            f"labelled fpr {cell['fpr']:.2f}"
        )
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
