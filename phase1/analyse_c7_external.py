"""C7 external-source confirmation: probe versus ShieldGemma on Aegis 2.0.

Thresholds come only from the external *tune* negatives; TPR and realised FPR are
reported on the disjoint external *test* partition. Three calibration modes are
reported: the frozen WildJailbreak plain threshold (true zero-shot source transfer),
external-English, and external-Swahili matched.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from phase1.analyse_phase1 import metrics, threshold_at_one_percent

DETECTORS = ("logistic", "centroid", "shieldgemma")
CONDITIONS = ("plain", "swahili")


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def thresholds_for(probe, guard, frozen, condition, mode):
    """Return {detector: threshold} for one calibration mode."""
    out = {}
    for detector in DETECTORS:
        source = guard if detector == "shieldgemma" else probe
        if mode == "frozen_wildjailbreak":
            out[detector] = frozen[detector]
        else:
            calibration = "plain" if mode == "external_english" else condition
            out[detector] = threshold_at_one_percent(
                source[f"tune_{calibration}_{detector}"]
            )
    return out


def bootstrap(probe, guard, frozen, condition, mode, repeats, seed):
    """Paired resample of calibration negatives, test positives and test negatives."""
    rng = np.random.default_rng(seed)
    labels = probe["test_labels"]
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    n_tune = len(probe["tune_labels"])
    tprs = {d: np.empty(repeats) for d in DETECTORS}
    fprs = {d: np.empty(repeats) for d in DETECTORS}
    differences = np.empty(repeats)

    for repeat in range(repeats):
        calibration_sample = rng.choice(n_tune, size=n_tune, replace=True)
        positive_sample = rng.choice(positive, size=len(positive), replace=True)
        negative_sample = rng.choice(negative, size=len(negative), replace=True)
        for detector in DETECTORS:
            source = guard if detector == "shieldgemma" else probe
            if mode == "frozen_wildjailbreak":
                threshold = frozen[detector]
            else:
                calibration = "plain" if mode == "external_english" else condition
                threshold = threshold_at_one_percent(
                    source[f"tune_{calibration}_{detector}"][calibration_sample]
                )
            scores = source[f"test_{condition}_{detector}"]
            tprs[detector][repeat] = np.mean(scores[positive_sample] > threshold)
            fprs[detector][repeat] = np.mean(scores[negative_sample] > threshold)
        differences[repeat] = tprs["logistic"][repeat] - tprs["shieldgemma"][repeat]

    return {
        "probe_minus_guard_tpr_mean": float(100 * differences.mean()),
        "probe_minus_guard_tpr_ci": [
            float(100 * np.percentile(differences, 2.5)),
            float(100 * np.percentile(differences, 97.5)),
        ],
        "fpr_ci": {
            detector: [
                float(100 * np.percentile(fprs[detector], 2.5)),
                float(100 * np.percentile(fprs[detector], 97.5)),
            ]
            for detector in DETECTORS
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default="data/c7_external_scores.npz")
    parser.add_argument("--guard", default="data/c7_external_guard.npz")
    parser.add_argument(
        "--frozen", default="data/phase1_activation_multilingual_27b.npz"
    )
    parser.add_argument("--frozen-guard", default="data/phase1_baselines_multilingual.npz")
    parser.add_argument("--out", default="data/c7_external_results.json")
    parser.add_argument("--repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    probe = load_npz(Path(args.scores))
    guard = load_npz(Path(args.guard))
    if not np.array_equal(probe["test_ids"], guard["test_ids"]):
        raise ValueError("probe and guard test partitions differ")
    if not np.array_equal(probe["test_labels"], guard["test_labels"]):
        raise ValueError("probe and guard test labels differ")

    # The frozen WildJailbreak thresholds: plain tune negatives of the source artefacts.
    source = load_npz(Path(args.frozen))
    source_guard = load_npz(Path(args.frozen_guard))
    source_negative = source["tune_labels"] == 0
    frozen = {
        "logistic": threshold_at_one_percent(
            source["tune_plain_logistic"][source_negative]
        ),
        "centroid": threshold_at_one_percent(
            source["tune_plain_centroid"][source_negative]
        ),
        "shieldgemma": threshold_at_one_percent(
            source_guard["tune_plain_shieldgemma"][source_negative]
        ),
    }

    labels = probe["test_labels"]
    results = {
        "counts": {
            "tune_negatives": int(len(probe["tune_labels"])),
            "test_positives": int(labels.sum()),
            "test_negatives": int((labels == 0).sum()),
        },
        "frozen_wildjailbreak_thresholds": frozen,
        "conditions": {},
    }

    for condition in CONDITIONS:
        per_mode = {}
        for mode in ("frozen_wildjailbreak", "external_english", "external_swahili"):
            if mode == "external_swahili" and condition != "swahili":
                continue
            chosen = thresholds_for(probe, guard, frozen, condition, mode)
            per_detector = {}
            for detector in DETECTORS:
                src = guard if detector == "shieldgemma" else probe
                per_detector[detector] = metrics(
                    labels, src[f"test_{condition}_{detector}"], chosen[detector]
                )
            per_detector["paired_bootstrap"] = bootstrap(
                probe, guard, frozen, condition, mode, args.repeats, args.seed
            )
            per_mode[mode] = per_detector
        results["conditions"][condition] = per_mode

    Path(args.out).write_text(json.dumps(results, indent=2) + "\n")

    for condition, modes in results["conditions"].items():
        for mode, per_detector in modes.items():
            print(f"\n[{condition}] {mode}")
            for detector in DETECTORS:
                m = per_detector[detector]
                print(
                    f"  {detector:12s} AUROC {m['auroc']:.3f}  "
                    f"TPR {100 * m['tpr']:5.1f}  FPR {100 * m['fpr']:5.2f}"
                )
            b = per_detector["paired_bootstrap"]
            lo, hi = b["probe_minus_guard_tpr_ci"]
            print(
                f"  probe - guard TPR {b['probe_minus_guard_tpr_mean']:+.1f} "
                f"[{lo:+.1f}, {hi:+.1f}]"
            )
    print(f"\n[done] {args.out}")


if __name__ == "__main__":
    main()
