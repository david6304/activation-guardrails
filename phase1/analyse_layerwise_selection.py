"""Single-layer readouts selected on plain tune data only (C2).

The all-layer concatenated logistic probe collapses under representational
shift (reverse AUROC 0.556) while single layers retain far more rank
information. Selection uses plain tune scores only: no transformed data and no
test data enter the rule.

Reports the full per-layer AUROC curves, the selected readout against the
all-layer logistic and layer-averaged centroid references under both
calibrations, paired bootstrap intervals on those differences, and the
prompt-length interaction that motivates C6.

The honesty constraint from DISSERTATION_FINISH_PLAN.md applies to any writeup
of this: the gain is rank-level and does not survive at the 1%-FPR operating
point.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from phase1.analyse_phase1 import threshold_at_one_percent

LAYERWISE = "data/phase1_layerwise_27b.npz"
ACTIVATION = "data/phase1_activation_multilingual_27b.npz"
PROMPTS = "data/judged_main_prompts.jsonl"
CONDITIONS = ("plain", "swahili", "reverse", "vowel")
KINDS = ("logistic", "centroid")


def auroc_curve(scores, labels):
    return np.array(
        [roc_auc_score(labels, scores[:, layer]) for layer in range(scores.shape[1])]
    )


def operating_point(reference_scores, test_scores, test_labels):
    threshold = threshold_at_one_percent(reference_scores)
    tpr = float((test_scores[test_labels == 1] > threshold).mean() * 100)
    fpr = float((test_scores[test_labels == 0] > threshold).mean() * 100)
    return tpr, fpr


def prompt_lengths(ids, path):
    texts = {}
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            key = str(row.get("id", row.get("prompt_id", "")))
            texts[key] = row.get("prompt") or row.get("text") or ""
    return np.array([len(texts[str(i)]) for i in ids])


def bootstrap_readouts(readouts, tune_labels, test_labels, repeats, seed):
    """Paired resampling of calibration negatives and matched test examples.

    `readouts` maps a detector name to (tune_plain, tune_matched, test) scores.
    Mirrors phase1.analyse_phase1.bootstrap_primary so intervals are comparable.
    """
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(test_labels == 1)
    negative = np.flatnonzero(test_labels == 0)
    tune_negative = np.flatnonzero(tune_labels == 0)
    names = list(readouts)
    draws = {
        name: {key: np.empty(repeats) for key in ("auroc", "strict_tpr", "matched_tpr")}
        for name in names
    }

    for repeat in range(repeats):
        calibration_sample = rng.choice(
            tune_negative, size=len(tune_negative), replace=True
        )
        positive_sample = rng.choice(positive, size=len(positive), replace=True)
        negative_sample = rng.choice(negative, size=len(negative), replace=True)
        sample = np.concatenate([positive_sample, negative_sample])
        sample_labels = test_labels[sample]
        for name in names:
            tune_plain, tune_matched, test_scores = readouts[name]
            strict = threshold_at_one_percent(tune_plain[calibration_sample])
            matched = threshold_at_one_percent(tune_matched[calibration_sample])
            draws[name]["auroc"][repeat] = roc_auc_score(
                sample_labels, test_scores[sample]
            )
            draws[name]["strict_tpr"][repeat] = np.mean(
                test_scores[positive_sample] > strict
            )
            draws[name]["matched_tpr"][repeat] = np.mean(
                test_scores[positive_sample] > matched
            )
    return draws


def interval(values):
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layerwise", default=LAYERWISE)
    parser.add_argument("--activation", default=ACTIVATION)
    parser.add_argument("--prompts", default=PROMPTS)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/c2_layerwise_selection.json")
    args = parser.parse_args()

    data = np.load(args.layerwise, allow_pickle=True)
    activation = np.load(args.activation, allow_pickle=True)
    assert np.array_equal(data["test_ids"], activation["test_ids"])
    assert np.array_equal(data["tune_ids"], activation["tune_ids"])
    tune_labels = data["tune_labels"]
    test_labels = data["test_labels"]
    layer_indices = data["layer_indices"]

    results = {
        "selection_rule": "argmax AUROC on plain tune scores",
        "layer_indices": layer_indices.tolist(),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "readouts": {},
    }

    # Readouts held fixed across conditions: the selected single layer of each
    # kind, plus the two aggregated references. The all-layer logistic has no
    # vowel-removal condition in the frozen artefact.
    selected = {}
    for kind in KINDS:
        tune_curves = {
            condition: auroc_curve(data[f"tune_{condition}_{kind}"], tune_labels)
            for condition in CONDITIONS
        }
        test_curves = {
            condition: auroc_curve(data[f"test_{condition}_{kind}"], test_labels)
            for condition in CONDITIONS
        }
        column = int(tune_curves["plain"].argmax())
        selected[kind] = column
        entry = {
            "selected_column": column,
            "selected_layer": int(layer_indices[column]),
            "plain_tune_auroc": float(tune_curves["plain"][column]),
            "conditions": {},
            "per_layer_test_auroc": {
                condition: test_curves[condition].tolist() for condition in CONDITIONS
            },
            "per_layer_tune_auroc": {
                condition: tune_curves[condition].tolist() for condition in CONDITIONS
            },
        }
        plain_tune = data[f"tune_plain_{kind}"][:, column]
        for condition in CONDITIONS:
            tune_scores = data[f"tune_{condition}_{kind}"][:, column]
            test_scores = data[f"test_{condition}_{kind}"][:, column]
            strict_tpr, strict_fpr = operating_point(
                plain_tune[tune_labels == 0], test_scores, test_labels
            )
            matched_tpr, matched_fpr = operating_point(
                tune_scores[tune_labels == 0], test_scores, test_labels
            )
            best_column = int(test_curves[condition].argmax())
            entry["conditions"][condition] = {
                "auroc": float(test_curves[condition][column]),
                "strict_tpr": strict_tpr,
                "strict_fpr": strict_fpr,
                "matched_tpr": matched_tpr,
                "matched_fpr": matched_fpr,
                "upper_bound_auroc": float(test_curves[condition][best_column]),
                "upper_bound_layer": int(layer_indices[best_column]),
                "own_tune_selected_layer": int(
                    layer_indices[int(tune_curves[condition].argmax())]
                ),
                "own_tune_selected_auroc": float(
                    test_curves[condition][int(tune_curves[condition].argmax())]
                ),
            }
        results["readouts"][kind] = entry

    # Per-condition comparison of the selected layers against both aggregated
    # references, with paired bootstrap intervals on the differences.
    results["comparison"] = {}
    for condition in CONDITIONS:
        readouts = {
            f"L{layer_indices[selected[kind]]}_{kind}": (
                data[f"tune_plain_{kind}"][:, selected[kind]],
                data[f"tune_{condition}_{kind}"][:, selected[kind]],
                data[f"test_{condition}_{kind}"][:, selected[kind]],
            )
            for kind in KINDS
        }
        # Frozen artefact where it exists, so reference rows match the reported
        # tables; vowel removal was only ever scored layerwise, and its
        # recomputed layer mean agrees with the frozen centroid to ~1e-4.
        if f"test_{condition}_centroid" in activation.files:
            readouts["layer_avg_centroid"] = (
                activation["tune_plain_centroid"],
                activation[f"tune_{condition}_centroid"],
                activation[f"test_{condition}_centroid"],
            )
            readouts["all_layer_logistic"] = (
                activation["tune_plain_logistic"],
                activation[f"tune_{condition}_logistic"],
                activation[f"test_{condition}_logistic"],
            )
        else:
            readouts["layer_avg_centroid"] = (
                data["tune_plain_centroid"].mean(axis=1),
                data[f"tune_{condition}_centroid"].mean(axis=1),
                data[f"test_{condition}_centroid"].mean(axis=1),
            )
        draws = bootstrap_readouts(
            readouts, tune_labels, test_labels, args.bootstrap, args.seed
        )
        condition_entry = {}
        for name, (tune_plain, tune_matched, test_scores) in readouts.items():
            strict_tpr, strict_fpr = operating_point(
                tune_plain[tune_labels == 0], test_scores, test_labels
            )
            matched_tpr, matched_fpr = operating_point(
                tune_matched[tune_labels == 0], test_scores, test_labels
            )
            condition_entry[name] = {
                "auroc": float(roc_auc_score(test_labels, test_scores)),
                "auroc_ci": interval(draws[name]["auroc"]),
                "strict_tpr": strict_tpr,
                "strict_fpr": strict_fpr,
                "strict_tpr_ci": interval(draws[name]["strict_tpr"] * 100),
                "matched_tpr": matched_tpr,
                "matched_fpr": matched_fpr,
                "matched_tpr_ci": interval(draws[name]["matched_tpr"] * 100),
            }
        baseline = (
            "all_layer_logistic" if "all_layer_logistic" in readouts else "layer_avg_centroid"
        )
        condition_entry["paired_differences"] = {
            "baseline": baseline,
            "detectors": {
                name: {
                    "auroc_delta": float(
                        np.mean(draws[name]["auroc"] - draws[baseline]["auroc"])
                    ),
                    "auroc_delta_ci": interval(
                        draws[name]["auroc"] - draws[baseline]["auroc"]
                    ),
                    "matched_tpr_delta": float(
                        np.mean(
                            (draws[name]["matched_tpr"] - draws[baseline]["matched_tpr"])
                            * 100
                        )
                    ),
                    "matched_tpr_delta_ci": interval(
                        (draws[name]["matched_tpr"] - draws[baseline]["matched_tpr"])
                        * 100
                    ),
                }
                for name in readouts
                if name != baseline
            },
        }
        results["comparison"][condition] = condition_entry

    lengths = prompt_lengths(data["test_ids"], args.prompts)
    quartiles = np.digitize(lengths, np.quantile(lengths, [0.25, 0.5, 0.75]))
    column = selected["centroid"]
    results["length_interaction"] = {
        "selected_layer": int(layer_indices[column]),
        "conditions": {},
    }
    for condition in CONDITIONS:
        scores = data[f"test_{condition}_centroid"][:, column]
        results["length_interaction"]["conditions"][condition] = [
            float(roc_auc_score(test_labels[quartiles == q], scores[quartiles == q]))
            for q in range(4)
        ]

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    for kind, entry in results["readouts"].items():
        print(f"[{kind}] plain-tune layer L{entry['selected_layer']}")
    for condition, entry in results["comparison"].items():
        print(f"-- {condition}")
        for name, cell in entry.items():
            if name == "paired_differences":
                continue
            print(
                f"   {name:22s} auroc {cell['auroc']:.3f} "
                f"[{cell['auroc_ci'][0]:.3f},{cell['auroc_ci'][1]:.3f}]  "
                f"strict {cell['strict_tpr']:5.1f}/{cell['strict_fpr']:.2f}  "
                f"matched {cell['matched_tpr']:5.1f}/{cell['matched_fpr']:.2f}"
            )
    for condition, curve in results["length_interaction"]["conditions"].items():
        print(f"[length] {condition}: " + " ".join(f"{a:.3f}" for a in curve))
    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
