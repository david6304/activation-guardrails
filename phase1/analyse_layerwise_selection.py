"""Single-layer readouts selected on plain tune data only (C2 pilot).

The all-layer concatenated logistic probe collapses under representational
shift (reverse AUROC 0.556) while single layers retain far more rank
information. Selection uses plain tune scores only: no transformed data and no
test data enter the rule. Also reports the prompt-length interaction that
motivates C6 and X5.

Pilot scope: AUROC and operating points for the plain-tune argmax layer, plus
the per-condition argmax as a clearly-labelled upper bound. The C2 deliverable
in DISSERTATION_FINISH_PLAN.md adds the full per-layer curve figure and paired
bootstrap intervals.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

LAYERWISE = "data/phase1_layerwise_27b.npz"
PROMPTS = "data/judged_main_prompts.jsonl"
CONDITIONS = ("plain", "swahili", "reverse", "vowel")
KINDS = ("logistic", "centroid")


def auroc_curve(scores, labels):
    return np.array(
        [roc_auc_score(labels, scores[:, layer]) for layer in range(scores.shape[1])]
    )


def operating_point(threshold_scores, tune_labels, test_scores, test_labels):
    threshold = np.quantile(threshold_scores[tune_labels == 0], 0.99)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layerwise", default=LAYERWISE)
    parser.add_argument("--prompts", default=PROMPTS)
    parser.add_argument("--out", default="data/c2_layerwise_selection_pilot.json")
    args = parser.parse_args()

    data = np.load(args.layerwise, allow_pickle=True)
    tune_labels = data["tune_labels"]
    test_labels = data["test_labels"]

    results = {"selection_rule": "argmax AUROC on plain tune scores", "readouts": {}}
    for kind in KINDS:
        tune_curves = {
            condition: auroc_curve(data[f"tune_{condition}_{kind}"], tune_labels)
            for condition in CONDITIONS
        }
        test_curves = {
            condition: auroc_curve(data[f"test_{condition}_{kind}"], test_labels)
            for condition in CONDITIONS
        }
        selected = int(tune_curves["plain"].argmax())
        entry = {"selected_layer": selected, "conditions": {}}
        plain_tune = data[f"tune_plain_{kind}"][:, selected]
        for condition in CONDITIONS:
            tune_scores = data[f"tune_{condition}_{kind}"][:, selected]
            test_scores = data[f"test_{condition}_{kind}"][:, selected]
            strict_tpr, strict_fpr = operating_point(
                plain_tune, tune_labels, test_scores, test_labels
            )
            matched_tpr, matched_fpr = operating_point(
                tune_scores, tune_labels, test_scores, test_labels
            )
            entry["conditions"][condition] = {
                "auroc": float(test_curves[condition][selected]),
                "strict_tpr": strict_tpr,
                "strict_fpr": strict_fpr,
                "matched_tpr": matched_tpr,
                "matched_fpr": matched_fpr,
                "upper_bound_auroc": float(test_curves[condition].max()),
                "upper_bound_layer": int(test_curves[condition].argmax()),
            }
        entry["per_layer_test_auroc"] = {
            condition: test_curves[condition].tolist() for condition in CONDITIONS
        }
        results["readouts"][kind] = entry

    lengths = prompt_lengths(data["test_ids"], args.prompts)
    quartiles = np.digitize(lengths, np.quantile(lengths, [0.25, 0.5, 0.75]))
    layer = results["readouts"]["centroid"]["selected_layer"]
    results["length_interaction"] = {"selected_layer": layer, "conditions": {}}
    for condition in CONDITIONS:
        scores = data[f"test_{condition}_centroid"][:, layer]
        results["length_interaction"]["conditions"][condition] = [
            float(roc_auc_score(test_labels[quartiles == q], scores[quartiles == q]))
            for q in range(4)
        ]

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    for kind, entry in results["readouts"].items():
        cells = " ".join(
            f"{c}:{entry['conditions'][c]['auroc']:.3f}" for c in CONDITIONS
        )
        print(f"[{kind}] plain-tune layer L{entry['selected_layer']}  {cells}")
    for condition, curve in results["length_interaction"]["conditions"].items():
        print(f"[length] {condition}: " + " ".join(f"{a:.3f}" for a in curve))
    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
