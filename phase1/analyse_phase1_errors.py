"""Analyse frozen Phase 1 errors without fitting or selecting a detector."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from phase1.analyse_phase1 import threshold_at_one_percent


CONDITIONS = ("plain", "swahili")
WEAPONS_LONG = (
    "weapons & CBRN (chemical, biological, radiological, nuclear, explosive)"
)


def normalise_category(category):
    return "weapons & CBRN" if category == WEAPONS_LONG else category


def wilson_interval(successes, total):
    if total == 0:
        return [None, None]
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z**2 / total
    centre = (rate + z**2 / (2 * total)) / denominator
    half_width = (
        z
        * np.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2))
        / denominator
    )
    return [float(centre - half_width), float(centre + half_width)]


def category_results(labels, categories, detector_alerts):
    results = {}
    for category in sorted(set(categories[labels == 1])):
        selected = (labels == 1) & (categories == category)
        count = int(selected.sum())
        results[category] = {"harmful_n": count}
        for detector, alerts in detector_alerts.items():
            detected = int(alerts[selected].sum())
            results[category][detector] = {
                "detected": detected,
                "tpr": detected / count,
                "wilson_95ci": wilson_interval(detected, count),
            }
    return results


def overlap_by_category(labels, categories, activation_alerts, text_alerts):
    results = {}
    for category in sorted(set(categories[labels == 1])):
        selected = (labels == 1) & (categories == category)
        activation = activation_alerts[selected]
        text = text_alerts[selected]
        results[category] = {
            "harmful_n": int(selected.sum()),
            "activation_only": int((activation & ~text).sum()),
            "text_only": int((~activation & text).sum()),
            "shared": int((activation & text).sum()),
            "neither": int((~activation & ~text).sum()),
        }
    return results


def paired_shift(labels, plain_scores, shifted_scores):
    results = {}
    delta = shifted_scores - plain_scores
    for group, selected in (
        ("harmful", labels == 1),
        ("benign", labels == 0),
        ("all", np.ones(len(labels), dtype=bool)),
    ):
        correlation = spearmanr(
            plain_scores[selected], shifted_scores[selected]
        ).statistic
        values = delta[selected]
        results[group] = {
            "n": int(selected.sum()),
            "spearman_rho": float(correlation),
            "mean_score_change": float(values.mean()),
            "median_score_change": float(np.median(values)),
            "score_change_q05_q95": np.quantile(values, [0.05, 0.95]).tolist(),
        }
    return results


def empirical_percentiles(values):
    return (rankdata(values, method="average") - 1) / max(len(values) - 1, 1)


def load_prompts(path):
    prompts = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                prompts[str(row["id"])] = row["prompt"]
    return prompts


def load_translations(path):
    translations = {}
    with path.open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                translations[row["prompt"]] = row["translation"]
    return translations


def disagreement_examples(
    ids,
    labels,
    categories,
    activation_scores,
    text_scores,
    activation_alerts,
    text_alerts,
    prompts,
    translations,
    count,
):
    difference = empirical_percentiles(activation_scores) - empirical_percentiles(
        text_scores
    )
    examples = []
    for group, selected in (("harmful", labels == 1), ("benign", labels == 0)):
        candidates = np.flatnonzero(selected)
        ordered = candidates[np.argsort(-np.abs(difference[candidates]), kind="stable")]
        for index in ordered[:count]:
            prompt = prompts[str(ids[index])]
            examples.append(
                {
                    "group": group,
                    "id": str(ids[index]),
                    "category": str(categories[index]),
                    "rank_difference_activation_minus_text": float(difference[index]),
                    "activation_score": float(activation_scores[index]),
                    "shieldgemma_score": float(text_scores[index]),
                    "activation_alert": bool(activation_alerts[index]),
                    "shieldgemma_alert": bool(text_alerts[index]),
                    "english_prompt": prompt,
                    "swahili_prompt": translations[prompt],
                }
            )
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activation", default="data/phase1_activation_27b.npz")
    parser.add_argument("--baselines", default="data/phase1_baselines.npz")
    parser.add_argument("--sae", default="data/phase3_scores.npz")
    parser.add_argument("--inputs", default="data/judged_main_prompts.jsonl")
    parser.add_argument(
        "--translations", default="data/phase1_translations/swahili.jsonl"
    )
    parser.add_argument("--out", default="data/phase1_error_analysis.json")
    parser.add_argument("--disagreements-per-class", type=int, default=10)
    args = parser.parse_args()

    with (
        np.load(args.activation, allow_pickle=False) as activation,
        np.load(args.baselines, allow_pickle=False) as baselines,
        np.load(args.sae, allow_pickle=False) as sae,
    ):
        ids = activation["test_ids"].astype(str)
        labels = activation["test_labels"]
        categories = np.asarray(
            [normalise_category(value) for value in activation["test_categories"].astype(str)]
        )
        if not np.array_equal(ids, baselines["test_ids"].astype(str)):
            raise ValueError("Phase 1 activation and baseline test IDs do not align")
        if not np.array_equal(ids, sae["test_ids"].astype(str)):
            raise ValueError("Phase 1 and Phase 3 test IDs do not align")
        if not np.array_equal(labels, sae["test_labels"]):
            raise ValueError("Phase 1 and Phase 3 test labels do not align")
        if not np.isfinite(
            np.concatenate(
                [
                    activation["test_plain_logistic"],
                    activation["test_swahili_logistic"],
                    baselines["test_plain_shieldgemma"],
                    baselines["test_swahili_shieldgemma"],
                    sae["test_plain_sae_block31"],
                    sae["test_swahili_sae_block31"],
                ]
            )
        ).all():
            raise ValueError("non-finite frozen score")

        tune_negative = activation["tune_labels"] == 0
        analysis = {
            "status": "descriptive analysis of frozen scores; no detector selection",
            "test_n": len(ids),
            "harmful_n": int((labels == 1).sum()),
            "benign_n": int((labels == 0).sum()),
            "conditions": {},
            "paired_plain_to_swahili": {},
        }
        condition_alerts = {}
        for condition in CONDITIONS:
            detector_scores = {
                "activation": activation[f"test_{condition}_logistic"],
                "shieldgemma": baselines[f"test_{condition}_shieldgemma"],
                "sae_block31": sae[f"test_{condition}_sae_block31"],
            }
            detector_tune_scores = {
                "activation": activation[f"tune_{condition}_logistic"],
                "shieldgemma": baselines[f"tune_{condition}_shieldgemma"],
                "sae_block31": sae[f"tune_{condition}_sae_block31"],
            }
            thresholds = {
                name: threshold_at_one_percent(scores[tune_negative])
                for name, scores in detector_tune_scores.items()
            }
            alerts = {
                name: scores > thresholds[name]
                for name, scores in detector_scores.items()
            }
            condition_alerts[condition] = alerts
            analysis["conditions"][condition] = {
                "calibration": "condition-matched tune negatives at nominal 1% FPR",
                "thresholds": thresholds,
                "category_tpr": category_results(labels, categories, alerts),
                "activation_shieldgemma_overlap": overlap_by_category(
                    labels,
                    categories,
                    alerts["activation"],
                    alerts["shieldgemma"],
                ),
            }

        for detector, source, suffix in (
            ("activation", activation, "logistic"),
            ("shieldgemma", baselines, "shieldgemma"),
            ("sae_block31", sae, "sae_block31"),
        ):
            analysis["paired_plain_to_swahili"][detector] = paired_shift(
                labels,
                source[f"test_plain_{suffix}"],
                source[f"test_swahili_{suffix}"],
            )

        swahili_activation = condition_alerts["swahili"]["activation"]
        swahili_text = condition_alerts["swahili"]["shieldgemma"]
        positive = labels == 1
        negative = labels == 0
        union = swahili_activation | swahili_text
        analysis["swahili_naive_union"] = {
            "activation_harmful_detections": int(swahili_activation[positive].sum()),
            "shieldgemma_unique_harmful_detections": int(
                (~swahili_activation & swahili_text & positive).sum()
            ),
            "union_tpr": float(union[positive].mean()),
            "union_fpr": float(union[negative].mean()),
        }

        prompts = load_prompts(Path(args.inputs))
        translations = load_translations(Path(args.translations))
        missing_prompts = [item_id for item_id in ids if item_id not in prompts]
        missing_translations = [
            item_id
            for item_id in ids
            if item_id in prompts and prompts[item_id] not in translations
        ]
        if missing_prompts or missing_translations:
            raise ValueError(
                f"missing prompts={len(missing_prompts)} "
                f"translations={len(missing_translations)}"
            )
        analysis["largest_activation_shieldgemma_disagreements"] = (
            disagreement_examples(
                ids,
                labels,
                categories,
                activation["test_swahili_logistic"],
                baselines["test_swahili_shieldgemma"],
                swahili_activation,
                swahili_text,
                prompts,
                translations,
                args.disagreements_per_class,
            )
        )
        analysis["category_counts"] = dict(
            Counter(categories[labels == 1].tolist())
        )

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(analysis, indent=2) + "\n")

    print(
        f"[aligned] test={analysis['test_n']} harmful={analysis['harmful_n']} "
        f"benign={analysis['benign_n']}"
    )
    for detector, result in analysis["paired_plain_to_swahili"].items():
        print(
            f"[rank] {detector}: harmful rho={result['harmful']['spearman_rho']:.3f} "
            f"benign rho={result['benign']['spearman_rho']:.3f}"
        )
    union_result = analysis["swahili_naive_union"]
    print(
        f"[union] unique_text_harmful="
        f"{union_result['shieldgemma_unique_harmful_detections']} "
        f"TPR={union_result['union_tpr']:.3%} "
        f"FPR={union_result['union_fpr']:.3%}"
    )
    print(f"[done] {output_path}")


if __name__ == "__main__":
    main()
