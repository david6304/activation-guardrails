"""Figures for C1 and C2 from their frozen result JSONs.

Both analyses are already complete and recorded in `RESULTS.md` §5 and §6; this
draws the two figures those sections reference. No numbers are recomputed here.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path("figures")
CONDITION_COLOURS = {
    "french": "#4c72b0",
    "hindi": "#dd8452",
    "swahili": "#55a868",
    "zulu": "#c44e52",
    "reverse": "#8172b3",
    "plain": "#937860",
    "vowel": "#da8bc3",
}


def plot_c1(path):
    result = json.load(open(path))
    ks = result["k_values"]
    pis = result["prevalences"]
    conditions = list(result["conditions"])

    figure, axes = plt.subplots(1, len(pis), figsize=(3.0 * len(pis), 3.2), sharey=True)
    for axis, pi in zip(axes, pis):
        for condition in conditions:
            probe = result["conditions"][condition]["probe"]
            colour = CONDITION_COLOURS[condition]
            means = [probe["unlabelled"][f"k{k}_pi{pi}"]["tpr_mean"] for k in ks]
            lows = [probe["unlabelled"][f"k{k}_pi{pi}"]["tpr_ci"][0] for k in ks]
            highs = [probe["unlabelled"][f"k{k}_pi{pi}"]["tpr_ci"][1] for k in ks]
            axis.plot(ks, means, marker="o", markersize=3, color=colour, label=condition)
            axis.fill_between(ks, lows, highs, color=colour, alpha=0.15, linewidth=0)
            axis.axhline(probe["oracle"]["tpr"], color=colour, linestyle="--", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_xticks(ks)
        axis.set_xticklabels([str(k) for k in ks])
        axis.set_xlabel("unlabelled prompts $k$")
        axis.set_title(f"$\\pi = {pi:g}$")
    axes[0].set_ylabel("test TPR (%)")
    axes[-1].legend(fontsize=7, frameon=False)
    figure.suptitle(
        "Probe TPR from unlabelled same-condition thresholds "
        "(dashed = oracle, labelled tune negatives)",
        fontsize=9,
    )
    figure.tight_layout()
    out = FIGURES / "c1_unlabelled_calibration.pdf"
    figure.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def plot_c2(path):
    result = json.load(open(path))
    layers = result["layer_indices"]

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for axis, readout in zip(axes, ("logistic", "centroid")):
        block = result["readouts"][readout]
        for condition, aurocs in block["per_layer_test_auroc"].items():
            axis.plot(
                layers,
                aurocs,
                color=CONDITION_COLOURS[condition],
                linewidth=1.2,
                label=condition,
            )
        axis.axvline(
            block["selected_layer"],
            color="black",
            linestyle=":",
            linewidth=1.0,
        )
        axis.set_xlabel("layer")
        axis.set_title(
            f"per-layer {readout} "
            f"(plain-tune selection: L{block['selected_layer']})",
            fontsize=9,
        )
        axis.axhline(0.5, color="grey", linewidth=0.6, linestyle="-")
    axes[0].set_ylabel("test AUROC")
    axes[0].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    out = FIGURES / "c2_layerwise_selection.pdf"
    figure.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    plot_c1("data/c1_unlabelled_calibration.json")
    plot_c2("data/c2_layerwise_selection.json")
