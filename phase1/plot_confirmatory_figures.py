"""Dissertation figures from frozen Phase 1 result artefacts.

The analyses are complete and recorded in `RESULTS.md`; this script only draws
the figures referenced by the results chapter.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phase1.analyse_text_encoders import analyse_detector, load_npz

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


def plot_calibration_summary(c1_path, c3_path):
    c1 = json.load(open(c1_path))
    c3 = json.load(open(c3_path))
    conditions = ["french", "hindi", "swahili", "zulu"]
    prevalences = c1["prevalences"]

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.25), sharey=True)

    for condition in conditions:
        cells = c1["conditions"][condition]["probe"]["unlabelled"]
        means = [cells[f"k300_pi{pi}"]["tpr_mean"] for pi in prevalences]
        lows = [cells[f"k300_pi{pi}"]["tpr_ci"][0] for pi in prevalences]
        highs = [cells[f"k300_pi{pi}"]["tpr_ci"][1] for pi in prevalences]
        axes[0].plot(
            [100 * pi for pi in prevalences],
            means,
            marker="o",
            markersize=3,
            linewidth=1.4,
            color=CONDITION_COLOURS[condition],
            label=condition.capitalize(),
        )
        axes[0].fill_between(
            [100 * pi for pi in prevalences],
            lows,
            highs,
            color=CONDITION_COLOURS[condition],
            alpha=0.06,
            linewidth=0,
        )
    axes[0].set_title("(a) Contamination sensitivity ($k=300$)", fontsize=9)
    axes[0].set_xlabel("Harmful contamination (%)")
    axes[0].set_xticks([0, 1, 2, 5, 10])
    axes[0].set_ylabel("Test TPR (%)")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    languages = ["plain", "french", "hindi", "swahili", "zulu"]
    x_positions = list(range(len(languages)))
    width = 0.36
    for offset, rate, label, colour in (
        (-width / 2, 0.01, "1% alert rate", "#4c72b0"),
        (width / 2, 0.001, "0.1% alert rate", "#dd8452"),
    ):
        cells = [
            c3["cells"][f"probe|pool=plain|rate={rate}|{language}"]
            for language in languages
        ]
        values = [cell["tpr"] for cell in cells]
        errors = [
            [value - cell["tpr_ci"][0] for value, cell in zip(values, cells)],
            [cell["tpr_ci"][1] - value for value, cell in zip(values, cells)],
        ]
        axes[1].bar(
            [x + offset for x in x_positions],
            values,
            width,
            yerr=errors,
            capsize=2,
            color=colour,
            label=label,
        )
    axes[1].set_title("(b) WildChat threshold", fontsize=9)
    axes[1].set_xlabel("Evaluation language")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels([language.capitalize() for language in languages])
    axes[1].legend(frameon=False, fontsize=7)

    for axis in axes:
        axis.set_ylim(0, 90)
        axis.grid(axis="y", color="0.88", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.tight_layout()
    out = FIGURES / "calibration_operating_points.pdf"
    figure.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def plot_p1_summary(path):
    result = json.load(open(path))
    figure = plt.figure(figsize=(8.0, 5.3))
    grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 1.1), hspace=0.42)
    instruction_axis = figure.add_subplot(grid[0, :])
    auroc_axis = figure.add_subplot(grid[1, 0])
    tpr_axis = figure.add_subplot(grid[1, 1], sharey=auroc_axis)

    rows = [
        ("Plain, $t_{inst}$", "plain", "t_inst"),
        ("Plain wrapped, $t_{cipher}$", "plain_wrapped", "t_cipher"),
        ("Plain wrapped, $t_{inst}$", "plain_wrapped", "t_inst"),
        ("Base64, $t_{cipher}$", "base64", "t_cipher"),
        ("Base64, $t_{inst}$", "base64", "t_inst"),
        ("Shuffled Base64, $t_{cipher}$", "base64_shuffled", "t_cipher"),
        ("Shuffled Base64, $t_{inst}$", "base64_shuffled", "t_inst"),
    ]
    y_positions = np.arange(len(rows))[::-1]
    cells = [
        result["results"]["matched"][condition][position]["logistic"]
        for _, condition, position in rows
    ]

    aurocs = [cell["auroc"] for cell in cells]
    auroc_axis.scatter(aurocs, y_positions, color="0.25", s=22, zorder=3)
    for y, value in zip(y_positions, aurocs):
        auroc_axis.annotate(
            f"{value:.3f}",
            (value, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.5,
        )
    auroc_axis.axvline(0.5, color="0.45", linestyle=":", linewidth=0.8)
    auroc_axis.set_xlim(0.44, 1.07)
    auroc_axis.set_xlabel("AUROC")
    auroc_axis.set_title("(b) Frozen English-direction discrimination", fontsize=9)
    auroc_axis.set_yticks(y_positions)
    auroc_axis.set_yticklabels([label for label, _, _ in rows], fontsize=7)

    tprs = [100 * cell["tpr"] for cell in cells]
    fprs = [100 * cell["fpr"] for cell in cells]
    tpr_axis.scatter(tprs, y_positions, color="0.25", s=22, zorder=3)
    for y, tpr, fpr in zip(y_positions, tprs, fprs):
        tpr_axis.annotate(
            f"{tpr:.1f} ({fpr:.1f})",
            (tpr, y),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=6.5,
        )
    tpr_axis.set_xlim(0, 95)
    tpr_axis.set_xlabel("TPR at matched threshold (%)")
    tpr_axis.set_title("(c) Frozen English-direction operating point", fontsize=9)
    tpr_axis.tick_params(labelleft=False)

    for axis in (auroc_axis, tpr_axis):
        axis.grid(axis="x", color="0.88", linewidth=0.6)
        axis.set_axisbelow(True)

    position = "t_inst"
    plain_trained = result["layer_curves"]["base64"][position]["logistic"][
        "auroc_by_layer"
    ]
    base64_trained = result["base64_selftrained"][position]["test_auroc_by_layer"]
    selected_layer = result["base64_selftrained"][position]["selected_layer"]
    instruction_axis.plot(
        range(len(plain_trained)),
        plain_trained,
        color="#4c72b0",
        linewidth=1.3,
        label="Plain-trained single-layer probe",
    )
    instruction_axis.plot(
        range(len(base64_trained)),
        base64_trained,
        color="#dd8452",
        linewidth=1.3,
        label="Base64-trained single-layer probe",
    )
    instruction_axis.axvline(
        selected_layer, color="0.45", linestyle="--", linewidth=0.8
    )
    instruction_axis.scatter(
        selected_layer,
        base64_trained[selected_layer],
        color="#dd8452",
        edgecolor="white",
        linewidth=0.5,
        s=24,
        zorder=3,
    )
    instruction_axis.annotate(
        f"0.912 at tune-selected index {selected_layer}",
        (selected_layer, base64_trained[selected_layer]),
        xytext=(-5, -14),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
    )
    instruction_axis.axhline(0.5, color="0.45", linestyle=":", linewidth=0.8)
    instruction_axis.set_xlabel("Hidden-state index")
    instruction_axis.set_ylabel("Held-out per-layer AUROC")
    instruction_axis.set_title(
        "(a) Condition-specific Base64 readout at the final instruction token",
        fontsize=9,
    )
    instruction_axis.grid(axis="y", color="0.88", linewidth=0.6)
    instruction_axis.set_axisbelow(True)
    instruction_axis.set_ylim(0.35, 1.0)
    instruction_axis.legend(frameon=False, fontsize=7, ncol=2, loc="lower right")

    figure.tight_layout()
    out = FIGURES / "p1_readability_summary.pdf"
    figure.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


def _annotated_heatmap(axis, values, annotations, title, vmin, vmax, cmap):
    image = axis.imshow(values, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    axis.set_title(title, fontsize=9)
    axis.tick_params(length=0)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if np.ma.is_masked(value):
                colour = "black"
            else:
                normalised = (value - vmin) / (vmax - vmin)
                colour = "white" if normalised > 0.62 else "black"
            axis.text(
                column,
                row,
                annotations[row][column],
                ha="center",
                va="center",
                fontsize=6.5,
                color=colour,
            )
    axis.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    return image


def plot_language_comparison(results_path, haloguard_path):
    report = json.load(open(results_path))
    haloguard = load_npz(haloguard_path)
    halo_results = analyse_detector(
        "haloguard",
        haloguard,
        "haloguard",
        haloguard["tune_labels"],
        haloguard["test_labels"],
    )

    conditions = ["plain", "french", "hindi", "swahili", "zulu"]
    detectors = [
        ("all_layer_logistic", "Activation probe"),
        ("centroid", "Zhao centroid"),
        ("shieldgemma", "ShieldGemma"),
        ("qwen3guard", "Qwen3Guard"),
        ("llamaguard4", "Llama Guard 4"),
        ("haloguard", "HaloGuard"),
        ("multilingual_e5", "Multilingual-e5"),
        ("small_guard", "DeBERTa-v3-small"),
        ("tfidf", "Character TF-IDF"),
    ]

    def cell(detector, condition):
        if detector == "haloguard":
            return halo_results["matched"][condition]
        return report["results"][detector]["matched"][condition]

    tpr = np.asarray(
        [[100 * cell(detector, condition)["tpr"] for condition in conditions]
         for detector, _ in detectors]
    )
    fpr = np.asarray(
        [[100 * cell(detector, condition)["fpr"] for condition in conditions]
         for detector, _ in detectors]
    )
    operating_annotations = [
        [f"{tpr_value:.1f} ({fpr_value:.1f})" for tpr_value, fpr_value in zip(tpr_row, fpr_row)]
        for tpr_row, fpr_row in zip(tpr, fpr)
    ]

    wildguard = report["wildguard_fixed_decision"]
    wildguard_annotations = [
        f"{100 * wildguard[condition]['tpr']:.1f} "
        f"({100 * wildguard[condition]['fpr']:.1f})"
        for condition in conditions
    ]
    operating_values = np.ma.vstack(
        [tpr, np.ma.masked_all((1, len(conditions)))]
    )
    operating_annotations.append(wildguard_annotations)

    figure, operating_axis = plt.subplots(figsize=(8.0, 4.3))

    operating_cmap = plt.get_cmap("Oranges").copy()
    operating_cmap.set_bad("#eeeeee")
    tpr_image = _annotated_heatmap(
        operating_axis,
        operating_values,
        operating_annotations,
        "TPR at thresholds targeting 1% FPR; realised FPR in parentheses",
        0.0,
        80.0,
        operating_cmap,
    )
    operating_axis.set_xticks(range(len(conditions)))
    operating_axis.set_xticklabels(
        [condition.capitalize() for condition in conditions]
    )
    operating_axis.set_yticks(range(len(detectors) + 1))
    operating_axis.set_yticklabels(
        [label for _, label in detectors] + ["WildGuard*"], fontsize=7
    )
    for boundary in (1.5, 5.5, 8.5):
        operating_axis.axhline(boundary, color="black", linewidth=1.2)
    figure.colorbar(tpr_image, ax=operating_axis, fraction=0.025, pad=0.02)

    figure.subplots_adjust(left=0.22, right=0.92, top=0.92, bottom=0.12)
    out = FIGURES / "prompt_language_comparison.pdf"
    figure.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    plot_c1("data/c1_unlabelled_calibration.json")
    plot_c2("data/c2_layerwise_selection.json")
    plot_calibration_summary(
        "data/c1_unlabelled_calibration.json",
        "data/c3_pool_results.json",
    )
    plot_p1_summary("data/p1_position_results.json")
    plot_language_comparison(
        "data/c4_lg4_results.json",
        "data/c4_haloguard.npz",
    )
