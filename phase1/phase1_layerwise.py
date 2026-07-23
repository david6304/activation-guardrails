"""Run the definitive operational-label per-layer diagnostic at ``t_inst``."""

import argparse
import json
from pathlib import Path

import numpy as np

from capability_check import build_sent
from phase1.phase1_activation import (
    NLLB_CODE,
    extract_features,
    iter_position_batches,
)
from probe_prompt import (
    file_sha256,
    load_judged_rows,
    load_model,
    load_or_translate,
    split_rows,
    strings_sha256,
    train_probe,
)


CONDITIONS = ("plain", "swahili", "reverse", "vowel")


def per_layer_centroid_scores(features, harmful_centroid, harmless_centroid):
    """Return one cosine-centroid score per transformer block output."""
    states = features[:, 1:, :]
    state_norm = np.linalg.norm(states, axis=2)
    harmful_norm = np.linalg.norm(harmful_centroid, axis=1)
    harmless_norm = np.linalg.norm(harmless_centroid, axis=1)
    harmful_cosine = np.einsum("nlh,lh->nl", states, harmful_centroid) / np.maximum(
        state_norm * harmful_norm, 1e-12
    )
    harmless_cosine = np.einsum(
        "nlh,lh->nl", states, harmless_centroid
    ) / np.maximum(state_norm * harmless_norm, 1e-12)
    return (harmful_cosine - harmless_cosine).astype(np.float32)


def score_layer_detectors(
    texts,
    model,
    tokenizer,
    batch_size,
    weights,
    intercepts,
    harmful_centroid,
    harmless_centroid,
):
    logistic = np.empty((len(texts), len(weights)), dtype=np.float32)
    centroid = np.empty_like(logistic)
    for indices, features in iter_position_batches(
        texts, model, tokenizer, batch_size, "t_inst"
    ):
        states = features[:, 1:, :]
        logistic[indices] = (
            np.einsum("nlh,lh->nl", states, weights) + intercepts
        )
        centroid[indices] = per_layer_centroid_scores(
            features, harmful_centroid, harmless_centroid
        )
    return logistic, centroid


def transformed_conditions(plain, swahili):
    return {
        "plain": plain,
        "swahili": swahili,
        "reverse": [
            build_sent(text, "reverse", in_obf=True, out_obf=False) for text in plain
        ],
        "vowel": [
            build_sent(text, "vowel", in_obf=True, out_obf=False) for text in plain
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--in", dest="inp", default="data/judged_main_prompts.jsonl")
    parser.add_argument("--nllb", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--translations-dir", default="data/phase1_translations")
    parser.add_argument("--out", default="data/phase1_layerwise_27b.npz")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    model, tokenizer, num_layers, hidden_size = load_model(args.model, args.seed)
    layer_count = num_layers + 1
    if args.smoke:
        _, features = next(
            iter_position_batches(
                ["Explain photosynthesis.", "What is two plus two?"],
                model,
                tokenizer,
                2,
                "t_inst",
            )
        )
        rng = np.random.default_rng(args.seed)
        weights = rng.standard_normal((num_layers, hidden_size)).astype(np.float32)
        weights /= np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
        intercepts = np.zeros(num_layers)
        harmful_centroid = rng.standard_normal(
            (num_layers, hidden_size)
        ).astype(np.float32)
        harmless_centroid = rng.standard_normal(
            (num_layers, hidden_size)
        ).astype(np.float32)
        logistic = np.einsum("nlh,lh->nl", features[:, 1:, :], weights) + intercepts
        centroid = per_layer_centroid_scores(
            features, harmful_centroid, harmless_centroid
        )
        expected = (2, num_layers)
        if (
            features.shape != (2, layer_count, hidden_size)
            or logistic.shape != expected
            or centroid.shape != expected
            or not np.isfinite(logistic).all()
            or not np.isfinite(centroid).all()
        ):
            raise ValueError("invalid layerwise smoke output")
        print(
            f"[smoke] position=t_inst hidden_states={layer_count} "
            f"layer_scores={expected} finite=true"
        )
        return

    rows, parse_errors, malformed, excluded_pilot, excluded_categories = (
        load_judged_rows(
            Path(args.inp), limit=0, seed=args.seed, keep_protected_group=False
        )
    )
    train_rows, tune_rows, test_rows = split_rows(rows, args.seed)
    print(
        f"[split] train={len(train_rows)} tune={len(tune_rows)} "
        f"test={len(test_rows)}",
        flush=True,
    )

    translations_dir = Path(args.translations_dir)
    translation_path = translations_dir / "swahili.jsonl"
    metadata_path = translations_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata["nllb_model"] != args.nllb:
        raise ValueError("translation model mismatch")
    if metadata["swahili_sha256"] != file_sha256(translation_path):
        raise ValueError("Swahili translation checksum mismatch")

    tune_plain = [row["prompt"] for row in tune_rows]
    test_plain = [row["prompt"] for row in test_rows]
    translated, truncated = load_or_translate(
        tune_plain + test_plain,
        translation_path,
        args.nllb,
        NLLB_CODE,
        allow_translate=False,
    )
    if len(truncated) != metadata["inputs_exceeding_256_tokens"]:
        raise ValueError("translation truncation metadata mismatch")
    tune_conditions = transformed_conditions(
        tune_plain, translated[: len(tune_rows)]
    )
    test_conditions = transformed_conditions(
        test_plain, translated[len(tune_rows) :]
    )

    print(
        f"[features] position=t_inst transformer outputs=1..{num_layers} "
        f"hidden={hidden_size}",
        flush=True,
    )
    print("[extract] plain train", flush=True)
    train_features = extract_features(
        [row["prompt"] for row in train_rows],
        model,
        tokenizer,
        args.batch_size,
        layer_count,
        hidden_size,
        "t_inst",
    )
    print("[extract] plain tune", flush=True)
    tune_features = extract_features(
        tune_plain,
        model,
        tokenizer,
        args.batch_size,
        layer_count,
        hidden_size,
        "t_inst",
    )
    train_labels = np.asarray([int(bool(row["harmful"])) for row in train_rows])
    tune_labels = np.asarray([int(bool(row["harmful"])) for row in tune_rows])
    test_labels = np.asarray([int(bool(row["harmful"])) for row in test_rows])

    train_states = train_features[:, 1:, :]
    tune_states = tune_features[:, 1:, :]
    harmful_centroid = train_states[train_labels == 1].mean(axis=0).astype(np.float32)
    harmless_centroid = train_states[train_labels == 0].mean(axis=0).astype(np.float32)
    weights = np.empty((num_layers, hidden_size), dtype=np.float32)
    intercepts = np.empty(num_layers, dtype=np.float64)
    selected_cs = np.empty(num_layers, dtype=np.float64)
    selected_losses = np.empty(num_layers, dtype=np.float64)
    selected_ses = np.empty(num_layers, dtype=np.float64)
    tuning = []
    for column, hidden_state_index in enumerate(range(1, layer_count)):
        print(
            f"[probe] hidden-state index {hidden_state_index}/{num_layers}",
            flush=True,
        )
        (
            weights[column],
            intercepts[column],
            selected_cs[column],
            selected_losses[column],
            selected_ses[column],
            layer_tuning,
        ) = train_probe(
            train_states[:, column, :].copy(),
            train_labels,
            tune_states[:, column, :].copy(),
            tune_labels,
            args.seed,
        )
        tuning.append(layer_tuning)

    scores = {
        "tune_plain_logistic": (
            np.einsum("nlh,lh->nl", tune_states, weights) + intercepts
        ).astype(np.float32),
        "tune_plain_centroid": per_layer_centroid_scores(
            tune_features, harmful_centroid, harmless_centroid
        ),
    }
    del train_features, train_states, tune_features, tune_states

    for split, conditions in (("tune", tune_conditions), ("test", test_conditions)):
        for condition in CONDITIONS:
            if split == "tune" and condition == "plain":
                continue
            print(f"[score] {split} {condition}", flush=True)
            logistic, centroid = score_layer_detectors(
                conditions[condition],
                model,
                tokenizer,
                args.batch_size,
                weights,
                intercepts,
                harmful_centroid,
                harmless_centroid,
            )
            scores[f"{split}_{condition}_logistic"] = logistic
            scores[f"{split}_{condition}_centroid"] = centroid

    if not all(np.isfinite(value).all() for value in scores.values()):
        raise ValueError("non-finite layerwise score")
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        model=np.asarray(args.model),
        model_revision=np.asarray(
            getattr(model.config, "_commit_hash", None) or ""
        ),
        seed=np.asarray(args.seed),
        position=np.asarray("t_inst"),
        conditions=np.asarray(CONDITIONS),
        layer_indices=np.arange(1, layer_count),
        train_ids=np.asarray([str(row["id"]) for row in train_rows]),
        tune_ids=np.asarray([str(row["id"]) for row in tune_rows]),
        test_ids=np.asarray([str(row["id"]) for row in test_rows]),
        tune_labels=tune_labels,
        test_labels=test_labels,
        test_categories=np.asarray([str(row.get("category")) for row in test_rows]),
        inputs_sha256=np.asarray(
            strings_sha256(
                [
                    text
                    for split_conditions in (tune_conditions, test_conditions)
                    for condition in CONDITIONS
                    for text in split_conditions[condition]
                ]
            )
        ),
        translation_metadata=np.asarray(json.dumps(metadata)),
        input_report=np.asarray(
            json.dumps(
                {
                    "parse_errors_dropped": parse_errors,
                    "malformed_lines_skipped": malformed,
                    "pilot_overlap_excluded": excluded_pilot,
                    "non_operational_positive_categories_excluded": (
                        excluded_categories
                    ),
                }
            )
        ),
        weights=weights,
        intercepts=intercepts,
        harmful_centroid=harmful_centroid,
        harmless_centroid=harmless_centroid,
        selected_cs=selected_cs,
        selected_tune_log_losses=selected_losses,
        selected_tune_log_loss_ses=selected_ses,
        tuning_json=np.asarray([json.dumps(value) for value in tuning]),
        **scores,
    )
    print(f"[done] {output_path}", flush=True)


if __name__ == "__main__":
    main()
