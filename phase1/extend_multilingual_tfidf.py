"""Extend the frozen Phase 1 TF-IDF artefact with French, Hindi and Zulu."""

import argparse
import json
from pathlib import Path

import numpy as np

from capability_check import build_sent
from capability_qa import LANGS
from probe_prompt import (
    file_sha256,
    load_judged_rows,
    load_or_translate,
    split_rows,
    strings_sha256,
)


NEW_CONDITIONS = ("french", "hindi", "zulu")
EXPECTED_SPLIT = (5341, 1781, 1781)
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"
REPRODUCTION_TOLERANCE = 1e-3


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def expected_manifest_hash(metadata, condition):
    expected = metadata.get("manifests", {}).get(condition)
    if isinstance(expected, dict):
        return expected.get("sha256")
    if condition == "swahili":
        return metadata.get("swahili_sha256")
    return expected


def load_translations(translations_dir, tune_rows, test_rows):
    metadata_path = translations_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("nllb_model") != NLLB_MODEL:
        raise ValueError("translation model mismatch")
    if metadata.get("nllb_revision") != NLLB_REVISION:
        raise ValueError("translation revision mismatch")

    prompts = [row["prompt"] for row in tune_rows + test_rows]
    translations = {}
    hashes = {}
    for condition in (*NEW_CONDITIONS, "swahili"):
        path = translations_dir / f"{condition}.jsonl"
        translated, _ = load_or_translate(
            prompts,
            path,
            NLLB_MODEL,
            LANGS[condition],
            allow_translate=False,
        )
        digest = file_sha256(path)
        if digest != expected_manifest_hash(metadata, condition):
            raise ValueError(f"checksum mismatch for frozen {condition} manifest")
        translations[condition] = translated
        hashes[condition] = digest
    return translations, hashes


def build_existing_conditions(rows, swahili):
    plain = [row["prompt"] for row in rows]
    return {
        "plain": plain,
        "swahili": swahili,
        "reverse": [
            build_sent(text, "reverse", in_obf=True, out_obf=False)
            for text in plain
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="data/judged_main_prompts.jsonl")
    parser.add_argument("--source", default="data/phase1_baselines.npz")
    parser.add_argument("--translations-dir", default="data/phase1_translations")
    parser.add_argument(
        "--out", default="data/phase1_baselines_multilingual_tfidf.npz"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.seed != 0:
        raise ValueError("the frozen Phase 1 split requires --seed 0")
    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    rows, _, _, _, _ = load_judged_rows(
        Path(args.inp), 0, args.seed, keep_protected_group=False
    )
    train_rows, tune_rows, test_rows = split_rows(rows, args.seed)
    split_sizes = (len(train_rows), len(tune_rows), len(test_rows))
    if split_sizes != EXPECTED_SPLIT:
        raise ValueError(f"unexpected frozen split sizes: {split_sizes}")

    source_path = Path(args.source)
    source = load_npz(source_path)
    for split, split_rows_ in (("tune", tune_rows), ("test", test_rows)):
        expected_ids = np.asarray([str(row["id"]) for row in split_rows_])
        if not np.array_equal(source[f"{split}_ids"], expected_ids):
            raise ValueError(f"{source_path} has different {split} IDs")

    translations, translation_hashes = load_translations(
        Path(args.translations_dir), tune_rows, test_rows
    )

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=100000,
    )
    train_matrix = vectorizer.fit_transform(
        [row["prompt"] for row in train_rows]
    )
    if len(vectorizer.vocabulary_) != int(source["tfidf_vocabulary_size"]):
        raise ValueError("reconstructed TF-IDF vocabulary size differs")
    train_labels = np.asarray(
        [int(bool(row["harmful"])) for row in train_rows]
    )
    selected_c = float(source["selected_tfidf_c"])
    classifier = LogisticRegression(
        C=selected_c,
        class_weight="balanced",
        random_state=args.seed,
        solver="liblinear",
        max_iter=2000,
    ).fit(train_matrix, train_labels)

    reproduction_errors = {}
    translated_swahili = translations["swahili"]
    for split, split_rows_, swahili in (
        ("tune", tune_rows, translated_swahili[: len(tune_rows)]),
        ("test", test_rows, translated_swahili[len(tune_rows) :]),
    ):
        conditions = build_existing_conditions(split_rows_, swahili)
        for condition, texts in conditions.items():
            reconstructed = classifier.decision_function(
                vectorizer.transform(texts)
            ).astype(np.float32)
            error = float(
                np.max(
                    np.abs(
                        reconstructed - source[f"{split}_{condition}_tfidf"]
                    )
                )
            )
            reproduction_errors[f"{split}_{condition}"] = error
    maximum_error = max(reproduction_errors.values())
    if maximum_error > REPRODUCTION_TOLERANCE:
        raise ValueError(
            f"frozen TF-IDF reconstruction differs by {maximum_error:.6g}; "
            f"tolerance={REPRODUCTION_TOLERANCE:.6g}"
        )
    print(
        f"[reproduction] vocabulary={len(vectorizer.vocabulary_)} C={selected_c:g} "
        f"maximum_existing_logit_error={maximum_error:.6g}",
        flush=True,
    )

    extension_scores = {}
    for split, offset, split_rows_ in (
        ("tune", 0, tune_rows),
        ("test", len(tune_rows), test_rows),
    ):
        for condition in NEW_CONDITIONS:
            texts = translations[condition][
                offset : offset + len(split_rows_)
            ]
            scores = classifier.decision_function(
                vectorizer.transform(texts)
            ).astype(np.float32)
            if scores.shape != (len(split_rows_),) or not np.isfinite(scores).all():
                raise ValueError(f"invalid scores for {split} {condition}")
            extension_scores[f"{split}_{condition}_tfidf"] = scores

    extension_metadata = {
        "source": str(source_path),
        "source_sha256": file_sha256(source_path),
        "seed": args.seed,
        "conditions_added": list(NEW_CONDITIONS),
        "translation_hashes": translation_hashes,
        "training": (
            "deterministic reconstruction on the unchanged plain train split at "
            "the already-selected C; no hyperparameter selection"
        ),
        "selected_C": selected_c,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "existing_score_reproduction_max_abs_errors": reproduction_errors,
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
    }
    output = {
        **source,
        **extension_scores,
        "multilingual_tfidf_extension_json": np.asarray(
            json.dumps(extension_metadata)
        ),
        "multilingual_tfidf_inputs_sha256": np.asarray(
            strings_sha256(
                [
                    text
                    for condition in NEW_CONDITIONS
                    for text in translations[condition]
                ]
            )
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    print(f"[done] {output_path}", flush=True)


if __name__ == "__main__":
    main()
