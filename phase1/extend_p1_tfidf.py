"""Score the frozen char TF-IDF baseline on the P1 cipher conditions (CPU).

The vectoriser and classifier are reconstructed deterministically on the
unchanged plain train split at the already-selected C, and the reconstruction is
checked against the frozen plain/swahili/reverse scores before anything new is
scored. Reproducing `reverse` also confirms that the condition strings rebuilt by
`prepare_p1_conditions.py` are byte-identical to the frozen ones.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from phase1.extend_multilingual_tfidf import (
    REPRODUCTION_TOLERANCE,
    build_existing_conditions,
    load_npz,
    load_translations,
)
from phase1.prepare_p1_conditions import CONDITIONS, build_p1_conditions
from probe_prompt import (
    file_sha256,
    load_judged_rows,
    split_rows,
    strings_sha256,
)


EXPECTED_SPLIT = (5341, 1781, 1781)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="data/judged_main_prompts.jsonl")
    parser.add_argument("--source", default="data/phase1_baselines_multilingual.npz")
    parser.add_argument("--translations-dir", default="data/phase1_translations")
    parser.add_argument("--manifest", default="data/p1_conditions_manifest.json")
    parser.add_argument("--out", default="data/p1_baselines_tfidf.npz")
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

    plaintexts = [row["prompt"] for row in tune_rows + test_rows]
    conditions, _ = build_p1_conditions(plaintexts, args.seed)
    manifest = json.loads(Path(args.manifest).read_text())
    for condition in CONDITIONS:
        if strings_sha256(conditions[condition]) != manifest["strings_sha256"][condition]:
            raise ValueError(f"reconstructed {condition} does not match the manifest")

    translations, _ = load_translations(
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
    train_matrix = vectorizer.fit_transform([row["prompt"] for row in train_rows])
    if len(vectorizer.vocabulary_) != int(source["tfidf_vocabulary_size"]):
        raise ValueError("reconstructed TF-IDF vocabulary size differs")
    train_labels = np.asarray([int(bool(row["harmful"])) for row in train_rows])
    selected_c = float(source["selected_tfidf_c"])
    classifier = LogisticRegression(
        C=selected_c,
        class_weight="balanced",
        random_state=args.seed,
        solver="liblinear",
        max_iter=2000,
    ).fit(train_matrix, train_labels)

    reproduction_errors = {}
    swahili = translations["swahili"]
    for split, split_rows_, split_swahili in (
        ("tune", tune_rows, swahili[: len(tune_rows)]),
        ("test", test_rows, swahili[len(tune_rows) :]),
    ):
        for condition, texts in build_existing_conditions(
            split_rows_, split_swahili
        ).items():
            reconstructed = classifier.decision_function(
                vectorizer.transform(texts)
            ).astype(np.float32)
            reproduction_errors[f"{split}_{condition}"] = float(
                np.max(np.abs(reconstructed - source[f"{split}_{condition}_tfidf"]))
            )
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

    scores = {}
    for split, offset, count in (
        ("tune", 0, len(tune_rows)),
        ("test", len(tune_rows), len(test_rows)),
    ):
        for condition in CONDITIONS:
            texts = conditions[condition][offset : offset + count]
            values = classifier.decision_function(
                vectorizer.transform(texts)
            ).astype(np.float32)
            if values.shape != (count,) or not np.isfinite(values).all():
                raise ValueError(f"invalid scores for {split} {condition}")
            scores[f"{split}_{condition}_tfidf"] = values

    metadata = {
        "source": str(source_path),
        "source_sha256": file_sha256(source_path),
        "conditions_manifest_sha256": file_sha256(Path(args.manifest)),
        "seed": args.seed,
        "conditions_added": list(CONDITIONS),
        "training": (
            "deterministic reconstruction on the unchanged plain train split at "
            "the already-selected C; no hyperparameter selection"
        ),
        "selected_C": selected_c,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "existing_score_reproduction_max_abs_errors": reproduction_errors,
        "reproduction_tolerance": REPRODUCTION_TOLERANCE,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        tune_ids=source["tune_ids"],
        test_ids=source["test_ids"],
        p1_tfidf_json=np.asarray(json.dumps(metadata)),
        p1_tfidf_inputs_sha256=np.asarray(
            strings_sha256(
                [text for condition in CONDITIONS for text in conditions[condition]]
            )
        ),
        **scores,
    )
    print(f"[done] {output_path}", flush=True)


if __name__ == "__main__":
    main()
