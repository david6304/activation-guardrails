"""Extend the frozen Phase 1 activation artefact with French, Hindi and Zulu."""

import argparse
import json
from pathlib import Path

import numpy as np

from capability_qa import LANGS
from phase1.phase1_activation import score_detectors
from probe_prompt import (
    file_sha256,
    load_judged_rows,
    load_model,
    load_or_translate,
    split_rows,
    strings_sha256,
)


NEW_CONDITIONS = ("french", "hindi", "zulu")
EXPECTED_MODEL = "google/gemma-3-27b-it"
EXPECTED_REVISION = "005ad3404e59d6023443cb575daa05336842228a"
EXPECTED_SPLIT = (5341, 1781, 1781)
NLLB_MODEL = "facebook/nllb-200-distilled-600M"
NLLB_REVISION = "f8d333a098d19b4fd9a8b18f94170487ad3f821d"


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


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
    for condition in NEW_CONDITIONS:
        path = translations_dir / f"{condition}.jsonl"
        translated, _ = load_or_translate(
            prompts,
            path,
            NLLB_MODEL,
            LANGS[condition],
            allow_translate=False,
        )
        digest = file_sha256(path)
        expected = metadata.get("manifests", {}).get(condition)
        if isinstance(expected, dict):
            expected = expected.get("sha256")
        if digest != expected:
            raise ValueError(f"checksum mismatch for frozen {condition} manifest")
        translations[condition] = translated
        hashes[condition] = digest
    return translations, hashes, metadata


def validate_source(source, source_path, train_rows, tune_rows, test_rows, seed):
    expected_ids = {
        "train_ids": np.asarray([str(row["id"]) for row in train_rows]),
        "tune_ids": np.asarray([str(row["id"]) for row in tune_rows]),
        "test_ids": np.asarray([str(row["id"]) for row in test_rows]),
    }
    for key, values in expected_ids.items():
        if not np.array_equal(source[key], values):
            raise ValueError(f"{source_path} has different {key}")
    if str(source["model"]) != EXPECTED_MODEL:
        raise ValueError("source activation model mismatch")
    if str(source["model_revision"]) != EXPECTED_REVISION:
        raise ValueError("source activation revision mismatch")
    if str(source["position"]) != "t_inst":
        raise ValueError("source activation position is not t_inst")
    if int(source["seed"]) != seed:
        raise ValueError("source activation seed mismatch")
    for split, rows in (("tune", tune_rows), ("test", test_rows)):
        labels = np.asarray([int(bool(row["harmful"])) for row in rows])
        if not np.array_equal(source[f"{split}_labels"], labels):
            raise ValueError(f"source activation {split} labels differ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="data/judged_main_prompts.jsonl")
    parser.add_argument("--source", default="data/phase1_activation_27b.npz")
    parser.add_argument("--translations-dir", default="data/phase1_translations")
    parser.add_argument(
        "--out", default="data/phase1_activation_multilingual_27b.npz"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.seed != 0:
        raise ValueError("the frozen Phase 1 split requires --seed 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
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
    validate_source(
        source, source_path, train_rows, tune_rows, test_rows, args.seed
    )
    translations, translation_hashes, translation_metadata = load_translations(
        Path(args.translations_dir), tune_rows, test_rows
    )
    tune_conditions = {
        condition: translations[condition][: len(tune_rows)]
        for condition in NEW_CONDITIONS
    }
    test_conditions = {
        condition: translations[condition][len(tune_rows) :]
        for condition in NEW_CONDITIONS
    }

    model, tokenizer, num_layers, hidden_size = load_model(EXPECTED_MODEL, args.seed)
    loaded_revision = str(getattr(model.config, "_commit_hash", "") or "")
    if loaded_revision != EXPECTED_REVISION:
        raise RuntimeError(
            f"loaded Gemma revision {loaded_revision!r} != {EXPECTED_REVISION!r}"
        )
    expected_dimensions = (num_layers + 1) * hidden_size
    if source["logistic_weight"].shape != (expected_dimensions,):
        raise ValueError("frozen logistic weight dimension does not match the model")
    if source["harmful_centroid"].shape != (num_layers, hidden_size):
        raise ValueError("frozen harmful centroid shape does not match the model")
    if source["harmless_centroid"].shape != (num_layers, hidden_size):
        raise ValueError("frozen harmless centroid shape does not match the model")

    extension_scores = {}
    for split, conditions in (("tune", tune_conditions), ("test", test_conditions)):
        for condition, texts in conditions.items():
            print(f"[score] {split} {condition}", flush=True)
            logistic, centroid = score_detectors(
                texts,
                model,
                tokenizer,
                args.batch_size,
                source["logistic_weight"],
                float(source["logistic_intercept"]),
                source["harmful_centroid"],
                source["harmless_centroid"],
                "t_inst",
            )
            if (
                logistic.shape != (len(texts),)
                or centroid.shape != (len(texts),)
                or not np.isfinite(logistic).all()
                or not np.isfinite(centroid).all()
            ):
                raise ValueError(f"invalid scores for {split} {condition}")
            extension_scores[f"{split}_{condition}_logistic"] = logistic
            extension_scores[f"{split}_{condition}_centroid"] = centroid

    extension_metadata = {
        "source": str(source_path),
        "source_sha256": file_sha256(source_path),
        "model": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "position": "t_inst",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "conditions_added": list(NEW_CONDITIONS),
        "translation_hashes": translation_hashes,
        "translation_metadata": translation_metadata,
        "training": "none; reused frozen folded logistic and centroid parameters",
        "scoring": "only missing tune/test conditions were forwarded",
    }
    output = {
        **source,
        **extension_scores,
        "multilingual_extension_json": np.asarray(
            json.dumps(extension_metadata)
        ),
        "multilingual_extension_inputs_sha256": np.asarray(
            strings_sha256(
                [
                    text
                    for conditions in (tune_conditions, test_conditions)
                    for condition in NEW_CONDITIONS
                    for text in conditions[condition]
                ]
            )
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    print(f"[done] {output_path}", flush=True)


if __name__ == "__main__":
    main()
