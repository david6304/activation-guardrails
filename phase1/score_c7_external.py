"""Score the frozen probe and centroid on the C7 external pool, plain and Swahili.

No training, no layer reselection: the logistic weight, intercept and both centroids come
straight out of the frozen Phase 1 artefact. One 27B forward pass, score-forward only.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from phase1.extend_multilingual_activation import (
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    load_npz,
)
from phase1.phase1_activation import score_detectors
from probe_prompt import file_sha256, load_model, strings_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", default="data/c7_judged_all.jsonl")
    parser.add_argument("--partition", default="data/c7_partition.json")
    parser.add_argument("--translations", default="data/c7_translations/swahili.jsonl")
    parser.add_argument("--manifest", default="data/c7_external_manifest.json")
    parser.add_argument("--source", default="data/phase1_activation_multilingual_27b.npz")
    parser.add_argument("--out", default="data/c7_external_scores.npz")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="score only N rows per split")
    args = parser.parse_args()

    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    partition = json.loads(Path(args.partition).read_text())
    tune_ids = list(partition["tune_ids"])
    test_ids = list(partition["test_ids"])

    rows = {}
    with Path(args.judged).open() as f:
        for line in f:
            row = json.loads(line)
            rows[row["id"]] = row

    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["swahili"]["sha256"] != file_sha256(Path(args.translations)):
        raise ValueError("Swahili manifest checksum does not match the frozen manifest")
    swahili = {}
    with Path(args.translations).open() as f:
        for line in f:
            entry = json.loads(line)
            swahili[entry["prompt"]] = entry["translation"]

    splits = {"tune": tune_ids, "test": test_ids}
    if args.limit:
        splits = {name: ids[: args.limit] for name, ids in splits.items()}

    conditions = {}
    labels = {}
    for name, ids in splits.items():
        plain = [rows[i]["prompt"] for i in ids]
        conditions[(name, "plain")] = plain
        conditions[(name, "swahili")] = [swahili[p] for p in plain]
        labels[name] = np.asarray([int(bool(rows[i]["harmful"])) for i in ids])

    source = load_npz(Path(args.source))
    model, tokenizer, num_layers, hidden_size = load_model(EXPECTED_MODEL, args.seed)
    loaded_revision = str(getattr(model.config, "_commit_hash", "") or "")
    if loaded_revision != EXPECTED_REVISION:
        raise RuntimeError(
            f"loaded Gemma revision {loaded_revision!r} != {EXPECTED_REVISION!r}"
        )
    if source["logistic_weight"].shape != ((num_layers + 1) * hidden_size,):
        raise ValueError("frozen logistic weight does not match the model")

    scores = {}
    for (split, condition), texts in conditions.items():
        print(f"[score] {split} {condition} n={len(texts)}", flush=True)
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
        if not np.isfinite(logistic).all() or not np.isfinite(centroid).all():
            raise ValueError(f"non-finite scores for {split} {condition}")
        scores[f"{split}_{condition}_logistic"] = logistic
        scores[f"{split}_{condition}_centroid"] = centroid

    metadata = {
        "source": str(args.source),
        "source_sha256": file_sha256(Path(args.source)),
        "model": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "position": "t_inst",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "judged_sha256": file_sha256(Path(args.judged)),
        "partition_sha256": file_sha256(Path(args.partition)),
        "swahili_sha256": manifest["swahili"]["sha256"],
        "training": "none; frozen Phase 1 logistic and centroid parameters",
    }
    output = {
        **scores,
        "tune_ids": np.asarray(splits["tune"]),
        "test_ids": np.asarray(splits["test"]),
        "tune_labels": labels["tune"],
        "test_labels": labels["test"],
        "logistic_weight": source["logistic_weight"],
        "logistic_intercept": source["logistic_intercept"],
        "metadata_json": np.asarray(json.dumps(metadata)),
        "inputs_sha256": np.asarray(
            strings_sha256([text for texts in conditions.values() for text in texts])
        ),
    }
    np.savez_compressed(output_path, **output)
    print(f"[done] {output_path} sha256={file_sha256(output_path)}", flush=True)


if __name__ == "__main__":
    main()
