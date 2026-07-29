"""Score ShieldGemma on the C7 external pool, plain and Swahili.

ShieldGemma is the C4-preselected comparator for the Swahili condition. Same
label-token probability as the frozen Phase 1 matrix, same prompts and partition as
`score_c7_external.py`.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from guard_screen import run_shieldgemma
from phase1.extend_multilingual_guards import (
    SHIELD_MODEL,
    resolve_cached_snapshot,
)
from phase1.phase1_baselines import guard_rows, unpack_guard
from probe_prompt import file_sha256, strings_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judged", default="data/c7_judged_all.jsonl")
    parser.add_argument("--partition", default="data/c7_partition.json")
    parser.add_argument("--translations", default="data/c7_translations/swahili.jsonl")
    parser.add_argument("--manifest", default="data/c7_external_manifest.json")
    parser.add_argument("--out", default="data/c7_external_guard.npz")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="score only N rows per split")
    args = parser.parse_args()

    output_path = Path(args.out)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    partition = json.loads(Path(args.partition).read_text())
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

    splits = {"tune": list(partition["tune_ids"]), "test": list(partition["test_ids"])}
    if args.limit:
        splits = {name: ids[: args.limit] for name, ids in splits.items()}

    snapshot, revision = resolve_cached_snapshot(SHIELD_MODEL)

    scored = []
    conditions_by_split = {}
    for split, ids in splits.items():
        plain = [rows[i]["prompt"] for i in ids]
        conditions = {"plain": plain, "swahili": [swahili[p] for p in plain]}
        conditions_by_split[split] = conditions
        batch = guard_rows(split, conditions)
        print(f"[score] {split} n={len(batch)}", flush=True)
        run_shieldgemma(batch, str(snapshot), args.batch_size)
        scored.extend(batch)

    output = {}
    for split, ids in splits.items():
        for condition in ("plain", "swahili"):
            scores = unpack_guard(scored, split, condition, "score").astype(np.float32)
            if len(scores) != len(ids) or not np.isfinite(scores).all():
                raise ValueError(f"invalid guard scores for {split} {condition}")
            output[f"{split}_{condition}_shieldgemma"] = scores
        output[f"{split}_ids"] = np.asarray(ids)
        output[f"{split}_labels"] = np.asarray(
            [int(bool(rows[i]["harmful"])) for i in ids]
        )

    metadata = {
        "model": SHIELD_MODEL,
        "model_revision": revision,
        "batch_size": args.batch_size,
        "judged_sha256": file_sha256(Path(args.judged)),
        "partition_sha256": file_sha256(Path(args.partition)),
        "swahili_sha256": manifest["swahili"]["sha256"],
        "score": "P(Yes) at the ShieldGemma decision position",
    }
    output["metadata_json"] = np.asarray(json.dumps(metadata))
    output["inputs_sha256"] = np.asarray(
        strings_sha256(
            [
                text
                for split in splits
                for condition in ("plain", "swahili")
                for text in conditions_by_split[split][condition]
            ]
        )
    )
    np.savez_compressed(output_path, **output)
    print(f"[done] {output_path} sha256={file_sha256(output_path)}", flush=True)


if __name__ == "__main__":
    main()
