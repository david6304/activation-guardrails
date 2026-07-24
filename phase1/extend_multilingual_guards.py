"""Extend the frozen Phase 1 guard artefact with French, Hindi and Zulu."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from phase1.extend_multilingual_tfidf import load_translations
from phase1.phase1_baselines import (
    SHIELD_POLICY,
    build_conditions,
    guard_rows,
    unpack_guard,
)
from probe_prompt import file_sha256, load_judged_rows, split_rows, strings_sha256


NEW_CONDITIONS = ("french", "hindi", "zulu")
OLD_CONDITIONS = ("plain", "swahili", "reverse")
CONDITIONS = ("plain", "french", "hindi", "swahili", "zulu", "reverse")
AUDIT_INDICES = (0, 254, 508, 763, 1017, 1272, 1526, 1780)
SHIELD_MODEL = "google/shieldgemma-9b"
WILDGUARD_MODEL = "allenai/wildguard"
SHIELD_AUDIT_TOLERANCE = 1e-3
EXPECTED_SPLIT = (5341, 1781, 1781)


def load_npz(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def resolve_cached_snapshot(model_id):
    from huggingface_hub.constants import HF_HUB_CACHE

    cache_dir = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    revision = (cache_dir / "refs" / "main").read_text().strip()
    if len(revision) != 40:
        raise RuntimeError(f"could not resolve cached revision for {model_id}")
    snapshot = cache_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"missing cached snapshot {snapshot}")
    return snapshot, revision


def validate_sources(source, frozen, source_path, frozen_path, tune_rows, test_rows):
    for split, rows in (("tune", tune_rows), ("test", test_rows)):
        expected_ids = np.asarray([str(row["id"]) for row in rows])
        if not np.array_equal(frozen[f"{split}_ids"], expected_ids):
            raise ValueError(f"{frozen_path} has different {split} IDs")
        if not np.array_equal(source[f"{split}_ids"], expected_ids):
            raise ValueError(f"{source_path} has different {split} IDs")
        for condition in OLD_CONDITIONS:
            for detector in ("shieldgemma", "wildguard"):
                key = f"{split}_{condition}_{detector}"
                if key not in frozen:
                    if detector == "wildguard" and split == "tune":
                        continue
                    raise ValueError(f"missing frozen score {key}")
                if key in source and not np.array_equal(source[key], frozen[key]):
                    raise ValueError(f"{source_path} changed frozen score {key}")
    for split in ("tune", "test"):
        for condition in NEW_CONDITIONS:
            for detector in ("shieldgemma", "wildguard"):
                key = f"{split}_{condition}_{detector}"
                if key in source:
                    raise ValueError(f"{source_path} already contains {key}")


def shield_audit(rows, frozen):
    lookup = {
        (row["split"], row["condition"], row["id"]): row for row in rows
    }
    differences = {}
    for split in ("tune", "test"):
        for condition in OLD_CONDITIONS:
            key = f"{split}_{condition}_shieldgemma"
            for index in AUDIT_INDICES:
                difference = abs(
                    float(lookup[(split, condition, index)]["score"])
                    - float(frozen[key][index])
                )
                differences[f"{split}:{condition}:{index}"] = difference
    maximum = max(differences.values())
    return {
        "comparison": "absolute probability difference",
        "tolerance": SHIELD_AUDIT_TOLERANCE,
        "maximum_difference": maximum,
        "differences": differences,
        "passed": maximum <= SHIELD_AUDIT_TOLERANCE,
    }


def wildguard_audit(rows, frozen):
    lookup = {(row["condition"], row["id"]): row for row in rows}
    mismatches = []
    for condition in OLD_CONDITIONS:
        key = f"test_{condition}_wildguard"
        for index in AUDIT_INDICES:
            if bool(lookup[(condition, index)]["flag"]) != bool(frozen[key][index]):
                mismatches.append(f"{condition}:{index}")
    return {
        "comparison": "exact native-decision equality",
        "mismatches": mismatches,
        "passed": not mismatches,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="data/judged_main_prompts.jsonl")
    parser.add_argument(
        "--source", default="data/phase1_baselines_multilingual_tfidf.npz"
    )
    parser.add_argument("--frozen-source", default="data/phase1_baselines.npz")
    parser.add_argument("--metadata", default="data/phase1_baselines.json")
    parser.add_argument("--translations-dir", default="data/phase1_translations")
    parser.add_argument(
        "--out", default="data/phase1_baselines_multilingual.npz"
    )
    parser.add_argument("--shield-batch-size", type=int, default=8)
    parser.add_argument("--wildguard-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.seed != 0:
        raise ValueError("the frozen Phase 1 split requires --seed 0")
    if args.shield_batch_size <= 0 or args.wildguard_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
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
    frozen_path = Path(args.frozen_source)
    source = load_npz(source_path)
    frozen = load_npz(frozen_path)
    validate_sources(
        source, frozen, source_path, frozen_path, tune_rows, test_rows
    )
    baseline_metadata = json.loads(Path(args.metadata).read_text())
    if baseline_metadata.get("shieldgemma_policy") != SHIELD_POLICY:
        raise ValueError("frozen ShieldGemma policy mismatch")

    translations, translation_hashes = load_translations(
        Path(args.translations_dir), tune_rows, test_rows
    )
    tune_swahili = translations["swahili"][: len(tune_rows)]
    test_swahili = translations["swahili"][len(tune_rows) :]
    old_tune_conditions = build_conditions(tune_rows, tune_swahili)
    old_test_conditions = build_conditions(test_rows, test_swahili)
    new_tune_conditions = {
        condition: translations[condition][: len(tune_rows)]
        for condition in NEW_CONDITIONS
    }
    new_test_conditions = {
        condition: translations[condition][len(tune_rows) :]
        for condition in NEW_CONDITIONS
    }
    tune_conditions = {
        condition: {**old_tune_conditions, **new_tune_conditions}[condition]
        for condition in CONDITIONS
    }
    test_conditions = {
        condition: {**old_test_conditions, **new_test_conditions}[condition]
        for condition in CONDITIONS
    }

    shield_snapshot, shield_revision = resolve_cached_snapshot(SHIELD_MODEL)
    wildguard_snapshot, wildguard_revision = resolve_cached_snapshot(
        WILDGUARD_MODEL
    )
    print(
        f"[models] ShieldGemma={shield_revision} WildGuard={wildguard_revision}",
        flush=True,
    )

    import guard_screen

    guard_screen.SG_GUIDELINE = SHIELD_POLICY
    shield_rows = guard_rows("tune", tune_conditions) + guard_rows(
        "test", test_conditions
    )
    print(
        f"[ShieldGemma] rows={len(shield_rows)} full_rescore=true",
        flush=True,
    )
    guard_screen.run_shieldgemma(
        shield_rows, str(shield_snapshot), args.shield_batch_size
    )
    shield_audit_result = shield_audit(shield_rows, frozen)
    print(
        f"[audit] ShieldGemma passed={shield_audit_result['passed']} "
        f"max_difference={shield_audit_result['maximum_difference']:.6g}",
        flush=True,
    )

    wildguard_rows = guard_rows("test", test_conditions)
    print(
        f"[WildGuard] rows={len(wildguard_rows)} full_rescore=true",
        flush=True,
    )
    guard_screen.run_wildguard(
        wildguard_rows, str(wildguard_snapshot), args.wildguard_batch_size
    )
    wildguard_audit_result = wildguard_audit(wildguard_rows, frozen)
    print(
        f"[audit] WildGuard passed={wildguard_audit_result['passed']} "
        f"mismatches={len(wildguard_audit_result['mismatches'])}",
        flush=True,
    )

    extension_scores = {}
    for split, conditions in (
        ("tune", tune_conditions),
        ("test", test_conditions),
    ):
        for condition in conditions:
            scores = unpack_guard(
                shield_rows, split, condition, "score"
            ).astype(np.float32)
            if scores.shape != (len(tune_rows),) or not np.isfinite(scores).all():
                raise ValueError(f"invalid ShieldGemma scores for {split} {condition}")
            extension_scores[f"{split}_{condition}_shieldgemma"] = scores
    for condition in CONDITIONS:
        flags = unpack_guard(
            wildguard_rows, "test", condition, "flag"
        ).astype(bool)
        if flags.shape != (len(test_rows),):
            raise ValueError(f"invalid WildGuard flags for {condition}")
        extension_scores[f"test_{condition}_wildguard"] = flags

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    import torch

    extension_metadata = {
        "source": str(source_path),
        "source_sha256": file_sha256(source_path),
        "frozen_audit_source": str(frozen_path),
        "frozen_audit_source_sha256": file_sha256(frozen_path),
        "source_commit": commit,
        "seed": args.seed,
        "conditions_scored": list(CONDITIONS),
        "translation_hashes": translation_hashes,
        "shieldgemma": {
            "model": SHIELD_MODEL,
            "revision": shield_revision,
            "policy": SHIELD_POLICY,
            "batch_size": args.shield_batch_size,
            "audit": shield_audit_result,
        },
        "wildguard": {
            "model": WILDGUARD_MODEL,
            "revision": wildguard_revision,
            "batch_size": args.wildguard_batch_size,
            "audit": wildguard_audit_result,
        },
        "audit_indices": list(AUDIT_INDICES),
        "audit_conditions": list(OLD_CONDITIONS),
        "cuda_device": torch.cuda.get_device_name(),
        "model_dtype": "bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        "training": "none",
        "scoring": (
            "all six guard conditions rescored in one environment; frozen old "
            "scores are retained only as an equivalence diagnostic"
        ),
    }
    output = {
        **source,
        **extension_scores,
        "multilingual_guard_extension_json": np.asarray(
            json.dumps(extension_metadata)
        ),
        "multilingual_guard_inputs_sha256": np.asarray(
            strings_sha256(
                [
                    text
                    for conditions in (tune_conditions, test_conditions)
                    for condition in CONDITIONS
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
