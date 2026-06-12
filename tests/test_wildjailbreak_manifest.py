from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from agguardrails.wildjailbreak_manifest import (
    DATASET_REVISION,
    ManifestValidationError,
    build_manifest,
    validate_manifest,
)


def test_manifest_is_balanced_and_group_safe() -> None:
    rows, metadata = build_manifest(_source_rows(), seed=17)

    assert len(rows) == 400
    assert Counter((row["harmfulness"], row["prompt_type"]) for row in rows) == {
        ("harmful", "vanilla"): 100,
        ("harmful", "adversarial"): 100,
        ("benign", "vanilla"): 100,
        ("benign", "adversarial"): 100,
    }
    assert Counter(row["split"] for row in rows) == {
        "train": 280,
        "calibration": 60,
        "test": 60,
    }

    splits_by_group = defaultdict(set)
    for row in rows:
        splits_by_group[row["group_id"]].add(row["split"])
    assert all(len(splits) == 1 for splits in splits_by_group.values())
    assert metadata["dataset"]["revision"] == DATASET_REVISION
    assert metadata["counts"]["selected_rows"] == 400


def test_manifest_is_deterministic_and_preserves_lineage() -> None:
    source_rows = _source_rows()
    source_rows[1]["tactics"] = '["roleplay", "encoding"]'
    source_rows[2]["tactics"] = "['fiction']"
    first, first_metadata = build_manifest(source_rows, seed=41)
    second, second_metadata = build_manifest(source_rows, seed=41)

    assert first == second
    assert first_metadata == second_metadata
    adversarial = next(row for row in first if row["prompt_type"] == "adversarial")
    assert adversarial["prompt"] == adversarial["adversarial_prompt"]
    assert adversarial["vanilla_prompt"].startswith("base request")
    assert adversarial["tactics"] in (["roleplay", "encoding"], ["fiction"])
    assert adversarial["source"]["row_index"] >= 0
    assert adversarial["example_id"].endswith(str(adversarial["source"]["row_index"]))


def test_schema_validation_rejects_bad_tactics_and_group_overlap() -> None:
    source_rows = _source_rows()
    source_rows[1]["tactics"] = "not a structured list"
    with pytest.raises(ManifestValidationError, match="structured list"):
        build_manifest(source_rows, seed=3)

    rows, _ = build_manifest(_source_rows(), seed=3)
    leaked = [dict(row) for row in rows]
    group_id = leaked[0]["group_id"]
    partner = next(
        row for row in leaked if row["group_id"] == group_id and row is not leaked[0]
    )
    partner["split"] = "test" if leaked[0]["split"] != "test" else "train"
    with pytest.raises(ManifestValidationError, match="multiple splits"):
        validate_manifest(leaked)

    bad_source = [dict(row) for row in rows]
    bad_source[0]["source"] = {**bad_source[0]["source"], "revision": "wrong"}
    with pytest.raises(ManifestValidationError, match="invalid lineage"):
        validate_manifest(bad_source)


def _source_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in ("harmful", "benign"):
        for group_index in range(120):
            vanilla = f"base request {label} {group_index}"
            rows.extend(
                [
                    {
                        "vanilla": vanilla,
                        "adversarial": "",
                        "tactics": [],
                        "completion": "source completion omitted from manifest",
                        "data_type": f"vanilla_{label}",
                    },
                    {
                        "vanilla": vanilla,
                        "adversarial": f"derived request {label} {group_index} a",
                        "tactics": ["roleplay", "encoding"],
                        "completion": "source completion omitted from manifest",
                        "data_type": f"adversarial_{label}",
                    },
                    {
                        "vanilla": vanilla,
                        "adversarial": f"derived request {label} {group_index} b",
                        "tactics": ["fiction"],
                        "completion": "source completion omitted from manifest",
                        "data_type": f"adversarial_{label}",
                    },
                ]
            )
    return rows
