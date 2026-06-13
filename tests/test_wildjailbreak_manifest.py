from __future__ import annotations

from collections import Counter, defaultdict

import pytest
from datasets import Features, Value

from agguardrails.wildjailbreak_manifest import (
    DATASET_REVISION,
    MANIFEST_SCHEMA_VERSION,
    ManifestValidationError,
    build_manifest,
    validate_manifest,
    validate_source_schema,
)


def test_manifest_is_balanced_and_group_safe(
    pinned_source_rows: list[dict[str, str]],
) -> None:
    rows, metadata = build_manifest(pinned_source_rows, seed=17)

    assert len(rows) == 400
    assert all(
        row["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION for row in rows
    )
    assert all("tactics" not in row for row in rows)
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
    assert metadata["exclusions"]["source_fields_unavailable"] == {
        "tactics": (
            "not present in the pinned train/train.tsv source schema; "
            "not derived or substituted"
        )
    }


def test_manifest_is_deterministic_and_preserves_lineage(
    pinned_source_rows: list[dict[str, str]],
) -> None:
    first, first_metadata = build_manifest(pinned_source_rows, seed=41)
    second, second_metadata = build_manifest(pinned_source_rows, seed=41)

    assert first == second
    assert first_metadata == second_metadata
    adversarial = next(row for row in first if row["prompt_type"] == "adversarial")
    assert adversarial["prompt"] == adversarial["adversarial_prompt"]
    assert adversarial["vanilla_prompt"].startswith("base request")
    assert adversarial["source"]["row_index"] >= 0
    assert adversarial["example_id"].endswith(str(adversarial["source"]["row_index"]))


def test_source_schema_preflight_matches_pinned_dataset_shape() -> None:
    actual_features = Features(
        {
            "vanilla": Value("string"),
            "adversarial": Value("string"),
            "completion": Value("string"),
            "data_type": Value("string"),
        }
    )
    validate_source_schema(actual_features)


def test_source_schema_preflight_rejects_metadata_only() -> None:
    mismatched_features = Features(
        {
            "vanilla": Value("string"),
            "adversarial": Value("string"),
            "tactics": Value("string"),
            "completion": Value("string"),
            "data_type": Value("string"),
        }
    )
    with pytest.raises(ManifestValidationError) as exc_info:
        validate_source_schema(mismatched_features)

    message = str(exc_info.value)
    assert "source schema mismatch" in message
    assert "tactics" in message
    assert "base request" not in message
    assert "source completion" not in message


def test_schema_validation_rejects_group_overlap(
    pinned_source_rows: list[dict[str, str]],
) -> None:
    rows, _ = build_manifest(pinned_source_rows, seed=3)
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

    unexpected_field = [dict(row) for row in rows]
    unexpected_field[0]["tactics"] = ["must not be accepted"]
    with pytest.raises(ManifestValidationError, match="unexpected fields"):
        validate_manifest(unexpected_field)


@pytest.fixture
def pinned_source_rows() -> list[dict[str, str]]:
    """Synthetic values with the exact four-field pinned train schema."""

    rows: list[dict[str, str]] = []
    for label in ("harmful", "benign"):
        for group_index in range(120):
            vanilla = f"base request {label} {group_index}"
            rows.extend(
                [
                    {
                        "vanilla": vanilla,
                        "adversarial": "",
                        "completion": "source completion omitted from manifest",
                        "data_type": f"vanilla_{label}",
                    },
                    {
                        "vanilla": vanilla,
                        "adversarial": f"derived request {label} {group_index} a",
                        "completion": "source completion omitted from manifest",
                        "data_type": f"adversarial_{label}",
                    },
                    {
                        "vanilla": vanilla,
                        "adversarial": f"derived request {label} {group_index} b",
                        "completion": "source completion omitted from manifest",
                        "data_type": f"adversarial_{label}",
                    },
                ]
            )
    return rows
