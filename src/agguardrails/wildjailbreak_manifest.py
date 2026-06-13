"""Build the approved non-reportable WildJailbreak smoke manifest."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DATASET_ID = "allenai/wildjailbreak"
DATASET_REVISION = "5ddc12a7894f842b0619b8e1c7ee496b198af009"
SOURCE_CONFIG = "train"
SOURCE_SPLIT = "train"
MANIFEST_SCHEMA_VERSION = 2

DATA_TYPES = {
    "vanilla_harmful": ("harmful", "vanilla"),
    "adversarial_harmful": ("harmful", "adversarial"),
    "vanilla_benign": ("benign", "vanilla"),
    "adversarial_benign": ("benign", "adversarial"),
}
LABELS = ("harmful", "benign")
PROMPT_TYPES = ("vanilla", "adversarial")
SPLIT_GROUP_COUNTS = {"train": 70, "calibration": 15, "test": 15}
SOURCE_SCHEMA = {
    "vanilla": "text",
    "adversarial": "text",
    "completion": "text",
    "data_type": "text",
}
TEXT_DTYPES = {"string", "large_string"}
SOURCE_FIELDS = set(SOURCE_SCHEMA)
MANIFEST_FIELDS = {
    "manifest_schema_version",
    "example_id",
    "group_id",
    "split",
    "harmfulness",
    "prompt_type",
    "data_type",
    "prompt",
    "vanilla_prompt",
    "adversarial_prompt",
    "source",
}


class ManifestValidationError(ValueError):
    """Raised when source rows or a manifest violate the data contract."""


def validate_source_schema(features: Mapping[str, Any]) -> None:
    """Fail before row iteration when the pinned dataset schema changes."""

    observed_fields = set(features)
    observed_dtypes = {
        name: str(getattr(feature, "dtype", type(feature).__name__))
        for name, feature in features.items()
    }
    invalid_text_fields = {
        name: observed_dtypes[name]
        for name in SOURCE_FIELDS & observed_fields
        if observed_dtypes[name] not in TEXT_DTYPES
    }
    if observed_fields != SOURCE_FIELDS or invalid_text_fields:
        raise ManifestValidationError(
            "source schema mismatch: "
            f"expected fields {sorted(SOURCE_FIELDS)} with text values; "
            f"observed fields {sorted(observed_fields)}"
            + (
                f", non-text fields {dict(sorted(invalid_text_fields.items()))}"
                if invalid_text_fields
                else ""
            )
        )


def build_manifest(
    source_rows: Iterable[Mapping[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select paired lineage groups and assign deterministic group-safe splits."""

    groups = {label: defaultdict(lambda: defaultdict(list)) for label in LABELS}
    source_counts: Counter[str] = Counter()
    malformed_counts: Counter[str] = Counter()
    for row_index, source_row in enumerate(source_rows):
        data_type = source_row.get("data_type")
        if data_type in DATA_TYPES:
            source_counts[data_type] += 1
        row = _normalize_source_row(source_row, row_index)
        if row is None:
            # The pinned source contains a few labelled rows with no usable prompt.
            malformed_counts[f"{data_type}:invalid_lineage"] += 1
            continue
        groups[row["harmfulness"]][row["group_id"]][row["prompt_type"]].append(row)

    overlap = set(groups["harmful"]) & set(groups["benign"])
    conflicting_rows = sum(
        sum(len(rows) for rows in groups[label][group_id].values())
        for group_id in overlap
        for label in LABELS
    )
    # Conflicting source labels cannot define a leakage-safe group.
    for group_id in overlap:
        for label in LABELS:
            del groups[label][group_id]

    manifest: list[dict[str, Any]] = []
    stats: dict[str, dict[str, int]] = {}
    for label in LABELS:
        label_groups = groups[label]
        eligible = [
            group_id
            for group_id, styles in label_groups.items()
            if styles["vanilla"] and styles["adversarial"]
        ]
        eligible = _ordered(eligible, seed, f"select:{label}")
        if len(eligible) < 100:
            raise ManifestValidationError(
                f"{label} has {len(eligible)} eligible groups; 100 required"
            )

        selected = eligible[:100]
        splits = _split_assignments(selected, seed, label)
        extra_rows = 0
        for group_id in selected:
            styles = label_groups[group_id]
            chosen = (
                _choose(styles["vanilla"], seed, f"vanilla:{group_id}"),
                _choose(styles["adversarial"], seed, f"adversarial:{group_id}"),
            )
            extra_rows += sum(map(len, styles.values())) - 2
            manifest.extend({**row, "split": splits[group_id]} for row in chosen)

        stats[label] = {
            "eligible": len(eligible),
            "missing_vanilla": _missing_count(label_groups, "vanilla"),
            "missing_adversarial": _missing_count(label_groups, "adversarial"),
            "extra_selected_rows": extra_rows,
        }

    split_order = {name: index for index, name in enumerate(SPLIT_GROUP_COUNTS)}
    manifest.sort(
        key=lambda row: (
            split_order[row["split"]],
            row["harmfulness"],
            row["group_id"],
            row["prompt_type"],
        )
    )
    validate_manifest(manifest)
    return manifest, _provenance(
        manifest,
        source_counts,
        malformed_counts,
        len(overlap),
        conflicting_rows,
        stats,
        seed,
    )


def validate_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    """Validate row schema, balance, and absence of lineage leakage."""

    if len(rows) != 400:
        raise ManifestValidationError(f"expected 400 rows, got {len(rows)}")

    quadrants: Counter[tuple[str, str]] = Counter()
    split_quadrants: Counter[tuple[str, str, str]] = Counter()
    splits_by_group: dict[str, str] = {}
    for index, row in enumerate(rows):
        missing = MANIFEST_FIELDS - row.keys()
        if missing:
            raise ManifestValidationError(f"row {index} missing {sorted(missing)}")
        unexpected = row.keys() - MANIFEST_FIELDS
        if unexpected:
            raise ManifestValidationError(
                f"row {index} has unexpected fields {sorted(unexpected)}"
            )

        label, prompt_type, split = (
            row["harmfulness"],
            row["prompt_type"],
            row["split"],
        )
        if (
            row["manifest_schema_version"] != MANIFEST_SCHEMA_VERSION
            or (label, prompt_type) not in DATA_TYPES.values()
            or split not in SPLIT_GROUP_COUNTS
        ):
            raise ManifestValidationError(f"row {index} has invalid labels or version")
        if not isinstance(row["source"], dict) or "row_index" not in row["source"]:
            raise ManifestValidationError(f"row {index} has invalid source identity")

        source = row["source"]
        expected_source = {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "config": SOURCE_CONFIG,
            "split": SOURCE_SPLIT,
            "row_index": source["row_index"],
        }
        vanilla = row["vanilla_prompt"]
        expected_prompt = (
            vanilla if prompt_type == "vanilla" else row["adversarial_prompt"]
        )
        if (
            not isinstance(vanilla, str)
            or not vanilla
            or source != expected_source
            or not isinstance(source["row_index"], int)
            or row["prompt"] != expected_prompt
            or row["group_id"] != _group_id(vanilla)
            or DATA_TYPES.get(row["data_type"]) != (label, prompt_type)
            or row["example_id"] != _example_id(source["row_index"])
        ):
            raise ManifestValidationError(f"row {index} has invalid lineage")

        previous = splits_by_group.setdefault(row["group_id"], split)
        if previous != split:
            raise ManifestValidationError(
                f"group {row['group_id']} appears in multiple splits"
            )
        quadrants[label, prompt_type] += 1
        split_quadrants[split, label, prompt_type] += 1

    expected_quadrants = Counter(
        {(label, prompt_type): 100 for label in LABELS for prompt_type in PROMPT_TYPES}
    )
    expected_split_quadrants = Counter(
        {
            (split, label, prompt_type): count
            for split, count in SPLIT_GROUP_COUNTS.items()
            for label in LABELS
            for prompt_type in PROMPT_TYPES
        }
    )
    if quadrants != expected_quadrants:
        raise ManifestValidationError("quadrant counts are invalid")
    if split_quadrants != expected_split_quadrants:
        raise ManifestValidationError("split quadrant counts are invalid")


def write_manifest(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    output_path: Path,
    metadata_path: Path,
) -> None:
    """Write JSONL plus compact provenance without logging row contents."""

    validate_manifest(rows)
    serialized = "".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n" for row in rows
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(serialized, encoding="utf-8")

    provenance = {
        **metadata,
        "manifest_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _normalize_source_row(row: Mapping[str, Any], index: int) -> dict[str, Any] | None:
    missing = SOURCE_FIELDS - row.keys()
    if missing:
        raise ManifestValidationError(f"source row {index} missing {sorted(missing)}")
    try:
        harmfulness, prompt_type = DATA_TYPES[row["data_type"]]
    except (KeyError, TypeError) as exc:
        raise ManifestValidationError(
            f"source row {index} has invalid data_type"
        ) from exc

    vanilla, adversarial = row["vanilla"], row["adversarial"]
    invalid = (
        not isinstance(vanilla, str)
        or not vanilla.strip()
        or not isinstance(adversarial, str)
        or (prompt_type == "vanilla" and bool(adversarial))
        or (prompt_type == "adversarial" and not adversarial.strip())
    )
    if invalid:
        return None

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "example_id": _example_id(index),
        "group_id": _group_id(vanilla),
        "harmfulness": harmfulness,
        "prompt_type": prompt_type,
        "data_type": row["data_type"],
        "prompt": vanilla if prompt_type == "vanilla" else adversarial,
        "vanilla_prompt": vanilla,
        "adversarial_prompt": adversarial or None,
        "source": {
            "dataset_id": DATASET_ID,
            "revision": DATASET_REVISION,
            "config": SOURCE_CONFIG,
            "split": SOURCE_SPLIT,
            "row_index": index,
        },
    }


def _provenance(
    manifest: Sequence[Mapping[str, Any]],
    source_counts: Counter[str],
    malformed_counts: Counter[str],
    conflicting_groups: int,
    conflicting_rows: int,
    stats: Mapping[str, Mapping[str, int]],
    seed: int,
) -> dict[str, Any]:
    selected = Counter(row["data_type"] for row in manifest)
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "classification": "exploratory_non_reportable_smoke",
        "dataset": {
            "id": DATASET_ID,
            "revision": DATASET_REVISION,
            "config": SOURCE_CONFIG,
            "split": SOURCE_SPLIT,
        },
        "seed": seed,
        "counts": {
            "source_rows": sum(source_counts.values()),
            "source_quadrants": dict(sorted(source_counts.items())),
            "eligible_groups": {label: stats[label]["eligible"] for label in LABELS},
            "selected_rows": len(manifest),
            "selected_quadrants": dict(sorted(selected.items())),
            "split_rows": dict(
                sorted(Counter(row["split"] for row in manifest).items())
            ),
        },
        "exclusions": {
            "conflicting_label_groups": conflicting_groups,
            "rows_in_conflicting_label_groups": conflicting_rows,
            "malformed_source_rows": dict(sorted(malformed_counts.items())),
            "source_rows_not_selected": {
                kind: count - selected[kind]
                for kind, count in sorted(source_counts.items())
            },
            "eligible_groups_not_selected": {
                label: stats[label]["eligible"] - 100 for label in LABELS
            },
            "groups_missing_vanilla": {
                label: stats[label]["missing_vanilla"] for label in LABELS
            },
            "groups_missing_adversarial": {
                label: stats[label]["missing_adversarial"] for label in LABELS
            },
            "extra_rows_in_selected_groups": {
                label: stats[label]["extra_selected_rows"] for label in LABELS
            },
            "source_fields_omitted": {
                "completion": "response text is outside this prompt manifest"
            },
            "source_fields_unavailable": {
                "tactics": (
                    "not present in the pinned train/train.tsv source schema; "
                    "not derived or substituted"
                )
            },
        },
        "group_definition": (
            "group_id is wj-v1- plus SHA-256 of the exact source vanilla string"
        ),
        "selection_rule": (
            "Seeded hash-order 100 eligible groups per label; retain one vanilla "
            "and one seeded hash-ordered adversarial derivative per group"
        ),
        "split_definition": {
            "algorithm": (
                "Seeded hash-order groups per label; first 70 train, next 15 "
                "calibration, final 15 test"
            ),
            "group_counts_per_label": SPLIT_GROUP_COUNTS,
        },
    }


def _group_id(vanilla: str) -> str:
    return f"wj-v1-{hashlib.sha256(vanilla.encode()).hexdigest()}"


def _example_id(row_index: int) -> str:
    return f"{DATASET_ID}@{DATASET_REVISION}:{SOURCE_CONFIG}:{SOURCE_SPLIT}:{row_index}"


def _missing_count(groups: Mapping[str, Mapping[str, Any]], prompt_type: str) -> int:
    return sum(not styles[prompt_type] for styles in groups.values())


def _split_assignments(
    group_ids: Sequence[str], seed: int, label: str
) -> dict[str, str]:
    ordered = _ordered(group_ids, seed, f"split:{label}")
    splits = [
        split for split, count in SPLIT_GROUP_COUNTS.items() for _ in range(count)
    ]
    return dict(zip(ordered, splits, strict=True))


def _choose(
    rows: Sequence[dict[str, Any]], seed: int, namespace: str
) -> dict[str, Any]:
    return _ordered(rows, seed, namespace, key=lambda row: row["example_id"])[0]


def _ordered(
    values: Sequence[Any], seed: int, namespace: str, *, key=lambda value: value
) -> list[Any]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(f"{seed}\0{namespace}\0{key(value)}".encode()).hexdigest(),
            key(value),
        ),
    )
