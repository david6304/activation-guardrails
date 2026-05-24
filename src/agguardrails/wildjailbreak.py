"""WildJailbreak data contract for the CC++ replication."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

DATA_TYPES = (
    "vanilla_harmful",
    "vanilla_benign",
    "adversarial_harmful",
    "adversarial_benign",
)

VANILLA_DATA_TYPES = ("vanilla_harmful", "vanilla_benign")
ADVERSARIAL_DATA_TYPES = ("adversarial_harmful", "adversarial_benign")


@dataclass(frozen=True)
class WildJailbreakExample:
    example_id: str
    row_id: str
    prompt: str
    completion: str
    label: int
    data_type: str
    source_family: str
    split: str
    tactics: list[str]
    metadata: dict[str, Any]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def load_rows_from_path(path: str | Path) -> list[dict[str, Any]]:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with source_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with source_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        raise ValueError(f"Unsupported JSON dataset shape: {source_path}")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with source_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"Unsupported dataset file extension: {source_path}")


def load_wildjailbreak_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    dataset_config = config["dataset"]
    local_path = dataset_config.get("local_path")
    if local_path:
        return load_rows_from_path(local_path)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "datasets is required to load WildJailbreak from Hugging Face. "
            "Set dataset.local_path to load a local fixture or cache file."
        ) from exc

    loader_kwargs = dict(dataset_config.get("loader_kwargs", {}))
    revision = dataset_config.get("revision")
    if revision:
        loader_kwargs["revision"] = revision
    dataset_name = dataset_config.get("name")
    legacy_subset = dataset_config.get("subset")
    split_name = dataset_config.get("split")
    if split_name is None and legacy_subset in {"train", "validation", "val", "test"}:
        split_name = legacy_subset
    elif dataset_name is None and legacy_subset:
        dataset_name = legacy_subset

    if split_name:
        loader_kwargs["split"] = split_name

    if dataset_name:
        dataset = load_dataset(dataset_config["id"], dataset_name, **loader_kwargs)
    else:
        dataset = load_dataset(dataset_config["id"], **loader_kwargs)
    return _dataset_rows(dataset, split_name=split_name)


def normalize_wildjailbreak_row(
    row: dict[str, Any],
    *,
    row_index: int,
    config: dict[str, Any],
) -> WildJailbreakExample:
    dataset_config = config.get("dataset", {})
    data_type_column = dataset_config.get("data_type_column", "data_type")
    data_type = str(row.get(data_type_column, "")).strip()
    if data_type not in DATA_TYPES:
        raise ValueError(f"Unsupported WildJailbreak data_type: {data_type!r}")

    prompt_columns = dataset_config.get("prompt_columns", {})
    vanilla_column = prompt_columns.get("vanilla", "vanilla")
    adversarial_column = prompt_columns.get("adversarial", "adversarial")
    prompt_column = (
        adversarial_column if data_type.startswith("adversarial") else vanilla_column
    )
    prompt = _clean_text(row.get(prompt_column))
    prompt_source_column = prompt_column
    prompt_fallback_from = None
    if not prompt and data_type.startswith("adversarial"):
        prompt = _clean_text(row.get(vanilla_column))
        prompt_source_column = vanilla_column
        prompt_fallback_from = prompt_column
    if not prompt:
        raise ValueError(f"Missing prompt for row {row_index} data_type={data_type}")

    row_id = _row_id(row, row_index)
    tactics_column = dataset_config.get("tactics_column", "tactics")
    completion_column = dataset_config.get("completion_column", "completion")
    source_family = "adversarial" if data_type.startswith("adversarial") else "vanilla"

    return WildJailbreakExample(
        example_id=f"{data_type}:{row_id}",
        row_id=row_id,
        prompt=prompt,
        completion=_clean_text(row.get(completion_column)),
        label=1 if data_type.endswith("_harmful") else 0,
        data_type=data_type,
        source_family=source_family,
        split="unassigned",
        tactics=_parse_tactics(row.get(tactics_column)),
        metadata={
            "upstream_dataset_id": dataset_config.get("id"),
            "upstream_dataset_revision": dataset_config.get("revision"),
            "upstream_row_index": row_index,
            "original_data_type": data_type,
            "prompt_source_column": prompt_source_column,
            "prompt_fallback_from": prompt_fallback_from,
        },
    )


def build_wildjailbreak_contract(
    rows: Iterable[dict[str, Any]],
    *,
    config: dict[str, Any],
    config_path: str | Path,
) -> tuple[list[WildJailbreakExample], dict[str, Any]]:
    normalized = [
        normalize_wildjailbreak_row(row, row_index=index, config=config)
        for index, row in enumerate(rows)
    ]
    examples = _sample_and_split(normalized, config=config)
    metadata = _build_metadata(examples, config=config, config_path=config_path)
    return examples, metadata


def write_wildjailbreak_contract(
    examples: list[WildJailbreakExample],
    metadata: dict[str, Any],
    *,
    data_path: str | Path,
    metadata_path: str | Path,
) -> None:
    data_output = Path(data_path)
    metadata_output = Path(metadata_path)
    data_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    with data_output.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(asdict(example), sort_keys=True) + "\n")

    with metadata_output.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_and_write_wildjailbreak_contract(
    *,
    config_path: str | Path,
    config: dict[str, Any] | None = None,
    rows: Iterable[dict[str, Any]] | None = None,
) -> tuple[list[WildJailbreakExample], dict[str, Any]]:
    resolved_config = load_config(config_path) if config is None else config
    source_rows = (
        list(rows) if rows is not None else load_wildjailbreak_rows(resolved_config)
    )
    examples, metadata = build_wildjailbreak_contract(
        source_rows,
        config=resolved_config,
        config_path=config_path,
    )
    outputs = resolved_config["outputs"]
    write_wildjailbreak_contract(
        examples,
        metadata,
        data_path=outputs["data_path"],
        metadata_path=outputs["metadata_path"],
    )
    return examples, metadata


def _sample_and_split(
    examples: list[WildJailbreakExample],
    *,
    config: dict[str, Any],
) -> list[WildJailbreakExample]:
    seed = int(config["sampling"]["seed"])
    per_data_type = config["sampling"].get("per_data_type", {})
    by_type: dict[str, list[WildJailbreakExample]] = {
        data_type: [] for data_type in DATA_TYPES
    }
    for example in examples:
        by_type[example.data_type].append(example)

    missing = [data_type for data_type, group in by_type.items() if not group]
    if missing:
        raise ValueError(f"Missing WildJailbreak data types: {', '.join(missing)}")

    selected: dict[str, list[WildJailbreakExample]] = {}
    for data_type, group in by_type.items():
        shuffled = sorted(group, key=lambda example: example.row_id)
        random.Random(f"{seed}:{data_type}:sample").shuffle(shuffled)
        limit = per_data_type.get(data_type)
        selected[data_type] = shuffled[: int(limit)] if limit is not None else shuffled

    split_examples: list[WildJailbreakExample] = []
    split_ratios = config["splits"]["vanilla"]
    split_names = tuple(split_ratios.keys())
    for data_type in VANILLA_DATA_TYPES:
        group = list(selected[data_type])
        random.Random(f"{seed}:{data_type}:split").shuffle(group)
        counts = _split_counts(
            len(group),
            [float(split_ratios[name]) for name in split_names],
        )
        start = 0
        for split_name, count in zip(split_names, counts, strict=True):
            for example in group[start : start + count]:
                split_examples.append(_replace_split(example, split_name, seed))
            start += count

    adversarial_split = config["splits"].get("adversarial_split", "transfer")
    for data_type in ADVERSARIAL_DATA_TYPES:
        for example in selected[data_type]:
            split_examples.append(_replace_split(example, adversarial_split, seed))

    return sorted(
        split_examples,
        key=lambda example: (example.split, example.data_type, example.row_id),
    )


def _build_metadata(
    examples: list[WildJailbreakExample],
    *,
    config: dict[str, Any],
    config_path: str | Path,
) -> dict[str, Any]:
    dataset_config = config.get("dataset", {})
    counts_by_data_type = Counter(example.data_type for example in examples)
    counts_by_split = Counter(example.split for example in examples)
    counts_by_split_and_label = Counter(
        f"{example.split}:harmful={example.label}" for example in examples
    )
    counts_by_split_and_data_type = Counter(
        f"{example.split}:{example.data_type}" for example in examples
    )
    return {
        "config_path": str(config_path),
        "seed": int(config["sampling"]["seed"]),
        "dataset": {
            "id": dataset_config.get("id"),
            "revision": dataset_config.get("revision"),
            "name": dataset_config.get("name"),
            "split": dataset_config.get("split", dataset_config.get("subset")),
            "required_data_types": list(
                dataset_config.get("required_data_types", DATA_TYPES)
            ),
        },
        "model": {
            "id": config.get("model", {}).get("id"),
            "revision": config.get("model", {}).get("revision"),
        },
        "sampling": {
            "requested_per_data_type": config["sampling"].get("per_data_type", {}),
            "selected_counts_by_data_type": dict(sorted(counts_by_data_type.items())),
        },
        "splits": {
            "counts_by_split": dict(sorted(counts_by_split.items())),
            "counts_by_split_and_label": dict(
                sorted(counts_by_split_and_label.items())
            ),
            "counts_by_split_and_data_type": dict(
                sorted(counts_by_split_and_data_type.items())
            ),
            "vanilla_ratios": config["splits"]["vanilla"],
            "adversarial_split": config["splits"].get("adversarial_split", "transfer"),
        },
    }


def _split_counts(total: int, ratios: list[float]) -> list[int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    ratio_sum = sum(ratios)
    if ratio_sum <= 0:
        raise ValueError("split ratios must sum to a positive value")
    exact = [total * ratio / ratio_sum for ratio in ratios]
    counts = [int(value) for value in exact]
    remainder = total - sum(counts)
    order = sorted(
        range(len(ratios)),
        key=lambda index: (exact[index] - counts[index], -index),
        reverse=True,
    )
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def _replace_split(
    example: WildJailbreakExample,
    split: str,
    seed: int,
) -> WildJailbreakExample:
    metadata = dict(example.metadata)
    metadata["split_seed"] = seed
    return WildJailbreakExample(
        example_id=example.example_id,
        row_id=example.row_id,
        prompt=example.prompt,
        completion=example.completion,
        label=example.label,
        data_type=example.data_type,
        source_family=example.source_family,
        split=split,
        tactics=example.tactics,
        metadata=metadata,
    )


def _row_id(row: dict[str, Any], row_index: int) -> str:
    for key in ("row_id", "id", "idx", "index"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return str(row_index)


def _dataset_rows(dataset: Any, *, split_name: str | None) -> list[dict[str, Any]]:
    if isinstance(dataset, Mapping):
        if split_name and split_name in dataset:
            dataset = dataset[split_name]
        elif len(dataset) == 1:
            dataset = next(iter(dataset.values()))
        else:
            splits = ", ".join(str(name) for name in dataset)
            raise ValueError(
                "Loaded a DatasetDict but no unambiguous split was configured. "
                f"Available splits: {splits}"
            )
    return [dict(row) for row in dataset]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_tactics(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [str(parsed)]
    return [str(value)]
