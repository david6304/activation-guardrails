"""Response-cache contract for CC++ prompt/response exchanges."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "agguardrails.response_cache.v1"


@dataclass(frozen=True)
class ResponseCacheRecord:
    schema_version: str
    example_id: str
    split: str
    label: int
    data_type: str
    source_family: str
    tactics: list[str]
    prompt: str
    messages: list[dict[str, str]]
    response: str
    model: dict[str, Any]
    tokenizer: dict[str, Any]
    generation: dict[str, Any]
    hashes: dict[str, str]
    generated_at: str
    metadata: dict[str, Any]


def load_normalized_examples(path: str | Path) -> list[dict[str, Any]]:
    examples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            _validate_normalized_example(example, line_number=line_number)
            examples.append(example)
    if not examples:
        raise ValueError(f"No examples found in {path}")
    return examples


def select_examples_for_response_cache(
    examples: Iterable[dict[str, Any]],
    *,
    seed: int,
    limit: int | None = None,
    splits: Iterable[str] | None = None,
    already_cached_example_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Select a deterministic debug subset and skip examples already cached."""
    selected_splits = set(splits) if splits is not None else None
    cached_ids = set(already_cached_example_ids or [])
    candidates = [
        example
        for example in examples
        if (selected_splits is None or example["split"] in selected_splits)
        and example["example_id"] not in cached_ids
    ]
    candidates = sorted(candidates, key=lambda row: row["example_id"])
    random.Random(f"{seed}:response-cache:subset").shuffle(candidates)
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        candidates = candidates[:limit]
    return sorted(
        candidates,
        key=lambda row: (row["split"], row["data_type"], row["example_id"]),
    )


def load_cached_example_ids(path: str | Path) -> set[str]:
    cache_path = Path(path)
    if not cache_path.exists():
        return set()
    example_ids = set()
    with cache_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            example_id = row.get("example_id")
            if not isinstance(example_id, str) or not example_id:
                raise ValueError(
                    f"Cached response row {line_number} is missing example_id"
                )
            example_ids.add(example_id)
    return example_ids


def make_response_cache_record(
    example: dict[str, Any],
    *,
    response: str,
    config: dict[str, Any],
    generated_at: str,
    messages: list[dict[str, str]] | None = None,
) -> ResponseCacheRecord:
    _validate_normalized_example(example, line_number=None)
    if not isinstance(response, str):
        raise TypeError("response must be a string")

    model = dict(config.get("model", {}))
    tokenizer = dict(config.get("tokenizer") or model)
    generation_config = config.get("generation", {})
    generation = {
        "params": dict(generation_config.get("params", {})),
        "seed": generation_config.get("seed", config.get("sampling", {}).get("seed")),
    }
    prompt = str(example["prompt"])
    exchange_messages = messages or [{"role": "user", "content": prompt}]
    return ResponseCacheRecord(
        schema_version=SCHEMA_VERSION,
        example_id=str(example["example_id"]),
        split=str(example["split"]),
        label=int(example["label"]),
        data_type=str(example["data_type"]),
        source_family=str(example["source_family"]),
        tactics=[str(item) for item in example.get("tactics", [])],
        prompt=prompt,
        messages=exchange_messages,
        response=response,
        model={
            "id": model.get("id"),
            "revision": model.get("revision"),
        },
        tokenizer={
            "id": tokenizer.get("id"),
            "revision": tokenizer.get("revision"),
        },
        generation=generation,
        hashes={
            "prompt_sha256": sha256_text(prompt),
            "response_sha256": sha256_text(response),
        },
        generated_at=generated_at,
        metadata={
            "source_example_metadata": dict(example.get("metadata", {})),
        },
    )


def validate_response_cache_record(
    record: ResponseCacheRecord | dict[str, Any],
) -> None:
    row = asdict(record) if isinstance(record, ResponseCacheRecord) else record
    required = {
        "schema_version",
        "example_id",
        "split",
        "label",
        "data_type",
        "source_family",
        "tactics",
        "prompt",
        "messages",
        "response",
        "model",
        "tokenizer",
        "generation",
        "hashes",
        "generated_at",
        "metadata",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Response cache record missing fields: {missing}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported response cache schema: {row['schema_version']}")
    if row["label"] not in {0, 1}:
        raise ValueError("label must be 0 or 1")
    for key in ("example_id", "split", "data_type", "prompt", "generated_at"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f"{key} must be a non-empty string")
    if not isinstance(row["messages"], list) or not row["messages"]:
        raise ValueError("messages must be a non-empty list")
    for message in row["messages"]:
        if set(message) != {"role", "content"}:
            raise ValueError("messages must contain role/content pairs")
    if row["hashes"].get("prompt_sha256") != sha256_text(row["prompt"]):
        raise ValueError("prompt_sha256 does not match prompt")
    if row["hashes"].get("response_sha256") != sha256_text(row["response"]):
        raise ValueError("response_sha256 does not match response")


def build_response_cache_metadata(
    records: Iterable[ResponseCacheRecord],
    *,
    config: dict[str, Any],
    config_path: str | Path,
    dataset_artifact_path: str | Path,
    response_cache_path: str | Path,
    created_at: str,
    git_commit: str | None = None,
) -> dict[str, Any]:
    record_list = list(records)
    for record in record_list:
        validate_response_cache_record(record)
    dataset_path = Path(dataset_artifact_path)
    generation_config = config.get("generation", {})
    model = dict(config.get("model", {}))
    tokenizer = dict(config.get("tokenizer") or model)
    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path),
        "git_commit": git_commit if git_commit is not None else current_git_commit(),
        "created_at": created_at,
        "seed": generation_config.get("seed", config.get("sampling", {}).get("seed")),
        "dataset_artifact": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "response_cache_path": str(response_cache_path),
        "model": {
            "id": model.get("id"),
            "revision": model.get("revision"),
        },
        "tokenizer": {
            "id": tokenizer.get("id"),
            "revision": tokenizer.get("revision"),
        },
        "generation_params": dict(generation_config.get("params", {})),
        "counts": {
            "records": len(record_list),
            "by_split": dict(
                sorted(Counter(record.split for record in record_list).items())
            ),
            "by_data_type": dict(
                sorted(Counter(record.data_type for record in record_list).items())
            ),
        },
    }


def write_response_cache(
    records: Iterable[ResponseCacheRecord],
    *,
    cache_path: str | Path,
    metadata_path: str | Path,
    metadata: dict[str, Any],
) -> None:
    cache_output = Path(cache_path)
    metadata_output = Path(metadata_path)
    cache_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    with cache_output.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_response_cache_record(record)
            if record.example_id in seen:
                raise ValueError(f"Duplicate cached example_id: {record.example_id}")
            seen.add(record.example_id)
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    with metadata_output.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _validate_normalized_example(
    example: dict[str, Any],
    *,
    line_number: int | None,
) -> None:
    required = {
        "example_id",
        "row_id",
        "prompt",
        "completion",
        "label",
        "data_type",
        "source_family",
        "split",
        "tactics",
        "metadata",
    }
    missing = sorted(required - set(example))
    location = f" on line {line_number}" if line_number is not None else ""
    if missing:
        raise ValueError(f"Normalized example{location} missing fields: {missing}")
    if example["label"] not in {0, 1}:
        raise ValueError(f"Normalized example{location} has non-binary label")
    if not str(example["prompt"]).strip():
        raise ValueError(f"Normalized example{location} has an empty prompt")
