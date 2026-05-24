"""Activation-cache contract for CC++ dense probe features."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from agguardrails.response_cache import (
    ResponseCacheRecord,
    current_git_commit,
    sha256_file,
    sha256_text,
    validate_response_cache_record,
)

SCHEMA_VERSION = "agguardrails.activation_cache.v1"


@dataclass(frozen=True)
class ActivationCacheRecord:
    schema_version: str
    example_id: str
    split: str
    label: int
    data_type: str
    source_family: str
    tactics: list[str]
    vector_index: int
    activation: dict[str, Any]
    token: dict[str, Any]
    response_cache: dict[str, Any]
    hashes: dict[str, str]
    metadata: dict[str, Any]


def activation_settings(config: dict[str, Any]) -> dict[str, Any]:
    activation_config = dict(config.get("activation", {}))
    return {
        "mode": activation_config.get("mode", "prompt_final"),
        "source": activation_config.get("source", "residual"),
        "layer": activation_config.get("layer", 31),
        "token_position": activation_config.get(
            "token_position",
            "final_prompt_token",
        ),
        "aggregation": dict(activation_config.get("aggregation", {"rule": "none"})),
        "segment": dict(activation_config.get("segment", {"enabled": False})),
    }


def mock_hidden_size(config: dict[str, Any]) -> int:
    activation_config = config.get("activation", {})
    mock_config = activation_config.get("mock", {})
    hidden_size = int(mock_config.get("hidden_size", 8))
    if hidden_size <= 0:
        raise ValueError("activation.mock.hidden_size must be positive")
    return hidden_size


def build_mock_activation_cache(
    response_records: Iterable[ResponseCacheRecord],
    *,
    config: dict[str, Any],
) -> tuple[list[ActivationCacheRecord], np.ndarray]:
    records = list(response_records)
    settings = activation_settings(config)
    hidden_size = mock_hidden_size(config)
    rows = []
    vectors = []
    for vector_index, response_record in enumerate(records):
        validate_response_cache_record(response_record)
        vector = deterministic_mock_activation_vector(
            response_record,
            config=config,
            hidden_size=hidden_size,
        )
        vectors.append(vector)
        rows.append(
            make_activation_cache_record(
                response_record,
                vector_index=vector_index,
                activation_shape=[hidden_size],
                settings=settings,
            )
        )
    if vectors:
        activations = np.stack(vectors).astype(np.float32)
    else:
        activations = np.empty((0, hidden_size), dtype=np.float32)
    return rows, activations


def deterministic_mock_activation_vector(
    response_record: ResponseCacheRecord,
    *,
    config: dict[str, Any],
    hidden_size: int,
) -> np.ndarray:
    settings = activation_settings(config)
    seed = config.get("activation", {}).get(
        "seed",
        config.get("sampling", {}).get("seed"),
    )
    key = ":".join(
        [
            str(seed),
            response_record.example_id,
            str(settings["mode"]),
            str(settings["source"]),
            str(settings["layer"]),
            str(settings["token_position"]),
            response_record.hashes["prompt_sha256"],
            response_record.hashes["response_sha256"],
        ]
    )
    values = []
    counter = 0
    while len(values) < hidden_size:
        digest = bytes.fromhex(sha256_text(f"{key}:{counter}"))
        for index in range(0, len(digest), 4):
            integer = int.from_bytes(digest[index : index + 4], "big", signed=False)
            values.append((integer / 2**32) * 2.0 - 1.0)
            if len(values) == hidden_size:
                break
        counter += 1
    return np.asarray(values, dtype=np.float32)


def make_activation_cache_record(
    response_record: ResponseCacheRecord,
    *,
    vector_index: int,
    activation_shape: list[int],
    settings: dict[str, Any],
) -> ActivationCacheRecord:
    validate_response_cache_record(response_record)
    if vector_index < 0:
        raise ValueError("vector_index must be non-negative")
    token_position = str(settings["token_position"])
    exchange_text = response_record.prompt + response_record.response
    return ActivationCacheRecord(
        schema_version=SCHEMA_VERSION,
        example_id=response_record.example_id,
        split=response_record.split,
        label=response_record.label,
        data_type=response_record.data_type,
        source_family=response_record.source_family,
        tactics=list(response_record.tactics),
        vector_index=vector_index,
        activation={
            "mode": settings["mode"],
            "source": settings["source"],
            "layer": settings["layer"],
            "shape": activation_shape,
        },
        token={
            "position": token_position,
            "index": None,
            "token_count": None,
        },
        response_cache={
            "schema_version": response_record.schema_version,
            "generated_at": response_record.generated_at,
        },
        hashes={
            "prompt_sha256": response_record.hashes["prompt_sha256"],
            "response_sha256": response_record.hashes["response_sha256"],
            "exchange_sha256": sha256_text(exchange_text),
        },
        metadata={
            "response_record_metadata": dict(response_record.metadata),
        },
    )


def validate_activation_cache_record(
    record: ActivationCacheRecord | dict[str, Any],
) -> None:
    row = asdict(record) if isinstance(record, ActivationCacheRecord) else record
    required = {
        "schema_version",
        "example_id",
        "split",
        "label",
        "data_type",
        "source_family",
        "tactics",
        "vector_index",
        "activation",
        "token",
        "response_cache",
        "hashes",
        "metadata",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"Activation cache record missing fields: {missing}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported activation cache schema: {row['schema_version']}"
        )
    if row["label"] not in {0, 1}:
        raise ValueError("label must be 0 or 1")
    if not isinstance(row["vector_index"], int) or row["vector_index"] < 0:
        raise ValueError("vector_index must be a non-negative integer")
    for key in ("example_id", "split", "data_type", "source_family"):
        if not isinstance(row[key], str) or not row[key]:
            raise ValueError(f"{key} must be a non-empty string")
    activation = row["activation"]
    if activation.get("mode") not in {"prompt_final", "exchange_stream"}:
        raise ValueError("activation.mode must be prompt_final or exchange_stream")
    if activation.get("source") not in {"residual", "attention", "mlp"}:
        raise ValueError("activation.source must be residual, attention, or mlp")
    if not isinstance(activation.get("shape"), list) or not activation["shape"]:
        raise ValueError("activation.shape must be a non-empty list")


def build_activation_cache_metadata(
    records: Iterable[ActivationCacheRecord],
    activations: np.ndarray,
    *,
    config: dict[str, Any],
    config_path: str | Path,
    dataset_artifact_path: str | Path,
    response_cache_path: str | Path,
    activation_cache_path: str | Path,
    index_path: str | Path,
    created_at: str,
    git_commit: str | None = None,
) -> dict[str, Any]:
    record_list = list(records)
    validate_activation_cache(records=record_list, activations=activations)
    dataset_path = Path(dataset_artifact_path)
    response_path = Path(response_cache_path)
    model = dict(config.get("model", {}))
    tokenizer = dict(config.get("tokenizer") or model)
    settings = activation_settings(config)
    return {
        "schema_version": SCHEMA_VERSION,
        "config_path": str(config_path),
        "git_commit": git_commit if git_commit is not None else current_git_commit(),
        "created_at": created_at,
        "seed": config.get("activation", {}).get(
            "seed",
            config.get("sampling", {}).get("seed"),
        ),
        "dataset_artifact": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "response_cache_artifact": {
            "path": str(response_path),
            "sha256": sha256_file(response_path),
        },
        "activation_cache_path": str(activation_cache_path),
        "activation_index_path": str(index_path),
        "model": {
            "id": model.get("id"),
            "revision": model.get("revision"),
        },
        "tokenizer": {
            "id": tokenizer.get("id"),
            "revision": tokenizer.get("revision"),
        },
        "activation": settings,
        "shape": list(activations.shape),
        "dtype": str(activations.dtype),
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


def write_activation_cache(
    records: Iterable[ActivationCacheRecord],
    activations: np.ndarray,
    *,
    activation_cache_path: str | Path,
    index_path: str | Path,
    metadata_path: str | Path,
    metadata: dict[str, Any],
) -> None:
    record_list = list(records)
    validate_activation_cache(records=record_list, activations=activations)
    activation_output = Path(activation_cache_path)
    index_output = Path(index_path)
    metadata_output = Path(metadata_path)
    activation_output.parent.mkdir(parents=True, exist_ok=True)
    index_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        activation_output,
        activations=activations.astype(np.float32, copy=False),
        labels=np.asarray([record.label for record in record_list], dtype=np.int64),
        example_ids=np.asarray([record.example_id for record in record_list]),
    )
    with index_output.open("w", encoding="utf-8") as handle:
        for record in record_list:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
    with metadata_output.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_activation_cache_index(path: str | Path) -> list[ActivationCacheRecord]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                record = ActivationCacheRecord(**row)
            except TypeError as exc:
                raise ValueError(
                    f"Activation cache row {line_number} has invalid fields"
                ) from exc
            validate_activation_cache_record(record)
            records.append(record)
    return records


def validate_activation_cache(
    *,
    records: list[ActivationCacheRecord],
    activations: np.ndarray,
) -> None:
    if activations.ndim != 2:
        raise ValueError("activations must be a 2D array")
    if activations.shape[0] != len(records):
        raise ValueError("activation row count must match index records")
    seen: set[str] = set()
    for expected_index, record in enumerate(records):
        validate_activation_cache_record(record)
        if record.example_id in seen:
            raise ValueError(f"Duplicate activation example_id: {record.example_id}")
        seen.add(record.example_id)
        if record.vector_index != expected_index:
            raise ValueError("vector_index must match activation row order")
        if record.activation["shape"] != list(activations[expected_index].shape):
            raise ValueError("activation shape does not match index record")
