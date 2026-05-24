import json

import pytest

from agguardrails.response_cache import (
    SCHEMA_VERSION,
    build_response_cache_metadata,
    load_cached_example_ids,
    make_response_cache_record,
    select_examples_for_response_cache,
    validate_response_cache_record,
    write_response_cache,
)


def _config():
    return {
        "model": {"id": "google/gemma-2-9b-it", "revision": "model-rev"},
        "tokenizer": {"id": "google/gemma-2-9b-it", "revision": "tokenizer-rev"},
        "sampling": {"seed": 123},
        "generation": {
            "seed": 456,
            "params": {"max_new_tokens": 32, "do_sample": False},
        },
    }


def _example(index=0, *, split="train", data_type="vanilla_harmful"):
    return {
        "example_id": f"{data_type}:{index}",
        "row_id": str(index),
        "prompt": f"prompt {index}",
        "completion": "",
        "label": 1 if data_type.endswith("_harmful") else 0,
        "data_type": data_type,
        "source_family": (
            "adversarial" if data_type.startswith("adversarial") else "vanilla"
        ),
        "split": split,
        "tactics": ["roleplay"] if data_type.startswith("adversarial") else [],
        "metadata": {
            "upstream_dataset_revision": "dataset-rev",
            "upstream_row_index": index,
        },
    }


def _examples():
    return [
        _example(0, split="train", data_type="vanilla_harmful"),
        _example(1, split="train", data_type="vanilla_benign"),
        _example(2, split="val", data_type="vanilla_harmful"),
        _example(3, split="transfer", data_type="adversarial_benign"),
        _example(4, split="transfer", data_type="adversarial_harmful"),
    ]


def test_response_cache_record_schema_includes_exchange_and_generation_fields():
    record = make_response_cache_record(
        _example(),
        response="cached answer",
        config=_config(),
        generated_at="2026-05-24T12:00:00Z",
    )

    validate_response_cache_record(record)

    assert record.schema_version == SCHEMA_VERSION
    assert record.example_id == "vanilla_harmful:0"
    assert record.messages == [{"role": "user", "content": "prompt 0"}]
    assert record.response == "cached answer"
    assert record.model == {"id": "google/gemma-2-9b-it", "revision": "model-rev"}
    assert record.tokenizer == {
        "id": "google/gemma-2-9b-it",
        "revision": "tokenizer-rev",
    }
    assert record.generation == {
        "params": {"max_new_tokens": 32, "do_sample": False},
        "seed": 456,
    }
    assert len(record.hashes["prompt_sha256"]) == 64
    assert len(record.hashes["response_sha256"]) == 64
    assert record.metadata["source_example_metadata"]["upstream_row_index"] == 0


def test_response_cache_validation_rejects_hash_mismatch():
    record = make_response_cache_record(
        _example(),
        response="cached answer",
        config=_config(),
        generated_at="2026-05-24T12:00:00Z",
    )
    row = record.__dict__ | {"hashes": record.hashes | {"response_sha256": "bad"}}

    with pytest.raises(ValueError, match="response_sha256"):
        validate_response_cache_record(row)


def test_response_cache_metadata_records_provenance_and_counts(tmp_path):
    dataset_path = tmp_path / "wildjailbreak.jsonl"
    dataset_path.write_text('{"example_id": "one"}\n', encoding="utf-8")
    records = [
        make_response_cache_record(
            _example(0, split="train", data_type="vanilla_harmful"),
            response="answer 0",
            config=_config(),
            generated_at="2026-05-24T12:00:00Z",
        ),
        make_response_cache_record(
            _example(1, split="transfer", data_type="adversarial_benign"),
            response="answer 1",
            config=_config(),
            generated_at="2026-05-24T12:00:01Z",
        ),
    ]

    metadata = build_response_cache_metadata(
        records,
        config=_config(),
        config_path="configs/ccpp/test.yaml",
        dataset_artifact_path=dataset_path,
        response_cache_path=tmp_path / "responses.jsonl",
        created_at="2026-05-24T12:00:02Z",
        git_commit="abc123",
    )

    assert metadata["schema_version"] == SCHEMA_VERSION
    assert metadata["config_path"] == "configs/ccpp/test.yaml"
    assert metadata["git_commit"] == "abc123"
    assert metadata["seed"] == 456
    assert metadata["dataset_artifact"]["path"] == str(dataset_path)
    assert len(metadata["dataset_artifact"]["sha256"]) == 64
    assert metadata["model"]["revision"] == "model-rev"
    assert metadata["tokenizer"]["revision"] == "tokenizer-rev"
    assert metadata["generation_params"] == {"max_new_tokens": 32, "do_sample": False}
    assert metadata["counts"]["records"] == 2
    assert metadata["counts"]["by_split"] == {"train": 1, "transfer": 1}
    assert metadata["counts"]["by_data_type"] == {
        "adversarial_benign": 1,
        "vanilla_harmful": 1,
    }


def test_write_response_cache_writes_jsonl_and_metadata(tmp_path):
    records = [
        make_response_cache_record(
            _example(),
            response="cached answer",
            config=_config(),
            generated_at="2026-05-24T12:00:00Z",
        )
    ]
    metadata = {"schema_version": SCHEMA_VERSION, "records": 1}
    cache_path = tmp_path / "responses.jsonl"
    metadata_path = tmp_path / "responses.metadata.json"

    write_response_cache(
        records,
        cache_path=cache_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )

    rows = [json.loads(line) for line in cache_path.read_text().splitlines()]
    written_metadata = json.loads(metadata_path.read_text())

    assert rows[0]["example_id"] == "vanilla_harmful:0"
    assert rows[0]["schema_version"] == SCHEMA_VERSION
    assert written_metadata == metadata


def test_deterministic_subset_selection_filters_splits_and_limit():
    first = select_examples_for_response_cache(
        _examples(),
        seed=123,
        limit=2,
        splits=["transfer", "val"],
    )
    second = select_examples_for_response_cache(
        _examples(),
        seed=123,
        limit=2,
        splits=["transfer", "val"],
    )

    assert [row["example_id"] for row in first] == [
        row["example_id"] for row in second
    ]
    assert len(first) == 2
    assert {row["split"] for row in first} <= {"transfer", "val"}


def test_resume_safe_selection_skips_already_cached_example_ids(tmp_path):
    record = make_response_cache_record(
        _example(0, split="train", data_type="vanilla_harmful"),
        response="cached answer",
        config=_config(),
        generated_at="2026-05-24T12:00:00Z",
    )
    cache_path = tmp_path / "responses.jsonl"
    write_response_cache(
        [record],
        cache_path=cache_path,
        metadata_path=tmp_path / "metadata.json",
        metadata={"schema_version": SCHEMA_VERSION},
    )

    selected = select_examples_for_response_cache(
        _examples(),
        seed=123,
        already_cached_example_ids=load_cached_example_ids(cache_path),
    )

    assert "vanilla_harmful:0" not in {row["example_id"] for row in selected}
