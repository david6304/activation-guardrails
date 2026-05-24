import json

import numpy as np
import pytest

from agguardrails.activation_cache import (
    SCHEMA_VERSION,
    activation_settings,
    build_activation_cache_metadata,
    build_mock_activation_cache,
    load_activation_cache_index,
    validate_activation_cache,
    validate_activation_cache_record,
    write_activation_cache,
)
from agguardrails.response_cache import make_response_cache_record


def _config():
    return {
        "model": {"id": "google/gemma-2-9b-it", "revision": "model-rev"},
        "tokenizer": {"id": "google/gemma-2-9b-it", "revision": "tokenizer-rev"},
        "sampling": {"seed": 123},
        "activation": {
            "seed": 456,
            "mode": "prompt_final",
            "source": "residual",
            "layer": 31,
            "token_position": "final_prompt_token",
            "aggregation": {"rule": "none"},
            "segment": {"enabled": False},
            "mock": {"hidden_size": 4},
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
            "upstream_dataset_id": "allenai/wildjailbreak",
            "upstream_dataset_revision": "dataset-rev",
            "upstream_row_index": index,
        },
    }


def _response_record(index=0, *, split="train", data_type="vanilla_harmful"):
    return make_response_cache_record(
        _example(index, split=split, data_type=data_type),
        response=f"response {index}",
        config=_config(),
        generated_at="2026-05-24T12:00:00Z",
    )


def test_activation_settings_defaults_to_prompt_final_residual():
    assert activation_settings({}) == {
        "mode": "prompt_final",
        "source": "residual",
        "layer": 31,
        "token_position": "final_prompt_token",
        "aggregation": {"rule": "none"},
        "segment": {"enabled": False},
    }


def test_build_mock_activation_cache_is_deterministic_and_aligned():
    response_records = [
        _response_record(0, split="train", data_type="vanilla_harmful"),
        _response_record(1, split="transfer", data_type="adversarial_benign"),
    ]

    first_records, first_activations = build_mock_activation_cache(
        response_records,
        config=_config(),
    )
    second_records, second_activations = build_mock_activation_cache(
        response_records,
        config=_config(),
    )

    validate_activation_cache(records=first_records, activations=first_activations)
    assert np.array_equal(first_activations, second_activations)
    assert [record.example_id for record in first_records] == [
        "vanilla_harmful:0",
        "adversarial_benign:1",
    ]
    assert [record.vector_index for record in first_records] == [0, 1]
    assert first_activations.shape == (2, 4)
    assert first_records[0].schema_version == SCHEMA_VERSION
    assert first_records[0].activation == {
        "mode": "prompt_final",
        "source": "residual",
        "layer": 31,
        "shape": [4],
    }
    assert first_records[0].token["position"] == "final_prompt_token"
    assert len(first_records[0].hashes["exchange_sha256"]) == 64


def test_activation_validation_rejects_shape_mismatch():
    response_records, activations = build_mock_activation_cache(
        [_response_record()],
        config=_config(),
    )
    bad_record = response_records[0].__dict__ | {
        "activation": response_records[0].activation | {"shape": []}
    }

    with pytest.raises(ValueError, match="shape"):
        validate_activation_cache_record(bad_record)
    with pytest.raises(ValueError, match="shape"):
        validate_activation_cache(
            records=response_records,
            activations=activations[:, :2],
        )


def test_activation_metadata_records_provenance_and_counts(tmp_path):
    dataset_path = tmp_path / "wildjailbreak.jsonl"
    response_path = tmp_path / "responses.jsonl"
    dataset_path.write_text('{"example_id": "one"}\n', encoding="utf-8")
    response_path.write_text('{"example_id": "one"}\n', encoding="utf-8")
    records, activations = build_mock_activation_cache(
        [
            _response_record(0, split="train", data_type="vanilla_harmful"),
            _response_record(1, split="transfer", data_type="adversarial_benign"),
        ],
        config=_config(),
    )

    metadata = build_activation_cache_metadata(
        records,
        activations,
        config=_config(),
        config_path="configs/ccpp/test.yaml",
        dataset_artifact_path=dataset_path,
        response_cache_path=response_path,
        activation_cache_path=tmp_path / "activations.npz",
        index_path=tmp_path / "activations.index.jsonl",
        created_at="2026-05-24T12:00:00Z",
        git_commit="abc123",
    )

    assert metadata["schema_version"] == SCHEMA_VERSION
    assert metadata["config_path"] == "configs/ccpp/test.yaml"
    assert metadata["git_commit"] == "abc123"
    assert metadata["dataset_artifact"]["path"] == str(dataset_path)
    assert metadata["response_cache_artifact"]["path"] == str(response_path)
    assert len(metadata["dataset_artifact"]["sha256"]) == 64
    assert len(metadata["response_cache_artifact"]["sha256"]) == 64
    assert metadata["model"]["revision"] == "model-rev"
    assert metadata["tokenizer"]["revision"] == "tokenizer-rev"
    assert metadata["activation"]["source"] == "residual"
    assert metadata["activation"]["segment"] == {"enabled": False}
    assert metadata["shape"] == [2, 4]
    assert metadata["dtype"] == "float32"
    assert metadata["counts"]["by_split"] == {"train": 1, "transfer": 1}


def test_write_activation_cache_writes_npz_index_and_metadata(tmp_path):
    records, activations = build_mock_activation_cache(
        [_response_record()],
        config=_config(),
    )
    metadata = {"schema_version": SCHEMA_VERSION, "records": 1}
    activation_path = tmp_path / "activations.npz"
    index_path = tmp_path / "activations.index.jsonl"
    metadata_path = tmp_path / "activations.metadata.json"

    write_activation_cache(
        records,
        activations,
        activation_cache_path=activation_path,
        index_path=index_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )

    with np.load(activation_path) as cache:
        assert cache["activations"].shape == (1, 4)
        assert cache["labels"].tolist() == [1]
        assert cache["example_ids"].tolist() == ["vanilla_harmful:0"]
    loaded_index = load_activation_cache_index(index_path)
    written_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert loaded_index[0].example_id == "vanilla_harmful:0"
    assert written_metadata == metadata
