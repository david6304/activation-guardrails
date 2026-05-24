import csv
import json

import joblib
import numpy as np
import pytest

from agguardrails.activation_cache import (
    build_mock_activation_cache,
    write_activation_cache,
)
from agguardrails.dense_probe import (
    SCHEMA_VERSION,
    load_activation_cache_arrays,
    run_dense_probe,
    split_activation_records,
)
from agguardrails.response_cache import make_response_cache_record


def _config():
    return {
        "model": {"id": "google/gemma-2-9b-it", "revision": "model-rev"},
        "tokenizer": {"id": "google/gemma-2-9b-it", "revision": "tokenizer-rev"},
        "dataset": {
            "id": "allenai/wildjailbreak",
            "revision": "dataset-rev",
            "name": "train",
            "split": "train",
        },
        "sampling": {"seed": 123},
        "activation": {
            "seed": 456,
            "mode": "prompt_final",
            "source": "residual",
            "layer": 31,
            "token_position": "final_prompt_token",
            "aggregation": {"rule": "none"},
            "segment": {"enabled": False},
            "mock": {"hidden_size": 2},
        },
        "dense_probe": {
            "seed": 789,
            "threshold": {"max_fpr": 0.01},
            "logistic_regression": {"max_iter": 200, "solver": "liblinear"},
        },
    }


def _example(index, *, split, data_type):
    label = 1 if data_type.endswith("_harmful") else 0
    source_family = "adversarial" if data_type.startswith("adversarial") else "vanilla"
    return {
        "example_id": f"{split}:{data_type}:{index}",
        "row_id": str(index),
        "prompt": f"prompt {split} {data_type} {index}",
        "completion": "",
        "label": label,
        "data_type": data_type,
        "source_family": source_family,
        "split": split,
        "tactics": ["encoding"] if source_family == "adversarial" else [],
        "metadata": {
            "upstream_dataset_id": "allenai/wildjailbreak",
            "upstream_dataset_revision": "dataset-rev",
            "upstream_row_index": index,
        },
    }


def _response_record(index, *, split, data_type):
    return make_response_cache_record(
        _example(index, split=split, data_type=data_type),
        response=f"response {split} {data_type} {index}",
        config=_config(),
        generated_at="2026-05-24T12:00:00Z",
    )


def _records_and_activations():
    response_records = []
    for split, count in {"train": 6, "val": 3, "test": 3}.items():
        for index in range(count):
            response_records.append(
                _response_record(index, split=split, data_type="vanilla_harmful")
            )
            response_records.append(
                _response_record(index, split=split, data_type="vanilla_benign")
            )
    for index in range(4):
        response_records.append(
            _response_record(index, split="transfer", data_type="adversarial_harmful")
        )
        response_records.append(
            _response_record(index, split="transfer", data_type="adversarial_benign")
        )

    records, _ = build_mock_activation_cache(response_records, config=_config())
    activations = []
    for record in records:
        sign = 1.0 if record.label else -1.0
        activations.append([sign * 2.0, (record.vector_index % 3) * 0.1])
    return records, np.asarray(activations, dtype=np.float32)


def test_dense_probe_trains_on_vanilla_and_writes_metadata_rich_outputs(tmp_path):
    records, activations = _records_and_activations()
    data_path = tmp_path / "wildjailbreak.jsonl"
    response_path = tmp_path / "responses.jsonl"
    activation_path = tmp_path / "activations.npz"
    index_path = tmp_path / "activations.index.jsonl"
    activation_metadata_path = tmp_path / "activations.metadata.json"
    model_path = tmp_path / "dense.joblib"
    scores_path = tmp_path / "scores.jsonl"
    metrics_path = tmp_path / "metrics.json"
    table_path = tmp_path / "table.csv"
    data_path.write_text('{"example_id": "one"}\n', encoding="utf-8")
    response_path.write_text('{"example_id": "one"}\n', encoding="utf-8")
    write_activation_cache(
        records,
        activations,
        activation_cache_path=activation_path,
        index_path=index_path,
        metadata_path=activation_metadata_path,
        metadata={"schema_version": "agguardrails.activation_cache.v1"},
    )

    metadata = run_dense_probe(
        activations=activations,
        records=records,
        config=_config(),
        config_path="configs/ccpp/test.yaml",
        dataset_artifact_path=data_path,
        response_cache_path=response_path,
        activation_cache_path=activation_path,
        activation_index_path=index_path,
        model_path=model_path,
        scores_path=scores_path,
        metrics_path=metrics_path,
        table_path=table_path,
        created_at="2026-05-24T12:00:00Z",
        git_commit="abc123",
    )

    assert metadata["schema_version"] == SCHEMA_VERSION
    assert metadata["git_commit"] == "abc123"
    assert metadata["dataset_artifact"]["path"] == str(data_path)
    assert metadata["response_cache_artifact"]["path"] == str(response_path)
    assert metadata["activation_cache_artifact"]["path"] == str(activation_path)
    assert len(metadata["activation_index_artifact"]["sha256"]) == 64
    assert metadata["model"]["revision"] == "model-rev"
    assert metadata["activation"]["mode"] == "prompt_final"
    assert metadata["activation"]["source"] == "residual"
    assert metadata["training"]["counts"]["by_data_type"] == {
        "vanilla_benign": 6,
        "vanilla_harmful": 6,
    }
    assert metadata["threshold_rule"]["max_fpr"] == 0.01
    assert metadata["metrics"]["validation"]["fpr"] <= 0.01
    assert (
        metadata["metrics"]["vanilla_test"]["threshold"]
        == metadata["threshold_rule"]["frozen_threshold"]
    )
    assert (
        metadata["metrics"]["adversarial_transfer"]["threshold"]
        == metadata["threshold_rule"]["frozen_threshold"]
    )
    assert metadata["package_versions"]["scikit-learn"]

    model_bundle = joblib.load(model_path)
    assert model_bundle["classifier"].classes_.tolist() == [0, 1]
    written_metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    score_rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
    table_rows = list(csv.DictReader(table_path.open(encoding="utf-8")))

    assert written_metadata == metadata
    assert len(score_rows) == len(records)
    assert all("exchange_sha256" in row["hashes"] for row in score_rows)
    assert [row["split"] for row in table_rows] == [
        "vanilla_val",
        "vanilla_test",
        "adversarial_transfer",
    ]


def test_load_activation_cache_arrays_rejects_misaligned_npz_labels(tmp_path):
    records, activations = _records_and_activations()
    activation_path = tmp_path / "activations.npz"
    index_path = tmp_path / "activations.index.jsonl"
    np.savez_compressed(
        activation_path,
        activations=activations,
        labels=np.zeros(len(records), dtype=np.int64),
        example_ids=np.asarray([record.example_id for record in records]),
    )
    with index_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.__dict__, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="labels"):
        load_activation_cache_arrays(activation_path, index_path)


def test_split_activation_records_rejects_one_class_eval_split():
    records, _ = _records_and_activations()
    records = [record for record in records if record.data_type != "vanilla_benign"]

    with pytest.raises(ValueError, match="must contain both labels"):
        split_activation_records(records)
