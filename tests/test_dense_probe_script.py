import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from agguardrails.activation_cache import (
    build_mock_activation_cache,
    write_activation_cache,
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
    activations = np.asarray(
        [
            [2.0 if record.label else -2.0, (record.vector_index % 3) * 0.1]
            for record in records
        ],
        dtype=np.float32,
    )
    return records, activations


def _write_config(
    tmp_path,
    *,
    data_path,
    response_path,
    activation_path,
    index_path,
    model_path,
    scores_path,
    metrics_path,
    table_path,
):
    config_path = tmp_path / "config.yaml"
    config = _config() | {
        "outputs": {
            "data_path": str(data_path),
            "response_cache_path": str(response_path),
            "activation_cache_path": str(activation_path),
            "activation_index_path": str(index_path),
            "dense_probe_model_path": str(model_path),
            "dense_probe_scores_path": str(scores_path),
            "dense_probe_metrics_path": str(metrics_path),
            "dense_probe_table_path": str(table_path),
        },
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return config_path


def _run_script(*args):
    script_path = Path("scripts/ccpp/train_dense_probe.py")
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_train_dense_probe_script_writes_outputs(tmp_path):
    data_path = tmp_path / "wildjailbreak.jsonl"
    response_path = tmp_path / "responses.jsonl"
    activation_path = tmp_path / "activations.npz"
    index_path = tmp_path / "activations.index.jsonl"
    activation_metadata_path = tmp_path / "activations.metadata.json"
    model_path = tmp_path / "dense.joblib"
    scores_path = tmp_path / "scores.jsonl"
    metrics_path = tmp_path / "metrics.json"
    table_path = tmp_path / "table.csv"
    records, activations = _records_and_activations()
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
    config_path = _write_config(
        tmp_path,
        data_path=data_path,
        response_path=response_path,
        activation_path=activation_path,
        index_path=index_path,
        model_path=model_path,
        scores_path=scores_path,
        metrics_path=metrics_path,
        table_path=table_path,
    )

    result = _run_script("--config", str(config_path))

    assert "Trained dense_prompt_final_logistic_regression" in result.stdout
    assert "Selected threshold" in result.stdout
    assert model_path.exists()
    assert scores_path.exists()
    assert table_path.exists()
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metadata["config_path"] == str(config_path)
    assert metadata["threshold_rule"]["selection_split"] == "val"
    assert metadata["metrics"]["vanilla_test"]["positives"] == 3
    assert metadata["metrics"]["adversarial_transfer"]["negatives"] == 4


def test_train_dense_probe_script_dry_run_does_not_write(tmp_path):
    data_path = tmp_path / "wildjailbreak.jsonl"
    response_path = tmp_path / "responses.jsonl"
    activation_path = tmp_path / "activations.npz"
    index_path = tmp_path / "activations.index.jsonl"
    activation_metadata_path = tmp_path / "activations.metadata.json"
    model_path = tmp_path / "dense.joblib"
    scores_path = tmp_path / "scores.jsonl"
    metrics_path = tmp_path / "metrics.json"
    table_path = tmp_path / "table.csv"
    records, activations = _records_and_activations()
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
    config_path = _write_config(
        tmp_path,
        data_path=data_path,
        response_path=response_path,
        activation_path=activation_path,
        index_path=index_path,
        model_path=model_path,
        scores_path=scores_path,
        metrics_path=metrics_path,
        table_path=table_path,
    )

    result = _run_script("--config", str(config_path), "--dry-run")

    assert "Loaded activation array [32, 2]" in result.stdout
    assert "transfer\t8" in result.stdout
    assert not model_path.exists()
    assert not scores_path.exists()
    assert not metrics_path.exists()
    assert not table_path.exists()
