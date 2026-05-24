import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from agguardrails.response_cache import make_response_cache_record, write_response_cache


def _config():
    return {
        "model": {"id": "google/gemma-2-9b-it", "revision": "model-rev"},
        "tokenizer": {"id": "google/gemma-2-9b-it", "revision": "tokenizer-rev"},
        "sampling": {"seed": 123},
        "generation": {
            "seed": 456,
            "params": {"max_new_tokens": 16, "do_sample": False},
        },
        "activation": {
            "seed": 789,
            "mode": "prompt_final",
            "source": "residual",
            "layer": 31,
            "token_position": "final_prompt_token",
            "aggregation": {"rule": "none"},
            "segment": {"enabled": False},
            "mock": {"hidden_size": 4},
        },
    }


def _example(index, *, split="train", data_type="vanilla_harmful"):
    return {
        "example_id": f"{data_type}:{index}",
        "row_id": str(index),
        "prompt": f"prompt {data_type} {index}",
        "completion": "",
        "label": 1 if data_type.endswith("_harmful") else 0,
        "data_type": data_type,
        "source_family": (
            "adversarial" if data_type.startswith("adversarial") else "vanilla"
        ),
        "split": split,
        "tactics": ["encoding"] if data_type.startswith("adversarial") else [],
        "metadata": {
            "upstream_dataset_id": "allenai/wildjailbreak",
            "upstream_dataset_revision": "dataset-rev",
            "upstream_row_index": index,
        },
    }


def _response_record(index, *, split="train", data_type="vanilla_harmful"):
    return make_response_cache_record(
        _example(index, split=split, data_type=data_type),
        response=f"response {index}",
        config=_config(),
        generated_at="2026-05-24T12:00:00Z",
    )


def _write_config(
    tmp_path,
    *,
    data_path,
    response_cache_path,
    activation_cache_path,
    index_path,
    metadata_path,
):
    config = _config() | {
        "outputs": {
            "data_path": str(data_path),
            "response_cache_path": str(response_cache_path),
            "activation_cache_path": str(activation_cache_path),
            "activation_index_path": str(index_path),
            "activation_metadata_path": str(metadata_path),
        },
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return config_path


def _run_script(*args, check=True):
    script_path = Path("scripts/ccpp/extract_activations.py")
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        check=check,
        capture_output=True,
        text=True,
    )


def test_extract_activations_script_writes_mock_cache_and_metadata(tmp_path):
    data_path = tmp_path / "wildjailbreak.jsonl"
    response_cache_path = tmp_path / "responses.jsonl"
    response_metadata_path = tmp_path / "responses.metadata.json"
    activation_cache_path = tmp_path / "activations.npz"
    index_path = tmp_path / "activations.index.jsonl"
    metadata_path = tmp_path / "activations.metadata.json"
    data_path.write_text('{"example_id": "one"}\n', encoding="utf-8")
    write_response_cache(
        [
            _response_record(0, split="train", data_type="vanilla_harmful"),
            _response_record(1, split="transfer", data_type="adversarial_benign"),
        ],
        cache_path=response_cache_path,
        metadata_path=response_metadata_path,
        metadata={"schema_version": "agguardrails.response_cache.v1"},
    )
    config_path = _write_config(
        tmp_path,
        data_path=data_path,
        response_cache_path=response_cache_path,
        activation_cache_path=activation_cache_path,
        index_path=index_path,
        metadata_path=metadata_path,
    )

    result = _run_script("--config", str(config_path), "--mock")

    assert "Loaded 2 response records" in result.stdout
    with np.load(activation_cache_path) as cache:
        assert cache["activations"].shape == (2, 4)
        assert cache["labels"].tolist() == [1, 0]
    index_rows = [json.loads(line) for line in index_path.read_text().splitlines()]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert [row["vector_index"] for row in index_rows] == [0, 1]
    assert metadata["mode"] == "mock"
    assert metadata["config_path"] == str(config_path)
    assert metadata["dataset_artifact"]["path"] == str(data_path)
    assert metadata["response_cache_artifact"]["path"] == str(response_cache_path)
    assert metadata["activation"]["mode"] == "prompt_final"
    assert metadata["activation"]["source"] == "residual"
    assert metadata["activation"]["segment"] == {"enabled": False}
    assert metadata["counts"]["records"] == 2


def test_extract_activations_script_dry_run_does_not_write(tmp_path):
    data_path = tmp_path / "wildjailbreak.jsonl"
    response_cache_path = tmp_path / "responses.jsonl"
    activation_cache_path = tmp_path / "activations.npz"
    index_path = tmp_path / "activations.index.jsonl"
    metadata_path = tmp_path / "activations.metadata.json"
    data_path.write_text('{"example_id": "one"}\n', encoding="utf-8")
    write_response_cache(
        [_response_record(0, split="train", data_type="vanilla_harmful")],
        cache_path=response_cache_path,
        metadata_path=tmp_path / "responses.metadata.json",
        metadata={"schema_version": "agguardrails.response_cache.v1"},
    )
    config_path = _write_config(
        tmp_path,
        data_path=data_path,
        response_cache_path=response_cache_path,
        activation_cache_path=activation_cache_path,
        index_path=index_path,
        metadata_path=metadata_path,
    )

    result = _run_script("--config", str(config_path), "--dry-run")

    assert "vanilla_harmful:0" in result.stdout
    assert not activation_cache_path.exists()
    assert not index_path.exists()
    assert not metadata_path.exists()


def test_extract_activations_script_gates_non_mock_extraction(tmp_path):
    data_path = tmp_path / "wildjailbreak.jsonl"
    response_cache_path = tmp_path / "responses.jsonl"
    activation_cache_path = tmp_path / "activations.npz"
    index_path = tmp_path / "activations.index.jsonl"
    metadata_path = tmp_path / "activations.metadata.json"
    data_path.write_text('{"example_id": "one"}\n', encoding="utf-8")
    write_response_cache(
        [_response_record(0, split="train", data_type="vanilla_harmful")],
        cache_path=response_cache_path,
        metadata_path=tmp_path / "responses.metadata.json",
        metadata={"schema_version": "agguardrails.response_cache.v1"},
    )
    config_path = _write_config(
        tmp_path,
        data_path=data_path,
        response_cache_path=response_cache_path,
        activation_cache_path=activation_cache_path,
        index_path=index_path,
        metadata_path=metadata_path,
    )

    result = _run_script("--config", str(config_path), check=False)

    assert result.returncode != 0
    assert "Activation extraction is intentionally gated" in result.stderr
    assert not activation_cache_path.exists()
    assert not index_path.exists()
    assert not metadata_path.exists()
