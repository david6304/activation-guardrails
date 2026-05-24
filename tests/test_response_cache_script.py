import json
import subprocess
import sys
from pathlib import Path

import yaml


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


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_config(tmp_path, *, data_path, cache_path, metadata_path, limit=3):
    config_path = tmp_path / "config.yaml"
    config = {
        "model": {"id": "google/gemma-2-9b-it", "revision": "model-rev"},
        "tokenizer": {"id": "google/gemma-2-9b-it", "revision": "tokenizer-rev"},
        "sampling": {"seed": 123},
        "generation": {
            "seed": 456,
            "params": {"max_new_tokens": 16, "do_sample": False},
            "runtime": {
                "backend": "transformers",
                "device_map": "auto",
                "torch_dtype": "auto",
            },
        },
        "response_cache": {
            "debug_subset": {
                "enabled": True,
                "limit": limit,
                "splits": ["train", "transfer"],
            },
        },
        "outputs": {
            "data_path": str(data_path),
            "metadata_path": str(tmp_path / "dataset.metadata.json"),
            "response_cache_path": str(cache_path),
            "response_cache_metadata_path": str(metadata_path),
        },
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return config_path


def _run_script(*args):
    script_path = Path("scripts/ccpp/generate_response_cache.py")
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_generate_response_cache_script_writes_mock_cache_and_metadata(tmp_path):
    data_path = tmp_path / "wildjailbreak.jsonl"
    cache_path = tmp_path / "responses.jsonl"
    metadata_path = tmp_path / "responses.metadata.json"
    rows = [
        _example(0, split="train", data_type="vanilla_harmful"),
        _example(1, split="train", data_type="vanilla_benign"),
        _example(2, split="test", data_type="vanilla_harmful"),
        _example(3, split="transfer", data_type="adversarial_benign"),
        _example(4, split="transfer", data_type="adversarial_harmful"),
    ]
    _write_jsonl(data_path, rows)
    config_path = _write_config(
        tmp_path,
        data_path=data_path,
        cache_path=cache_path,
        metadata_path=metadata_path,
        limit=3,
    )

    result = _run_script("--config", str(config_path), "--mock")

    assert "Selected 3 uncached examples" in result.stdout
    written_rows = [json.loads(line) for line in cache_path.read_text().splitlines()]
    metadata = json.loads(metadata_path.read_text())

    assert len(written_rows) == 3
    assert {row["split"] for row in written_rows} <= {"train", "transfer"}
    assert all(row["response"].startswith("[mock response]") for row in written_rows)
    assert metadata["config_path"] == str(config_path)
    assert metadata["mode"] == "mock"
    assert metadata["dataset_artifact"]["path"] == str(data_path)
    assert len(metadata["dataset_artifact"]["sha256"]) == 64
    assert metadata["model"] == {
        "id": "google/gemma-2-9b-it",
        "revision": "model-rev",
    }
    assert metadata["tokenizer"] == {
        "id": "google/gemma-2-9b-it",
        "revision": "tokenizer-rev",
    }
    assert metadata["generation_params"] == {
        "max_new_tokens": 16,
        "do_sample": False,
    }
    assert metadata["seed"] == 456
    assert metadata["counts"]["records"] == 3
    assert metadata["selection"] == {
        "debug_subset_enabled": True,
        "limit": 3,
        "splits": ["train", "transfer"],
    }
    assert metadata["resume"] == {
        "existing_records": 0,
        "new_records": 3,
        "skipped_cached_example_ids": 0,
    }


def test_generate_response_cache_script_resume_does_not_duplicate_or_top_off(tmp_path):
    data_path = tmp_path / "wildjailbreak.jsonl"
    cache_path = tmp_path / "responses.jsonl"
    metadata_path = tmp_path / "responses.metadata.json"
    rows = [
        _example(index, split="train", data_type="vanilla_harmful")
        for index in range(8)
    ]
    _write_jsonl(data_path, rows)
    config_path = _write_config(
        tmp_path,
        data_path=data_path,
        cache_path=cache_path,
        metadata_path=metadata_path,
        limit=4,
    )

    _run_script("--config", str(config_path), "--mock")
    second = _run_script("--config", str(config_path), "--mock")

    assert "Selected 0 uncached examples" in second.stdout
    written_rows = [json.loads(line) for line in cache_path.read_text().splitlines()]
    metadata = json.loads(metadata_path.read_text())
    example_ids = [row["example_id"] for row in written_rows]

    assert len(written_rows) == 4
    assert len(example_ids) == len(set(example_ids))
    assert metadata["counts"]["records"] == 4
    assert metadata["resume"] == {
        "existing_records": 4,
        "new_records": 0,
        "skipped_cached_example_ids": 4,
    }


def test_generate_response_cache_script_dry_run_does_not_write(tmp_path):
    data_path = tmp_path / "wildjailbreak.jsonl"
    cache_path = tmp_path / "responses.jsonl"
    metadata_path = tmp_path / "responses.metadata.json"
    _write_jsonl(
        data_path,
        [_example(0, split="train", data_type="vanilla_harmful")],
    )
    config_path = _write_config(
        tmp_path,
        data_path=data_path,
        cache_path=cache_path,
        metadata_path=metadata_path,
        limit=1,
    )

    result = _run_script("--config", str(config_path), "--dry-run")

    assert "Selected 1 uncached examples" in result.stdout
    assert "vanilla_harmful:0" in result.stdout
    assert not cache_path.exists()
    assert not metadata_path.exists()


def test_generate_response_cache_script_gates_real_generation(tmp_path):
    data_path = tmp_path / "wildjailbreak.jsonl"
    cache_path = tmp_path / "responses.jsonl"
    metadata_path = tmp_path / "responses.metadata.json"
    _write_jsonl(
        data_path,
        [_example(0, split="train", data_type="vanilla_harmful")],
    )
    config_path = _write_config(
        tmp_path,
        data_path=data_path,
        cache_path=cache_path,
        metadata_path=metadata_path,
        limit=1,
    )
    script_path = Path("scripts/ccpp/generate_response_cache.py")

    result = subprocess.run(
        [sys.executable, str(script_path), "--config", str(config_path)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Response generation is intentionally gated" in result.stderr
    assert not cache_path.exists()
    assert not metadata_path.exists()


def test_generate_response_cache_script_real_flag_handles_empty_selection(tmp_path):
    data_path = tmp_path / "wildjailbreak.jsonl"
    cache_path = tmp_path / "responses.jsonl"
    metadata_path = tmp_path / "responses.metadata.json"
    _write_jsonl(
        data_path,
        [_example(0, split="train", data_type="vanilla_harmful")],
    )
    config_path = _write_config(
        tmp_path,
        data_path=data_path,
        cache_path=cache_path,
        metadata_path=metadata_path,
        limit=0,
    )

    result = _run_script("--config", str(config_path), "--generate-real")

    assert "Selected 0 uncached examples" in result.stdout
    assert cache_path.read_text(encoding="utf-8") == ""
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["mode"] == "model"
    assert metadata["generation_backend"] == {
        "backend": "transformers",
        "device_map": "auto",
        "torch_dtype": "auto",
    }
    assert metadata["counts"]["records"] == 0
    assert metadata["resume"] == {
        "existing_records": 0,
        "new_records": 0,
        "skipped_cached_example_ids": 0,
    }
