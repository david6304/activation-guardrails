import json
import subprocess
import sys
from pathlib import Path

import yaml


def _example(index, *, split, data_type):
    label = 1 if data_type.endswith("_harmful") else 0
    source_family = "adversarial" if data_type.startswith("adversarial") else "vanilla"
    topic = "malware exploit harmful" if label else "garden recipe benign"
    return {
        "example_id": f"{data_type}:{index}",
        "row_id": str(index),
        "prompt": f"{topic} {source_family} prompt {index}",
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


def _examples():
    rows = []
    for split, count in {"train": 8, "val": 4, "test": 4}.items():
        for index in range(count):
            rows.append(_example(index, split=split, data_type="vanilla_harmful"))
            rows.append(_example(index, split=split, data_type="vanilla_benign"))
    for index in range(4):
        rows.append(
            _example(index, split="transfer", data_type="adversarial_harmful")
        )
        rows.append(_example(index, split="transfer", data_type="adversarial_benign"))
    return rows


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_config(
    tmp_path,
    *,
    data_path,
    model_path,
    scores_path,
    metrics_path,
    table_path,
):
    config_path = tmp_path / "config.yaml"
    config = {
        "model": {"id": "google/gemma-2-9b-it", "revision": "model-rev"},
        "dataset": {
            "id": "allenai/wildjailbreak",
            "revision": "dataset-rev",
            "name": "train",
            "split": "train",
        },
        "sampling": {"seed": 123},
        "text_baseline": {
            "seed": 456,
            "threshold": {"max_fpr": 0.01},
            "tfidf": {"ngram_range": [1, 2]},
            "logistic_regression": {"max_iter": 200, "solver": "liblinear"},
        },
        "outputs": {
            "data_path": str(data_path),
            "text_baseline_model_path": str(model_path),
            "text_baseline_scores_path": str(scores_path),
            "text_baseline_metrics_path": str(metrics_path),
            "text_baseline_table_path": str(table_path),
        },
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)
    return config_path


def _run_script(*args):
    script_path = Path("scripts/ccpp/train_text_baseline.py")
    return subprocess.run(
        [sys.executable, str(script_path), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def test_train_text_baseline_script_writes_outputs(tmp_path):
    data_path = tmp_path / "wildjailbreak.jsonl"
    model_path = tmp_path / "tfidf.joblib"
    scores_path = tmp_path / "scores.jsonl"
    metrics_path = tmp_path / "metrics.json"
    table_path = tmp_path / "table.csv"
    _write_jsonl(data_path, _examples())
    config_path = _write_config(
        tmp_path,
        data_path=data_path,
        model_path=model_path,
        scores_path=scores_path,
        metrics_path=metrics_path,
        table_path=table_path,
    )

    result = _run_script("--config", str(config_path))

    assert "Trained tfidf_logistic_regression" in result.stdout
    assert "Selected threshold" in result.stdout
    assert model_path.exists()
    assert scores_path.exists()
    assert table_path.exists()
    metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metadata["config_path"] == str(config_path)
    assert metadata["threshold_rule"]["selection_split"] == "val"
    assert metadata["metrics"]["vanilla_test"]["positives"] == 4
    assert metadata["metrics"]["adversarial_transfer"]["negatives"] == 4


def test_train_text_baseline_script_dry_run_does_not_write(tmp_path):
    data_path = tmp_path / "wildjailbreak.jsonl"
    model_path = tmp_path / "tfidf.joblib"
    scores_path = tmp_path / "scores.jsonl"
    metrics_path = tmp_path / "metrics.json"
    table_path = tmp_path / "table.csv"
    _write_jsonl(data_path, _examples())
    config_path = _write_config(
        tmp_path,
        data_path=data_path,
        model_path=model_path,
        scores_path=scores_path,
        metrics_path=metrics_path,
        table_path=table_path,
    )

    result = _run_script("--config", str(config_path), "--dry-run")

    assert "Loaded 40 normalized examples" in result.stdout
    assert "transfer\t8" in result.stdout
    assert not model_path.exists()
    assert not scores_path.exists()
    assert not metrics_path.exists()
    assert not table_path.exists()
