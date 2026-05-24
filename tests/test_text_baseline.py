import csv
import json

import joblib

from agguardrails.text_baseline import (
    SCHEMA_VERSION,
    run_tfidf_logistic_baseline,
    split_baseline_examples,
)


def _example(index, *, split, data_type):
    label = 1 if data_type.endswith("_harmful") else 0
    source_family = "adversarial" if data_type.startswith("adversarial") else "vanilla"
    topic = "malware exploit harmful" if label else "garden recipe benign"
    prompt = f"{topic} {source_family} prompt {index}"
    return {
        "example_id": f"{data_type}:{index}",
        "row_id": str(index),
        "prompt": prompt,
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
    split_counts = {"train": 8, "val": 4, "test": 4}
    for split, count in split_counts.items():
        for index in range(count):
            rows.append(_example(index, split=split, data_type="vanilla_harmful"))
            rows.append(_example(index, split=split, data_type="vanilla_benign"))
    for index in range(5):
        rows.append(
            _example(index, split="transfer", data_type="adversarial_harmful")
        )
        rows.append(_example(index, split="transfer", data_type="adversarial_benign"))
    return rows


def _config():
    return {
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
            "tfidf": {"ngram_range": [1, 2], "min_df": 1},
            "logistic_regression": {"max_iter": 200, "solver": "liblinear"},
        },
    }


def _write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_text_baseline_trains_on_vanilla_and_writes_metadata_rich_outputs(tmp_path):
    examples = _examples()
    data_path = tmp_path / "wildjailbreak.jsonl"
    model_path = tmp_path / "tfidf.joblib"
    scores_path = tmp_path / "scores.jsonl"
    metrics_path = tmp_path / "metrics.json"
    table_path = tmp_path / "table.csv"
    _write_jsonl(data_path, examples)

    metadata = run_tfidf_logistic_baseline(
        examples=examples,
        config=_config(),
        config_path="configs/ccpp/test.yaml",
        dataset_artifact_path=data_path,
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
    assert len(metadata["dataset_artifact"]["sha256"]) == 64
    assert metadata["training"]["counts"]["by_data_type"] == {
        "vanilla_benign": 8,
        "vanilla_harmful": 8,
    }
    assert metadata["evaluation"]["adversarial_transfer"]["counts"][
        "by_data_type"
    ] == {
        "adversarial_benign": 5,
        "adversarial_harmful": 5,
    }
    assert metadata["threshold_rule"]["max_fpr"] == 0.01
    assert metadata["threshold_rule"]["score_direction"] == "larger_is_more_harmful"
    assert metadata["metrics"]["validation"]["fpr"] <= 0.01
    assert metadata["metrics"]["vanilla_test"]["threshold"] == metadata[
        "threshold_rule"
    ]["frozen_threshold"]
    assert metadata["metrics"]["adversarial_transfer"]["threshold"] == metadata[
        "threshold_rule"
    ]["frozen_threshold"]
    assert metadata["package_versions"]["scikit-learn"]

    assert joblib.load(model_path).classes_.tolist() == [0, 1]
    written_metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
    score_rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
    table_rows = list(csv.DictReader(table_path.open(encoding="utf-8")))

    assert written_metadata == metadata
    assert len(score_rows) == len(examples)
    assert all(len(row["prompt_sha256"]) == 64 for row in score_rows)
    assert [row["split"] for row in table_rows] == [
        "vanilla_val",
        "vanilla_test",
        "adversarial_transfer",
    ]


def test_split_baseline_examples_rejects_one_class_eval_split():
    rows = [row for row in _examples() if row["data_type"] != "vanilla_benign"]

    try:
        split_baseline_examples(rows)
    except ValueError as exc:
        assert "must contain both labels" in str(exc)
    else:
        raise AssertionError("Expected one-class split validation to fail")
