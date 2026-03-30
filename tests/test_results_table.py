"""Tests for results-table assembly script helpers."""

import json

from scripts.main.make_results_table import maybe_best_layer_rows, maybe_text_rows
from scripts.mvp.make_results_table import maybe_probe_test_row


def test_maybe_probe_test_row_returns_none_when_missing(tmp_path):
    assert maybe_probe_test_row(tmp_path / "missing.json") is None


def test_maybe_probe_test_row_extracts_best_layer_row(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "best_layer": 16,
                "per_layer": {
                    "8": {"test": {"model_name": "activation_probe", "layer": 8}},
                    "16": {"test": {"model_name": "activation_probe", "layer": 16}},
                },
            }
        )
    )

    row = maybe_probe_test_row(path)
    assert row == {"model_name": "activation_probe", "layer": 16}


def test_maybe_text_rows_extracts_available_splits(tmp_path):
    path = tmp_path / "text_metrics.json"
    path.write_text(
        json.dumps(
            {
                "val_metrics": {"model_name": "tfidf_lr", "split": "val"},
                "test_metrics": {"model_name": "tfidf_lr", "split": "test"},
                "adversarial_metrics": {"model_name": "tfidf_lr", "split": "adversarial"},
            }
        )
    )

    rows = maybe_text_rows(path)
    assert [row["split"] for row in rows] == ["val", "test", "adversarial"]


def test_maybe_best_layer_rows_extracts_all_present_splits(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "best_layer": 20,
                "per_layer": {
                    "20": {
                        "val": {"model_name": "sae_probe", "split": "val", "layer": 20},
                        "test": {"model_name": "sae_probe", "split": "test", "layer": 20},
                        "adversarial": {
                            "model_name": "sae_probe",
                            "split": "adversarial",
                            "layer": 20,
                        },
                    }
                },
            }
        )
    )

    rows = maybe_best_layer_rows(path)
    assert [row["split"] for row in rows] == ["val", "test", "adversarial"]
