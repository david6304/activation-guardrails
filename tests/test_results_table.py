"""Tests for the MVP results-table assembly script helpers."""

import json

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
