"""Tests for agguardrails.io."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from agguardrails.io import (
    load_artifact,
    read_jsonl,
    save_artifact,
    save_metadata,
    write_jsonl,
)


class TestJsonl:
    def test_round_trip(self, tmp_path):
        records = [{"a": 1, "b": "hello"}, {"a": 2, "b": "world"}]
        path = tmp_path / "test.jsonl"
        write_jsonl(path, records)
        loaded = read_jsonl(path)
        assert loaded == records

    def test_empty_records(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        write_jsonl(path, [])
        loaded = read_jsonl(path)
        assert loaded == []

    def test_unicode(self, tmp_path):
        records = [{"text": "caf\u00e9 \u2603"}]
        path = tmp_path / "unicode.jsonl"
        write_jsonl(path, records)
        loaded = read_jsonl(path)
        assert loaded == records

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "test.jsonl"
        write_jsonl(path, [{"x": 1}])
        assert path.exists()


class TestArtifacts:
    def test_numpy_round_trip(self, tmp_path):
        arr = np.random.rand(10, 5).astype(np.float32)
        saved_path = save_artifact(arr, tmp_path / "test_array")
        assert saved_path.suffix == ".npz"
        loaded = load_artifact(saved_path)
        np.testing.assert_array_almost_equal(arr, loaded)

    def test_sklearn_round_trip(self, tmp_path):
        model = LogisticRegression(C=0.5, max_iter=100, random_state=42)
        X = np.random.rand(20, 3)
        y = np.array([0] * 10 + [1] * 10)
        model.fit(X, y)

        saved_path = save_artifact(model, tmp_path / "model")
        assert saved_path.suffix == ".joblib"
        loaded = load_artifact(saved_path)
        np.testing.assert_array_equal(model.predict(X), loaded.predict(X))

    def test_unsupported_extension(self, tmp_path):
        path = tmp_path / "bad.xyz"
        path.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported"):
            load_artifact(path)


class TestMetadata:
    def test_saves_with_git_and_timestamp(self, tmp_path):
        path = tmp_path / "meta.json"
        save_metadata(path, seed=42, config="mvp.yaml")
        with path.open() as f:
            meta = json.load(f)
        assert "git_hash" in meta
        assert "timestamp" in meta
        assert meta["seed"] == 42
        assert meta["config"] == "mvp.yaml"
