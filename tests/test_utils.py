"""Tests for agguardrails.utils."""

import numpy as np
import torch

from agguardrails.utils import batched, get_git_hash, seed_everything, timestamp_str


class TestSeedEverything:
    def test_determinism_numpy(self):
        seed_everything(123)
        a = np.random.rand(5)
        seed_everything(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_determinism_torch(self):
        seed_everything(123)
        a = torch.randn(5)
        seed_everything(123)
        b = torch.randn(5)
        assert torch.equal(a, b)

    def test_different_seeds_differ(self):
        seed_everything(1)
        a = np.random.rand(5)
        seed_everything(2)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)


class TestBatched:
    def test_exact_division(self):
        result = list(batched(range(6), 3))
        assert result == [[0, 1, 2], [3, 4, 5]]

    def test_remainder(self):
        result = list(batched(range(7), 3))
        assert result == [[0, 1, 2], [3, 4, 5], [6]]

    def test_empty(self):
        result = list(batched([], 3))
        assert result == []

    def test_single_element_batches(self):
        result = list(batched(range(3), 1))
        assert result == [[0], [1], [2]]


class TestProvenance:
    def test_git_hash_returns_string(self):
        h = get_git_hash()
        assert isinstance(h, str)
        assert len(h) > 0

    def test_timestamp_format(self):
        ts = timestamp_str()
        assert ts.endswith("Z")
        assert "T" in ts
