"""Tests for agguardrails.sae."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from agguardrails.sae import (
    build_pretrained_sae_spec,
    encode_with_sae,
    load_pretrained_sae,
    resolve_gemma_scope_sae_id,
    width_to_slug,
)


def test_width_to_slug_formats_gemma_scope_widths():
    assert width_to_slug(16384) == "16k"
    assert width_to_slug(131072) == "131k"
    assert width_to_slug(1000) == "1k"


def test_resolve_gemma_scope_sae_id_matches_expected_pattern():
    assert (
        resolve_gemma_scope_sae_id(layer=20, width=16384)
        == "layer_20/width_16k/canonical"
    )


def test_build_pretrained_sae_spec_uses_resolved_id():
    spec = build_pretrained_sae_spec(
        release="gemma-scope-9b-it-res",
        layer=31,
        width=16384,
    )
    assert spec.release == "gemma-scope-9b-it-res"
    assert spec.layer == 31
    assert spec.width == 16384
    assert spec.sae_id == "layer_31/width_16k/canonical"


def test_load_pretrained_sae_raises_helpful_error_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "sae_lens", None)
    with pytest.raises(ImportError, match="sae_lens is not installed"):
        load_pretrained_sae(release="release", sae_id="sae_id")


def test_load_pretrained_sae_accepts_tuple_return(monkeypatch):
    class FakeSAEClass:
        @staticmethod
        def from_pretrained(**kwargs):
            sae = SimpleNamespace(eval=lambda: None)
            return (sae, {"release": kwargs["release"]})

    fake_module = ModuleType("sae_lens")
    fake_module.SAE = FakeSAEClass
    monkeypatch.setitem(sys.modules, "sae_lens", fake_module)

    sae = load_pretrained_sae(
        release="gemma-scope-9b-it-res",
        sae_id="layer_9/width_16k/canonical",
    )

    assert hasattr(sae, "eval")


def test_load_pretrained_sae_adds_helpful_hint_for_missing_variant(monkeypatch):
    class FakeSAEClass:
        @staticmethod
        def from_pretrained(**kwargs):
            raise ValueError(
                "ID layer_9/width_16k/canonical not found in release "
                "gemma-scope-9b-it-res. Valid IDs are ['layer_9/width_16k/average_l0_14']."
            )

    fake_module = ModuleType("sae_lens")
    fake_module.SAE = FakeSAEClass
    monkeypatch.setitem(sys.modules, "sae_lens", fake_module)

    with pytest.raises(ValueError, match="Set `sae.variant` in the config"):
        load_pretrained_sae(
            release="gemma-scope-9b-it-res",
            sae_id="layer_9/width_16k/canonical",
        )


def test_encode_with_sae_batches_and_returns_float32():
    class FakeSAE:
        cfg = SimpleNamespace(device="cpu")

        def encode(self, batch):
            return batch[:, :2].to(torch.float64) + 1

    activations = np.arange(12, dtype=np.float32).reshape(3, 4)
    encoded = encode_with_sae(
        sae=FakeSAE(),
        activations=activations,
        batch_size=2,
    )

    assert encoded.dtype == np.float32
    assert encoded.shape == (3, 2)
    np.testing.assert_array_equal(
        encoded,
        np.array([[1, 2], [5, 6], [9, 10]], dtype=np.float32),
    )
