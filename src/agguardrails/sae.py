"""Utilities for loading pretrained SAEs and encoding cached activations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from agguardrails.utils import batched


@dataclass(frozen=True)
class PretrainedSAESpec:
    """Configuration needed to load one pretrained SAE."""

    release: str
    sae_id: str
    layer: int
    width: int


def width_to_slug(width: int) -> str:
    """Convert a numeric SAE width to the Gemma Scope path fragment."""
    if width < 1000:
        return str(width)
    return f"{width // 1000}k"


def resolve_gemma_scope_sae_id(
    *,
    layer: int,
    width: int,
    variant: str = "canonical",
) -> str:
    """Resolve the SAE id used by Gemma Scope checkpoints in SAELens."""
    return f"layer_{layer}/width_{width_to_slug(width)}/{variant}"


def build_pretrained_sae_spec(
    *,
    release: str,
    layer: int,
    width: int,
    variant: str = "canonical",
) -> PretrainedSAESpec:
    """Build the pretrained SAE spec for one configured layer."""
    return PretrainedSAESpec(
        release=release,
        sae_id=resolve_gemma_scope_sae_id(layer=layer, width=width, variant=variant),
        layer=layer,
        width=width,
    )


def load_pretrained_sae(
    *,
    release: str,
    sae_id: str,
    device: str = "cpu",
    dtype: str = "float32",
):
    """Load one pretrained SAE via SAELens.

    The SAELens API has changed across major versions. In v6.x the documented
    path is ``SAE.from_pretrained(release=..., sae_id=..., device=..., dtype=...)``.
    Some versions return the SAE directly and older variants return a tuple.
    """
    try:
        from sae_lens import SAE
    except ImportError as exc:
        msg = (
            "sae_lens is not installed. Install the Phase C dependency first, "
            "for example `pip install sae-lens` in the active environment."
        )
        raise ImportError(msg) from exc

    try:
        loaded = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
            dtype=dtype,
        )
    except ValueError as exc:
        if "not found in release" in str(exc):
            msg = (
                f"{exc} Hint: many Gemma Scope releases use explicit SAE variants "
                "such as `average_l0_14` rather than a `canonical` alias. "
                "Set `sae.variant` in the config to one of the valid IDs listed in the error."
            )
            raise ValueError(msg) from exc
        raise
    sae = loaded[0] if isinstance(loaded, tuple) else loaded
    if hasattr(sae, "eval"):
        sae.eval()
    return sae


def _infer_sae_device(sae) -> torch.device:
    if hasattr(sae, "cfg") and getattr(sae.cfg, "device", None) is not None:
        return torch.device(str(sae.cfg.device))
    if hasattr(sae, "parameters"):
        try:
            return next(sae.parameters()).device
        except StopIteration:
            pass
    return torch.device("cpu")


def encode_with_sae(
    *,
    sae,
    activations: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Encode dense hidden states into SAE feature activations."""
    if activations.ndim != 2:
        raise ValueError(
            f"Expected activations with shape [n_examples, hidden_size], got {activations.shape}"
        )

    device = _infer_sae_device(sae)
    outputs: list[np.ndarray] = []

    for batch in batched(activations, batch_size):
        batch_arr = np.asarray(batch, dtype=np.float32)
        batch_tensor = torch.from_numpy(batch_arr).to(device=device)
        with torch.inference_mode():
            encoded = sae.encode(batch_tensor)
        outputs.append(encoded.detach().to(dtype=torch.float32).cpu().numpy())

    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, 0), dtype=np.float32)
