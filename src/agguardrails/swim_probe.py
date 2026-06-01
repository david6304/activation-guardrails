"""SWiM objective and streaming scoring helpers for CC++ probes."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def sliding_window_mean(logits: Tensor, window_size: int) -> Tensor:
    """Return causal sliding-window means for a 1D token-logit tensor."""

    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if logits.ndim != 1:
        raise ValueError("logits must be a 1D tensor")
    if logits.numel() == 0:
        raise ValueError("logits must be non-empty")

    cumsum = torch.cumsum(logits, dim=0)
    starts = torch.arange(logits.numel(), device=logits.device) - window_size
    previous = torch.where(
        starts >= 0,
        cumsum[starts.clamp_min(0)],
        torch.zeros_like(cumsum),
    )
    totals = cumsum - previous
    counts = torch.arange(1, logits.numel() + 1, device=logits.device).clamp_max(
        window_size
    )
    return totals / counts.to(dtype=logits.dtype)


def swim_sequence_loss(
    token_logits: Tensor,
    label: int | float | Tensor,
    *,
    window_size: int,
    softmax_temperature: float,
) -> Tensor:
    """Compute CC++ SWiM softmax-weighted BCE for one sequence."""

    if softmax_temperature <= 0:
        raise ValueError("softmax_temperature must be > 0")

    window_logits = sliding_window_mean(token_logits, window_size)
    weights = torch.softmax(window_logits.detach() / softmax_temperature, dim=0)
    labels = torch.full_like(window_logits, float(label))
    per_window_loss = F.binary_cross_entropy_with_logits(
        window_logits,
        labels,
        reduction="none",
    )
    return torch.sum(weights * per_window_loss)


def swim_batch_loss(
    batch_token_logits: list[Tensor],
    labels: Tensor,
    *,
    window_size: int,
    softmax_temperature: float,
) -> Tensor:
    """Average SWiM loss across variable-length sequences."""

    if len(batch_token_logits) == 0:
        raise ValueError("batch_token_logits must be non-empty")
    if labels.ndim != 1 or labels.numel() != len(batch_token_logits):
        raise ValueError("labels must be 1D and aligned with batch_token_logits")

    losses = [
        swim_sequence_loss(
            token_logits,
            labels[index],
            window_size=window_size,
            softmax_temperature=softmax_temperature,
        )
        for index, token_logits in enumerate(batch_token_logits)
    ]
    return torch.stack(losses).mean()


def ema_smooth(logits: Tensor, gamma: float) -> Tensor:
    """Causal exponential moving average over token logits."""

    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if logits.ndim != 1:
        raise ValueError("logits must be a 1D tensor")
    if logits.numel() == 0:
        raise ValueError("logits must be non-empty")

    smoothed = torch.empty_like(logits)
    smoothed[0] = logits[0]
    for index in range(1, logits.numel()):
        smoothed[index] = gamma * logits[index] + (1 - gamma) * smoothed[index - 1]
    return smoothed


def streaming_sequence_score(
    token_logits: Tensor,
    *,
    ema_gamma: float | None = None,
) -> Tensor:
    """Score a sequence by the maximum streaming logit after optional EMA."""

    logits = (
        ema_smooth(token_logits, ema_gamma)
        if ema_gamma is not None
        else token_logits
    )
    if logits.ndim != 1:
        raise ValueError("token_logits must be a 1D tensor")
    if logits.numel() == 0:
        raise ValueError("token_logits must be non-empty")
    return torch.max(logits)
