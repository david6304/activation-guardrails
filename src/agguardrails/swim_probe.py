"""SWiM objective and streaming scoring helpers for CC++ probes."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from agguardrails.activations import ActivationExample


@dataclass(frozen=True)
class SwimTrainingConfig:
    seed: int
    epochs: int
    learning_rate: float
    weight_decay: float
    batch_size: int
    window_size: int
    softmax_temperature: float


class LinearSWiMProbe(nn.Module):
    """Token-level linear probe trained with the SWiM sequence objective."""

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, features: Tensor) -> Tensor:
        if features.ndim != 2:
            raise ValueError("features must have shape [tokens, feature_dim]")
        return self.linear(features).squeeze(-1)


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


def train_linear_swim_probe(
    examples: list[ActivationExample],
    *,
    config: SwimTrainingConfig,
) -> tuple[LinearSWiMProbe, list[float]]:
    """Train a minimal linear SWiM probe on cached activation examples."""

    if not examples:
        raise ValueError("examples must be non-empty")
    feature_dim = _feature_dim(examples)
    _set_seed(config.seed)
    model = LinearSWiMProbe(feature_dim)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    losses: list[float] = []
    for epoch in range(config.epochs):
        del epoch
        shuffled = list(examples)
        random.Random(config.seed).shuffle(shuffled)
        for batch in _batches(shuffled, config.batch_size):
            batch_features = [
                torch.as_tensor(example.features, dtype=torch.float32)
                for example in batch
            ]
            labels = torch.tensor(
                [example.label for example in batch],
                dtype=torch.float32,
            )
            token_logits = [model(features) for features in batch_features]
            loss = swim_batch_loss(
                token_logits,
                labels,
                window_size=config.window_size,
                softmax_temperature=config.softmax_temperature,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return model, losses


def score_activation_examples(
    model: LinearSWiMProbe,
    examples: Sequence[ActivationExample],
    *,
    ema_gamma: float | None,
) -> list[dict[str, float | int | str]]:
    """Score cached activation examples with max/any-token streaming scoring."""

    rows: list[dict[str, float | int | str]] = []
    model.eval()
    with torch.no_grad():
        for example in examples:
            features = torch.as_tensor(example.features, dtype=torch.float32)
            token_logits = model(features)
            score = streaming_sequence_score(token_logits, ema_gamma=ema_gamma)
            rows.append(
                {
                    "example_id": example.example_id,
                    "group_id": example.group_id,
                    "split": example.split,
                    "label": example.label,
                    "score": float(score.cpu()),
                }
            )
    return rows


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


def _feature_dim(examples: Sequence[ActivationExample]) -> int:
    feature_dims = {int(example.features.shape[1]) for example in examples}
    if len(feature_dims) != 1:
        msg = f"activation feature dimensions differ: {sorted(feature_dims)}"
        raise ValueError(msg)
    return next(iter(feature_dims))


def _batches(examples: Sequence[ActivationExample], batch_size: int):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for start in range(0, len(examples), batch_size):
        yield list(examples[start : start + batch_size])


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
