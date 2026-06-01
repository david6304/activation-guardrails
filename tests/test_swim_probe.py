from __future__ import annotations

import numpy as np
import pytest
import torch

from agguardrails.activations import ActivationExample
from agguardrails.swim_probe import (
    SwimTrainingConfig,
    ema_smooth,
    score_activation_examples,
    sliding_window_mean,
    streaming_sequence_score,
    swim_batch_loss,
    swim_sequence_loss,
    train_linear_swim_probe,
)


def test_sliding_window_mean_uses_available_tokens_for_short_prefixes() -> None:
    logits = torch.tensor([1.0, 3.0, 5.0, 7.0])

    means = sliding_window_mean(logits, window_size=3)

    torch.testing.assert_close(means, torch.tensor([1.0, 2.0, 3.0, 5.0]))


def test_sliding_window_mean_handles_sequence_shorter_than_window() -> None:
    logits = torch.tensor([2.0, 4.0])

    means = sliding_window_mean(logits, window_size=16)

    torch.testing.assert_close(means, torch.tensor([2.0, 3.0]))


def test_swim_sequence_loss_backpropagates_to_high_scoring_window() -> None:
    token_logits = torch.tensor([-2.0, -1.0, 5.0], requires_grad=True)

    loss = swim_sequence_loss(
        token_logits,
        1,
        window_size=1,
        softmax_temperature=0.1,
    )
    loss.backward()

    assert token_logits.grad is not None
    assert abs(float(token_logits.grad[2])) > abs(float(token_logits.grad[0]))


def test_swim_batch_loss_averages_variable_length_sequences() -> None:
    batch = [torch.tensor([0.0, 1.0]), torch.tensor([-1.0, -2.0, -3.0])]
    labels = torch.tensor([1.0, 0.0])

    loss = swim_batch_loss(
        batch,
        labels,
        window_size=2,
        softmax_temperature=1.0,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_ema_smooth_is_causal() -> None:
    logits = torch.tensor([0.0, 10.0, 0.0])

    smoothed = ema_smooth(logits, gamma=0.5)

    torch.testing.assert_close(smoothed, torch.tensor([0.0, 5.0, 2.5]))


def test_streaming_sequence_score_flags_any_token_after_smoothing() -> None:
    logits = torch.tensor([-1.0, 3.0, 0.0])

    raw_score = streaming_sequence_score(logits)
    smoothed_score = streaming_sequence_score(logits, ema_gamma=0.5)

    assert raw_score.item() == pytest.approx(3.0)
    assert smoothed_score.item() == pytest.approx(1.0)


def test_sliding_window_mean_rejects_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="1D"):
        sliding_window_mean(torch.zeros(2, 2), window_size=2)


def test_train_linear_swim_probe_scores_separable_synthetic_features() -> None:
    examples = [
        _activation_example("p1", 1, np.full((3, 2), 2.0, dtype=np.float32)),
        _activation_example("p2", 1, np.full((4, 2), 2.5, dtype=np.float32)),
        _activation_example("n1", 0, np.full((3, 2), -2.0, dtype=np.float32)),
        _activation_example("n2", 0, np.full((4, 2), -2.5, dtype=np.float32)),
    ]
    model, losses = train_linear_swim_probe(
        examples,
        config=SwimTrainingConfig(
            seed=1,
            epochs=20,
            learning_rate=0.05,
            weight_decay=0.0,
            batch_size=2,
            window_size=2,
            softmax_temperature=1.0,
        ),
    )

    scores = score_activation_examples(model, examples, ema_gamma=None)
    positive_scores = [row["score"] for row in scores if row["label"] == 1]
    negative_scores = [row["score"] for row in scores if row["label"] == 0]

    assert losses[-1] < losses[0]
    assert min(positive_scores) > max(negative_scores)


def _activation_example(example_id: str, label: int, features: np.ndarray):
    return ActivationExample(
        example_id=example_id,
        label=label,
        split="train",
        group_id=example_id,
        features=features,
        token_ids=np.arange(features.shape[0], dtype=np.int64),
        token_mask=np.ones(features.shape[0], dtype=bool),
        layers=[1],
        metadata={},
    )
