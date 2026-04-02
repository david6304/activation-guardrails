"""Tests for agguardrails.judge."""

from __future__ import annotations

import torch

from agguardrails.judge import _parse_refusal_label, label_refusals


def test_parse_refusal_label_refusal():
    assert _parse_refusal_label("refusal") == 1
    assert _parse_refusal_label("Refusal") == 1
    assert _parse_refusal_label("REFUSAL") == 1
    assert _parse_refusal_label("This is a refusal.") == 1


def test_parse_refusal_label_compliance():
    assert _parse_refusal_label("compliance") == 0
    assert _parse_refusal_label("Compliance") == 0
    assert _parse_refusal_label("This is compliance.") == 0


def test_parse_refusal_label_unparseable():
    assert _parse_refusal_label("") is None
    assert _parse_refusal_label("I don't know") is None
    assert _parse_refusal_label("yes") is None


def test_parse_refusal_label_first_keyword_wins():
    # "compliance" appears before "refusal" → compliance (0)
    assert _parse_refusal_label("compliance then refusal") == 0
    # "refusal" appears before "compliance" → refusal (1)
    assert _parse_refusal_label("refusal then compliance") == 1


def _make_fake_judge_model(responses: list[str]):
    """Return a mock model that generates a fixed sequence of text outputs."""

    class FakeTokenizer:
        pad_token_id = 0
        padding_side = "left"

        def __call__(self, texts, *, return_tensors, padding, truncation, max_length):
            n = len(texts)
            return {
                "input_ids": torch.zeros((n, 4), dtype=torch.long),
                "attention_mask": torch.ones((n, 4), dtype=torch.long),
            }

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return messages[-1]["content"]

        def decode(self, tokens, skip_special_tokens=True):
            # Pop from front of responses list.
            return responses.pop(0) if responses else ""

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, input_ids, attention_mask, **kwargs):
            n, seq_len = input_ids.shape
            return torch.cat([input_ids, torch.ones((n, 2), dtype=torch.long)], dim=1)

    return FakeModel(), FakeTokenizer()


def test_label_refusals_assigns_labels():
    records = [
        {"example_id": "h1", "prompt": "harmful", "response": "I cannot...", "source": "advbench", "source_label": "harmful", "label": 1},
        {"example_id": "b1", "prompt": "benign", "response": "Sure, here is...", "source": "alpaca", "source_label": "benign", "label": 0},
    ]
    judge_outputs = ["refusal", "compliance"]
    model, tokenizer = _make_fake_judge_model(judge_outputs)

    labelled = label_refusals(
        model,
        tokenizer,
        records,
        max_new_tokens=16,
        batch_size=2,
        temperature=0.0,
    )

    assert len(labelled) == 2
    assert labelled[0]["refusal_label"] == 1
    assert labelled[1]["refusal_label"] == 0
    assert "judge_output" in labelled[0]


def test_label_refusals_uses_fallback_for_unparseable():
    records = [
        {"example_id": "x1", "prompt": "p", "response": "r", "source": "s", "source_label": "sl", "label": 0},
    ]
    # Unparseable judge output.
    model, tokenizer = _make_fake_judge_model(["yes, probably"])

    labelled = label_refusals(
        model,
        tokenizer,
        records,
        max_new_tokens=16,
        batch_size=1,
        temperature=0.0,
        fallback_label=0,
    )

    assert labelled[0]["refusal_label"] == 0


def test_label_refusals_preserves_input_fields():
    records = [
        {"example_id": "h1", "prompt": "harmful", "response": "response", "source": "advbench",
         "source_label": "harmful", "label": 1, "extra_field": "preserved"},
    ]
    model, tokenizer = _make_fake_judge_model(["refusal"])

    labelled = label_refusals(
        model, tokenizer, records, max_new_tokens=16, batch_size=1, temperature=0.0
    )

    assert labelled[0]["extra_field"] == "preserved"
    assert labelled[0]["prompt"] == "harmful"
