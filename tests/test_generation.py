"""Tests for agguardrails.generation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from agguardrails.data import PromptExample
from agguardrails.generation import _load_completed_ids, generate_responses


def _make_examples(n: int) -> list[PromptExample]:
    return [
        PromptExample(
            example_id=f"ex_{i}",
            prompt=f"prompt {i}",
            label=1,
            split="train",
            source="advbench",
            source_id=str(i),
            source_label="harmful",
        )
        for i in range(n)
    ]


def _make_fake_model_and_tokenizer():
    """Minimal mock model and tokenizer for generation tests."""

    class FakeTokenizer:
        pad_token_id = 0
        padding_side = "left"

        def __call__(self, texts, *, return_tensors, padding, truncation, max_length):
            n = len(texts)
            input_ids = torch.zeros((n, 4), dtype=torch.long)
            attention_mask = torch.ones((n, 4), dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            return messages[-1]["content"]

        def decode(self, tokens, skip_special_tokens=True):
            return "I cannot assist with that."

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, input_ids, attention_mask, **kwargs):
            # Return input_ids with 2 new tokens appended.
            n, seq_len = input_ids.shape
            new_tokens = torch.ones((n, 2), dtype=torch.long)
            return torch.cat([input_ids, new_tokens], dim=1)

    return FakeModel(), FakeTokenizer()


def test_load_completed_ids_empty_file(tmp_path):
    path = tmp_path / "responses.jsonl"
    assert _load_completed_ids(path) == set()


def test_load_completed_ids_reads_existing_records(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text(
        json.dumps({"example_id": "ex_0", "response": "r"}) + "\n"
        + json.dumps({"example_id": "ex_1", "response": "r"}) + "\n"
    )
    assert _load_completed_ids(path) == {"ex_0", "ex_1"}


def test_generate_responses_writes_checkpoint(tmp_path):
    examples = _make_examples(3)
    model, tokenizer = _make_fake_model_and_tokenizer()
    checkpoint = tmp_path / "responses.jsonl"

    results = generate_responses(
        model,
        tokenizer,
        examples,
        max_new_tokens=8,
        batch_size=2,
        temperature=0.0,
        checkpoint_path=checkpoint,
        checkpoint_every=10,
    )

    assert len(results) == 3
    assert checkpoint.exists()
    lines = [json.loads(l) for l in checkpoint.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    assert {r["example_id"] for r in lines} == {"ex_0", "ex_1", "ex_2"}


def test_generate_responses_skips_completed_ids(tmp_path):
    examples = _make_examples(4)
    checkpoint = tmp_path / "responses.jsonl"

    # Pre-populate checkpoint with first two completed.
    existing = [
        {"example_id": "ex_0", "prompt": "p", "response": "r", "source": "advbench",
         "source_label": "harmful", "label": 1},
        {"example_id": "ex_1", "prompt": "p", "response": "r", "source": "advbench",
         "source_label": "harmful", "label": 1},
    ]
    checkpoint.write_text("\n".join(json.dumps(r) for r in existing) + "\n")

    model, tokenizer = _make_fake_model_and_tokenizer()
    results = generate_responses(
        model,
        tokenizer,
        examples,
        max_new_tokens=8,
        batch_size=2,
        temperature=0.0,
        checkpoint_path=checkpoint,
    )

    assert len(results) == 4
    ids = {r["example_id"] for r in results}
    assert ids == {"ex_0", "ex_1", "ex_2", "ex_3"}


def test_generate_responses_output_record_fields(tmp_path):
    examples = _make_examples(1)
    model, tokenizer = _make_fake_model_and_tokenizer()
    checkpoint = tmp_path / "responses.jsonl"

    results = generate_responses(
        model,
        tokenizer,
        examples,
        max_new_tokens=8,
        batch_size=1,
        temperature=0.0,
        checkpoint_path=checkpoint,
    )

    record = results[0]
    assert "example_id" in record
    assert "prompt" in record
    assert "response" in record
    assert "source" in record
    assert "source_label" in record
    assert "label" in record
