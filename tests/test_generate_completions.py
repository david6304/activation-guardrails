from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from agguardrails.ccpp_benign import build_benign_prompts
from agguardrails.ccpp_data import (
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    apply_grouped_split,
    normalize_clearharm_prompt_row,
    validate_dataset_gates,
)
from agguardrails.ccpp_generation import (
    COMPLETION_SOURCE,
    DecodingParams,
    MockGenerator,
    build_exchange_record,
    generate_exchanges,
    infer_label,
    read_generation_prompts,
)

MODEL_ID = "local/gemma-3-4b-it-heretic"


def _clearharm_prompt(text: str, row_index: int):
    return normalize_clearharm_prompt_row(
        {"clf_label": 1, "content": [text]},
        row_index=row_index,
        source_subset="rep40",
        source_split="train",
    )


def test_infer_label_from_provenance() -> None:
    benign = build_benign_prompts(target_unique_groups=1, seed=0)[0]
    positive = _clearharm_prompt("synthetic cbrn positive prompt", 0)
    assert infer_label(benign) == LABEL_NEGATIVE
    assert infer_label(positive) == LABEL_POSITIVE


def test_build_exchange_record_is_on_policy() -> None:
    prompt = build_benign_prompts(target_unique_groups=1, seed=0)[0]
    decoding = DecodingParams(backend="mock", seed=3)
    exchange = build_exchange_record(
        prompt,
        "A sufficiently long benign completion about safety and oversight.",
        label=LABEL_NEGATIVE,
        generator_model_id=MODEL_ID,
        protected_model_id=MODEL_ID,
        decoding=decoding,
    )
    assert exchange.label == LABEL_NEGATIVE
    assert exchange.completion_source == COMPLETION_SOURCE
    assert exchange.group_id == prompt.group_id
    assert exchange.metadata["generator_model_id"] == MODEL_ID
    assert exchange.metadata["protected_model_id"] == MODEL_ID
    assert exchange.metadata["decoding"]["seed"] == 3
    assert exchange.metadata["source_prompt_id"] == prompt.prompt_id
    assert "hard_negative" in exchange.faithfulness_tags
    assert "requires_generated_uncensored_completion" not in exchange.faithfulness_tags


def test_mock_generator_is_deterministic_and_nontrivial() -> None:
    gen = MockGenerator(seed=1)
    a = gen.generate("In the context of radon mitigation, explain ALARA.")
    b = gen.generate("In the context of radon mitigation, explain ALARA.")
    assert a == b
    assert len(a) >= 20


def test_generated_exchanges_pass_dataset_gates() -> None:
    decoding = DecodingParams(backend="mock", seed=0)
    generator = MockGenerator(seed=0)

    benign = build_benign_prompts(target_unique_groups=20, seed=0)
    positives = [
        _clearharm_prompt(f"synthetic cbrn chemical synthesis prompt {i}", i)
        for i in range(20)
    ]

    exchanges = generate_exchanges(
        benign + positives,
        generator,
        generator_model_id=MODEL_ID,
        protected_model_id=MODEL_ID,
        decoding=decoding,
    )
    labels = {exchange.label for exchange in exchanges}
    assert labels == {LABEL_NEGATIVE, LABEL_POSITIVE}

    exchanges = apply_grouped_split(
        exchanges, seed=0, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2
    )
    # Should not raise: on-policy, length-balance, hard-negative, group-leakage.
    validate_dataset_gates(
        exchanges,
        on_policy_config={
            "enabled": True,
            "allowed_completion_sources": [COMPLETION_SOURCE],
            "require_single_generator": True,
            "require_on_policy_model": True,
        },
        length_balance_config={
            "enabled": True,
            "max_median_assistant_word_ratio": 2.0,
            "max_length_only_roc_auc": 0.65,
        },
    )


def test_read_generation_prompts_roundtrip() -> None:
    prompts = build_benign_prompts(target_unique_groups=3, seed=0)
    rows = [p.to_json_dict() for p in prompts]
    restored = read_generation_prompts(rows)
    assert [p.prompt_id for p in restored] == [p.prompt_id for p in prompts]
    assert [p.group_id for p in restored] == [p.group_id for p in prompts]


def test_build_exchange_rejects_bad_label() -> None:
    prompt = build_benign_prompts(target_unique_groups=1, seed=0)[0]
    with pytest.raises(ValueError, match="label must be 0 or 1"):
        build_exchange_record(
            prompt,
            "text",
            label=2,
            generator_model_id=MODEL_ID,
            protected_model_id=MODEL_ID,
            decoding=DecodingParams(backend="mock"),
        )


def test_script_mock_run_no_raw_completion_on_stdout(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "exchanges.jsonl"
    prompts = build_benign_prompts(target_unique_groups=8, seed=0)
    manifest.write_text(
        "\n".join(json.dumps(p.to_json_dict()) for p in prompts) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ccpp/generate_completions.py",
            "--manifest",
            str(manifest),
            "--output",
            str(output),
            "--model-id",
            MODEL_ID,
            "--backend",
            "mock",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 8
    assert all(row["completion_source"] == COMPLETION_SOURCE for row in rows)
    assert all(
        row["metadata"]["generator_model_id"]
        == row["metadata"]["protected_model_id"]
        == MODEL_ID
        for row in rows
    )

    summary = json.loads(result.stdout)
    assert summary["num_exchanges"] == 8
    assert summary["on_policy"] is True
    for row in rows:
        assert row["assistant_text"] not in result.stdout
