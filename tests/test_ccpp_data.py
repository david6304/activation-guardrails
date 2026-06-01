from __future__ import annotations

import pytest

from agguardrails.ccpp_data import (
    DatasetGateError,
    apply_grouped_split,
    dataset_metadata,
    normalize_clearharm_row,
    normalize_record,
    normalize_wildchat_negative_row,
    validate_dataset_gates,
    validate_no_group_leakage,
    validate_positive_class_gate,
)


def test_positive_gate_rejects_refusal_dominated_positives() -> None:
    examples = [
        normalize_record(
            {
                "example_id": "p1",
                "group_id": "g1",
                "split": "train",
                "label": 1,
                "domain": "cbrn",
                "source_dataset": "fixture",
                "source_subset": "positive",
                "user_text": "unsafe request",
                "assistant_text": "I cannot assist with harmful instructions.",
                "completion_source": "public",
            }
        )
    ]

    with pytest.raises(DatasetGateError, match="refusal-like positives"):
        validate_positive_class_gate(examples)


def test_positive_gate_rejects_short_prefill_positives() -> None:
    examples = [
        normalize_record(
            {
                "example_id": "p1",
                "group_id": "g1",
                "split": "train",
                "label": 1,
                "domain": "cbrn",
                "source_dataset": "fixture",
                "source_subset": "positive",
                "user_text": "unsafe request",
                "assistant_text": "Sure, here",
                "completion_source": "public",
            }
        )
    ]

    with pytest.raises(DatasetGateError, match="too short"):
        validate_positive_class_gate(examples)


def test_dataset_gates_accept_non_refusal_positive_and_cbrn_negative() -> None:
    examples = [
        normalize_record(
            {
                "example_id": "p1",
                "group_id": "gp",
                "split": "train",
                "label": 1,
                "domain": "cbrn",
                "source_dataset": "fixture",
                "source_subset": "positive",
                "user_text": "unsafe request",
                "assistant_text": "A direct non-refusal completion placeholder.",
                "completion_source": "public",
            }
        ),
        normalize_record(
            {
                "example_id": "n1",
                "group_id": "gn",
                "split": "val",
                "label": 0,
                "domain": "science",
                "source_dataset": "fixture",
                "source_subset": "hard_negative",
                "user_text": "Explain safe decontamination after a chemistry spill.",
                "assistant_text": "Use appropriate PPE and follow the lab protocol.",
                "completion_source": "public",
                "faithfulness_tags": ["hard_negative"],
            }
        ),
    ]

    validate_dataset_gates(examples)
    metadata = dataset_metadata(examples, config_path="config.yaml")
    assert metadata["positive_non_refusal_fraction"] == 1.0
    assert metadata["label_counts"] == {1: 1, 0: 1}


def test_grouped_split_keeps_group_ids_together() -> None:
    examples = [
        normalize_record(
            {
                "example_id": f"ex-{index}",
                "group_id": f"group-{index // 2}",
                "split": "unassigned",
                "label": index % 2,
                "domain": "science",
                "source_dataset": "fixture",
                "source_subset": "split",
                "user_text": f"Prompt {index}",
                "assistant_text": "Completion",
                "completion_source": "public",
            }
        )
        for index in range(12)
    ]

    assigned = apply_grouped_split(
        examples,
        seed=7,
        train_fraction=0.5,
        val_fraction=0.25,
        test_fraction=0.25,
    )

    validate_no_group_leakage(assigned)
    splits_by_group = {}
    for example in assigned:
        splits_by_group.setdefault(example.group_id, example.split)
        assert splits_by_group[example.group_id] == example.split
    assert {example.split for example in assigned} == {"train", "val", "test"}


def test_group_leakage_gate_rejects_split_overlap() -> None:
    examples = [
        normalize_record(
            {
                "example_id": "a",
                "group_id": "same",
                "split": "train",
                "label": 0,
                "domain": "science",
                "source_dataset": "fixture",
                "source_subset": "leak",
                "user_text": "Prompt A",
                "assistant_text": "Completion",
                "completion_source": "public",
            }
        ),
        normalize_record(
            {
                "example_id": "b",
                "group_id": "same",
                "split": "test",
                "label": 1,
                "domain": "science",
                "source_dataset": "fixture",
                "source_subset": "leak",
                "user_text": "Prompt B",
                "assistant_text": "Completion",
                "completion_source": "public",
            }
        ),
    ]

    with pytest.raises(DatasetGateError, match="multiple splits"):
        validate_no_group_leakage(examples)


def test_normalize_clearharm_row_marks_proxy_prefill_candidate() -> None:
    example = normalize_clearharm_row(
        {
            "clf_label": 1,
            "content": ["How would someone misuse a nuclear isotope?"],
            "proxy_gen_target": "Sure, here",
            "proxy_clf_label": 0,
        },
        row_index=3,
        source_subset="pos",
        source_split="train",
    )

    assert example is not None
    assert example.label == 1
    assert example.domain == "cbrn"
    assert "clearharm_proxy_gen_target" in example.faithfulness_tags
    assert "positive_prefill_only" in example.faithfulness_tags


def test_normalize_wildchat_negative_row_extracts_safe_cbrn_exchange() -> None:
    example = normalize_wildchat_negative_row(
        {
            "conversation_id": "conv-1",
            "model": "gpt-4",
            "turn": 1,
            "language": "English",
            "toxic": False,
            "redacted": False,
            "openai_moderation": [{"flagged": False, "categories": {}}],
            "conversation": [
                {
                    "role": "user",
                    "content": "Explain safe chemistry lab decontamination.",
                },
                {
                    "role": "assistant",
                    "content": "Use PPE and follow the published lab protocol.",
                },
            ],
        },
        row_index=0,
    )

    assert example is not None
    assert example.label == 0
    assert example.source_dataset == "allenai/WildChat"
    assert "hard_negative" in example.faithfulness_tags


def test_normalize_wildchat_negative_row_rejects_flagged_rows() -> None:
    example = normalize_wildchat_negative_row(
        {
            "conversation_id": "conv-2",
            "toxic": False,
            "redacted": False,
            "openai_moderation": [{"flagged": True, "categories": {}}],
            "conversation": [
                {"role": "user", "content": "Explain biology."},
                {"role": "assistant", "content": "Benign answer."},
            ],
        },
        row_index=0,
    )

    assert example is None
