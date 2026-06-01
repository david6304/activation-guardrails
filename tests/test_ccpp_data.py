from __future__ import annotations

import pytest

from agguardrails.ccpp_data import (
    DatasetGateError,
    apply_grouped_split,
    dataset_metadata,
    normalize_record,
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

