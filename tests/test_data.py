"""Tests for agguardrails.data."""

from collections import Counter

import pytest

from agguardrails.data import (
    build_mvp_dataset,
    load_harmbench_examples,
    load_xstest_examples,
    make_stratified_splits,
    sample_examples,
    write_prompt_dataset,
)
from agguardrails.io import read_jsonl, write_jsonl


def test_load_harmbench_examples_extracts_prompt(tmp_path):
    path = tmp_path / "harmbench.jsonl"
    write_jsonl(
        path,
        [
            {"content": [" harmful prompt  "]},
            {"content": [""]},
            {"content": []},
        ],
    )

    loaded = load_harmbench_examples(path)

    assert loaded == [
        {
            "source": "harmbench",
            "source_id": "0",
            "source_label": "harmful",
            "label": 1,
            "prompt": "harmful prompt",
        }
    ]


def test_load_xstest_examples_filters_label(tmp_path):
    path = tmp_path / "xstest.jsonl"
    write_jsonl(
        path,
        [
            {"id": 1, "prompt": "safe prompt", "label": "safe"},
            {"id": 2, "prompt": "unsafe prompt", "label": "unsafe"},
        ],
    )

    loaded = load_xstest_examples(path, label="safe")

    assert loaded == [
        {
            "source": "xstest",
            "source_id": "1",
            "source_label": "safe",
            "label": 0,
            "prompt": "safe prompt",
        }
    ]


def test_sample_examples_raises_when_request_exceeds_available():
    with pytest.raises(ValueError, match="Requested 3 examples but only 2 available"):
        sample_examples([{"x": 1}, {"x": 2}], n=3)


def test_make_stratified_splits_preserves_label_balance():
    examples = [
        {"source": "harmbench", "source_id": f"h{i}", "source_label": "harmful", "label": 1, "prompt": f"h{i}"}
        for i in range(20)
    ] + [
        {"source": "xstest", "source_id": f"s{i}", "source_label": "safe", "label": 0, "prompt": f"s{i}"}
        for i in range(20)
    ]

    dataset = make_stratified_splits(
        examples,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        seed=42,
    )

    split_label_counts = Counter((row.split, row.label) for row in dataset)
    assert split_label_counts[("train", 1)] == 14
    assert split_label_counts[("train", 0)] == 14
    assert split_label_counts[("val", 1)] == 3
    assert split_label_counts[("val", 0)] == 3
    assert split_label_counts[("test", 1)] == 3
    assert split_label_counts[("test", 0)] == 3


def test_build_mvp_dataset_and_write_round_trip(tmp_path):
    harmful_path = tmp_path / "harmbench.jsonl"
    benign_path = tmp_path / "xstest.jsonl"

    write_jsonl(
        harmful_path,
        [{"content": [f"harmful {i}"]} for i in range(10)],
    )
    write_jsonl(
        benign_path,
        [{"id": i, "prompt": f"safe {i}", "label": "safe"} for i in range(12)]
        + [{"id": 100 + i, "prompt": f"unsafe {i}", "label": "unsafe"} for i in range(4)],
    )

    dataset = build_mvp_dataset(
        harmful_path=harmful_path,
        benign_path=benign_path,
        benign_label="safe",
        n_harmful=10,
        n_benign=12,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        seed=123,
    )
    output_path = write_prompt_dataset(tmp_path / "mvp_prompts.jsonl", dataset)

    loaded = read_jsonl(output_path)

    assert len(loaded) == 22
    assert {row["source"] for row in loaded} == {"harmbench", "xstest"}
    assert Counter(row["label"] for row in loaded) == Counter({0: 12, 1: 10})
