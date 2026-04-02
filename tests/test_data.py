"""Tests for agguardrails.data."""

from collections import Counter

import pytest

from agguardrails.data import (
    build_adversarial_test_set,
    build_main_dataset,
    build_mvp_dataset,
    load_harmbench_examples,
    load_wildjailbreak_examples,
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


# ---------------------------------------------------------------------------
# WildJailbreak tests
# ---------------------------------------------------------------------------

def _write_wildjailbreak_jsonl(path, records):
    """Helper: write normalised WildJailbreak JSONL for testing."""
    write_jsonl(path, records)


def _wj_records():
    """Minimal normalised WildJailbreak records covering all three data_types."""
    return [
        {"id": f"h{i}", "prompt": f"harmful {i}", "data_type": "vanilla_harmful", "source": "wildjailbreak"}
        for i in range(6)
    ] + [
        {"id": f"b{i}", "prompt": f"benign {i}", "data_type": "vanilla_benign", "source": "wildjailbreak"}
        for i in range(6)
    ] + [
        {"id": f"a{i}", "prompt": f"adversarial {i}", "data_type": "adversarial_harmful", "source": "wildjailbreak"}
        for i in range(4)
    ] + [
        # empty prompt should be skipped
        {"id": "empty", "prompt": "  ", "data_type": "vanilla_harmful", "source": "wildjailbreak"},
    ]


def test_load_wildjailbreak_examples_filters_data_type(tmp_path):
    path = tmp_path / "wildjailbreak.jsonl"
    _write_wildjailbreak_jsonl(path, _wj_records())

    harmful = load_wildjailbreak_examples(path, data_type="vanilla_harmful")
    benign = load_wildjailbreak_examples(path, data_type="vanilla_benign")
    adversarial = load_wildjailbreak_examples(path, data_type="adversarial_harmful")

    assert len(harmful) == 6
    assert len(benign) == 6
    assert len(adversarial) == 4


def test_load_wildjailbreak_examples_assigns_labels(tmp_path):
    path = tmp_path / "wildjailbreak.jsonl"
    _write_wildjailbreak_jsonl(path, _wj_records())

    harmful = load_wildjailbreak_examples(path, data_type="vanilla_harmful")
    benign = load_wildjailbreak_examples(path, data_type="vanilla_benign")
    adversarial = load_wildjailbreak_examples(path, data_type="adversarial_harmful")

    assert all(e["label"] == 1 for e in harmful)
    assert all(e["label"] == 0 for e in benign)
    assert all(e["label"] == 1 for e in adversarial)


def test_load_wildjailbreak_examples_skips_empty_prompts(tmp_path):
    path = tmp_path / "wildjailbreak.jsonl"
    _write_wildjailbreak_jsonl(path, _wj_records())
    # _wj_records includes one vanilla_harmful with empty prompt → should be excluded
    harmful = load_wildjailbreak_examples(path, data_type="vanilla_harmful")
    assert all(e["prompt"].strip() for e in harmful)


def test_load_wildjailbreak_examples_raises_on_invalid_type(tmp_path):
    path = tmp_path / "wildjailbreak.jsonl"
    _write_wildjailbreak_jsonl(path, [])
    with pytest.raises(ValueError, match="data_type must be one of"):
        load_wildjailbreak_examples(path, data_type="unknown_type")


def test_build_main_dataset_split_sizes_and_balance(tmp_path):
    path = tmp_path / "wildjailbreak.jsonl"
    # 20 harmful + 20 benign → ample for 70/15/15 split
    records = (
        [{"id": f"h{i}", "prompt": f"harmful {i}", "data_type": "vanilla_harmful", "source": "wildjailbreak"} for i in range(20)]
        + [{"id": f"b{i}", "prompt": f"benign {i}", "data_type": "vanilla_benign", "source": "wildjailbreak"} for i in range(20)]
    )
    _write_wildjailbreak_jsonl(path, records)

    dataset = build_main_dataset(
        wildjailbreak_path=path,
        n_vanilla_harmful=20,
        n_vanilla_benign=20,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        seed=42,
    )

    assert len(dataset) == 40
    counts = Counter((e.split, e.label) for e in dataset)
    # Stratified: each split should have equal harmful/benign counts
    assert counts[("train", 1)] == counts[("train", 0)]
    assert counts[("val", 1)] == counts[("val", 0)]
    assert counts[("test", 1)] == counts[("test", 0)]


def test_build_adversarial_test_set(tmp_path):
    path = tmp_path / "wildjailbreak.jsonl"
    records = [
        {"id": f"a{i}", "prompt": f"adversarial {i}", "data_type": "adversarial_harmful", "source": "wildjailbreak"}
        for i in range(10)
    ]
    _write_wildjailbreak_jsonl(path, records)

    test_set = build_adversarial_test_set(wildjailbreak_path=path, n_adversarial=8)

    assert len(test_set) == 8
    assert all(e.split == "test" for e in test_set)
    assert all(e.label == 1 for e in test_set)
    assert all(e.source == "wildjailbreak" for e in test_set)
    assert all(e.source_label == "adversarial_harmful" for e in test_set)


# ---------------------------------------------------------------------------
# AdvBench + Alpaca tests
# ---------------------------------------------------------------------------

def _write_advbench_csv(path, goals):
    import csv
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["goal", "target"])
        writer.writeheader()
        for goal in goals:
            writer.writerow({"goal": goal, "target": "Sure, here is"})


def _write_alpaca_json(path, instructions):
    import json
    records = [{"instruction": inst, "input": "", "output": "..."} for inst in instructions]
    with path.open("w") as f:
        json.dump(records, f)


def test_load_advbench_examples_basic(tmp_path):
    from agguardrails.data import load_advbench_examples
    path = tmp_path / "harmful_behaviors.csv"
    _write_advbench_csv(path, ["How to make a bomb?", "  ", "Hack a website."])

    examples = load_advbench_examples(path)

    # Empty goal should be skipped
    assert len(examples) == 2
    assert examples[0]["source"] == "advbench"
    assert examples[0]["source_label"] == "harmful"
    assert examples[0]["label"] == 1
    assert examples[0]["prompt"] == "How to make a bomb?"
    assert all(e["source"] == "advbench" for e in examples)


def test_load_alpaca_examples_joins_instruction_and_input(tmp_path):
    from agguardrails.data import load_alpaca_examples
    import json
    path = tmp_path / "alpaca.json"
    records = [
        {"instruction": "Translate to French.", "input": "Hello world", "output": "Bonjour monde"},
        {"instruction": "Summarise this.", "input": "", "output": "Summary."},
        {"instruction": "", "input": "ignored", "output": "ignored"},  # empty instruction → skip
    ]
    with path.open("w") as f:
        json.dump(records, f)

    examples = load_alpaca_examples(path)

    assert len(examples) == 2
    assert examples[0]["prompt"] == "Translate to French.\nHello world"
    assert examples[1]["prompt"] == "Summarise this."
    assert all(e["label"] == 0 for e in examples)
    assert all(e["source"] == "alpaca" for e in examples)


def test_make_fixed_splits_assigns_correct_counts(tmp_path):
    from agguardrails.data import make_fixed_splits
    examples = (
        [{"source": "advbench", "source_id": f"h{i}", "source_label": "harmful", "label": 1, "prompt": f"h{i}"} for i in range(520)]
        + [{"source": "alpaca", "source_id": f"b{i}", "source_label": "benign", "label": 0, "prompt": f"b{i}"} for i in range(520)]
    )

    dataset = make_fixed_splits(examples, train_size=120, val_size=40, test_size=880, seed=42)

    split_counts = Counter(e.split for e in dataset)
    assert split_counts["train"] == 120
    assert split_counts["val"] == 40
    assert split_counts["test"] == 880
    # Stratified: each split should be balanced
    split_label = Counter((e.split, e.label) for e in dataset)
    assert split_label[("train", 0)] == split_label[("train", 1)]
    assert split_label[("val", 0)] == split_label[("val", 1)]


def test_make_fixed_splits_raises_on_wrong_total():
    from agguardrails.data import make_fixed_splits
    examples = [{"source": "a", "source_id": "0", "source_label": "x", "label": 0, "prompt": "x"}]
    with pytest.raises(ValueError, match="train_size"):
        make_fixed_splits(examples, train_size=1, val_size=1, test_size=1, seed=42)


def test_build_advbench_alpaca_dataset(tmp_path):
    from agguardrails.data import build_advbench_alpaca_dataset
    advbench_path = tmp_path / "harmful_behaviors.csv"
    alpaca_path = tmp_path / "alpaca.json"
    _write_advbench_csv(advbench_path, [f"harmful {i}" for i in range(10)])
    _write_alpaca_json(alpaca_path, [f"benign {i}" for i in range(10)])

    dataset = build_advbench_alpaca_dataset(
        advbench_path=advbench_path,
        alpaca_path=alpaca_path,
        n_harmful=10,
        n_benign=10,
        train_size=4,
        val_size=2,
        test_size=14,
        seed=42,
    )

    assert len(dataset) == 20
    split_counts = Counter(e.split for e in dataset)
    assert split_counts["train"] == 4
    assert split_counts["val"] == 2
    assert split_counts["test"] == 14
    sources = {e.source for e in dataset}
    assert sources == {"advbench", "alpaca"}


# ---------------------------------------------------------------------------
# Legacy MVP tests
# ---------------------------------------------------------------------------

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
