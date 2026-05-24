import json
from collections import Counter

from agguardrails.wildjailbreak import (
    _dataset_rows,
    build_wildjailbreak_contract,
    normalize_wildjailbreak_row,
    write_wildjailbreak_contract,
)

CONFIG_PATH = "configs/ccpp/gemma2_9b_it_wildjailbreak.yaml"


def _config(tmp_path=None):
    config = {
        "model": {"id": "google/gemma-2-9b-it", "revision": None},
        "dataset": {
            "id": "allenai/wildjailbreak",
            "revision": "test-revision",
            "name": "train",
            "split": "train",
            "prompt_columns": {"vanilla": "vanilla", "adversarial": "adversarial"},
            "completion_column": "completion",
            "tactics_column": "tactics",
            "data_type_column": "data_type",
            "required_data_types": [
                "vanilla_harmful",
                "vanilla_benign",
                "adversarial_harmful",
                "adversarial_benign",
            ],
        },
        "sampling": {
            "seed": 123,
            "per_data_type": {
                "vanilla_harmful": 20,
                "vanilla_benign": 20,
                "adversarial_harmful": 5,
                "adversarial_benign": 5,
            },
        },
        "splits": {
            "vanilla": {"train": 0.70, "val": 0.15, "test": 0.15},
            "adversarial_split": "transfer",
        },
    }
    if tmp_path is not None:
        config["outputs"] = {
            "data_path": str(tmp_path / "wildjailbreak.jsonl"),
            "metadata_path": str(tmp_path / "wildjailbreak.metadata.json"),
        }
    return config


def _row(data_type, index):
    is_adversarial = data_type.startswith("adversarial")
    return {
        "id": f"{data_type}-{index}",
        "vanilla": f"vanilla prompt {data_type} {index}",
        "adversarial": (
            f"adversarial prompt {data_type} {index}" if is_adversarial else ""
        ),
        "completion": f"completion {index}",
        "tactics": '["roleplay", "obfuscation"]' if is_adversarial else "",
        "data_type": data_type,
    }


def _rows():
    data_types = [
        "vanilla_harmful",
        "vanilla_benign",
        "adversarial_harmful",
        "adversarial_benign",
    ]
    return [_row(data_type, index) for data_type in data_types for index in range(24)]


def test_all_four_data_types_normalize_correctly():
    config = _config()
    expected = {
        "vanilla_harmful": ("vanilla", 1, []),
        "vanilla_benign": ("vanilla", 0, []),
        "adversarial_harmful": ("adversarial", 1, ["roleplay", "obfuscation"]),
        "adversarial_benign": ("adversarial", 0, ["roleplay", "obfuscation"]),
    }

    for index, (data_type, (family, label, tactics)) in enumerate(expected.items()):
        example = normalize_wildjailbreak_row(
            _row(data_type, index),
            row_index=index,
            config=config,
        )

        assert example.data_type == data_type
        assert example.source_family == family
        assert example.label == label
        assert example.tactics == tactics
        assert example.metadata["original_data_type"] == data_type
        assert example.metadata["upstream_dataset_revision"] == "test-revision"
        assert example.prompt.startswith(family)


def test_adversarial_benign_falls_back_to_vanilla_when_adversarial_is_empty():
    row = _row("adversarial_benign", 0)
    row["adversarial"] = ""

    example = normalize_wildjailbreak_row(row, row_index=0, config=_config())

    assert example.prompt.startswith("vanilla prompt")
    assert example.metadata["prompt_source_column"] == "vanilla"
    assert example.metadata["prompt_fallback_from"] == "adversarial"


def test_adversarial_benign_is_included_in_transfer_split():
    examples, metadata = build_wildjailbreak_contract(
        _rows(),
        config=_config(),
        config_path=CONFIG_PATH,
    )

    transfer_types = {
        example.data_type for example in examples if example.split == "transfer"
    }

    assert "adversarial_benign" in transfer_types
    assert (
        metadata["sampling"]["selected_counts_by_data_type"]["adversarial_benign"]
        == 5
    )
    assert metadata["splits"]["counts_by_split_and_data_type"][
        "transfer:adversarial_benign"
    ] == 5


def test_splits_are_deterministic_and_balanced():
    config = _config()
    first, _ = build_wildjailbreak_contract(
        _rows(), config=config, config_path=CONFIG_PATH
    )
    second, _ = build_wildjailbreak_contract(
        _rows(), config=config, config_path=CONFIG_PATH
    )

    assert [example.example_id for example in first] == [
        example.example_id for example in second
    ]

    counts = Counter((example.split, example.label) for example in first)
    assert counts[("train", 1)] == 14
    assert counts[("train", 0)] == 14
    assert counts[("val", 1)] == 3
    assert counts[("val", 0)] == 3
    assert counts[("test", 1)] == 3
    assert counts[("test", 0)] == 3
    assert counts[("transfer", 1)] == 5
    assert counts[("transfer", 0)] == 5


def test_metadata_is_written_with_config_seed_dataset_info_and_counts(tmp_path):
    config = _config(tmp_path)
    examples, metadata = build_wildjailbreak_contract(
        _rows(),
        config=config,
        config_path=CONFIG_PATH,
    )

    write_wildjailbreak_contract(
        examples,
        metadata,
        data_path=config["outputs"]["data_path"],
        metadata_path=config["outputs"]["metadata_path"],
    )

    with open(config["outputs"]["metadata_path"], encoding="utf-8") as handle:
        written_metadata = json.load(handle)
    with open(config["outputs"]["data_path"], encoding="utf-8") as handle:
        written_rows = [json.loads(line) for line in handle]

    assert written_metadata["config_path"] == CONFIG_PATH
    assert written_metadata["seed"] == 123
    assert written_metadata["dataset"]["id"] == "allenai/wildjailbreak"
    assert written_metadata["dataset"]["revision"] == "test-revision"
    assert written_metadata["dataset"]["name"] == "train"
    assert written_metadata["dataset"]["split"] == "train"
    assert written_metadata["sampling"]["selected_counts_by_data_type"] == {
        "adversarial_benign": 5,
        "adversarial_harmful": 5,
        "vanilla_benign": 20,
        "vanilla_harmful": 20,
    }
    assert written_metadata["splits"]["counts_by_split"] == {
        "test": 6,
        "train": 28,
        "transfer": 10,
        "val": 6,
    }
    assert written_rows[0]["metadata"]["split_seed"] == 123
    assert written_rows[0]["row_id"]


def test_loaded_dataset_dict_uses_configured_split():
    rows = _dataset_rows(
        {
            "train": [{"data_type": "vanilla_harmful", "vanilla": "train row"}],
            "test": [{"data_type": "vanilla_harmful", "vanilla": "test row"}],
        },
        split_name="train",
    )

    assert rows == [{"data_type": "vanilla_harmful", "vanilla": "train row"}]
