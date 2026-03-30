"""Tests for the WildJailbreak download/normalisation script."""

from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import pytest

from agguardrails.io import read_jsonl

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "download_wildjailbreak.py"
)
SPEC = importlib.util.spec_from_file_location("download_wildjailbreak", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
download_wildjailbreak = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_wildjailbreak)


def test_normalise_record_uses_vanilla_for_vanilla_types():
    record = {
        "id": "row-1",
        "vanilla": "  benign-looking prompt  ",
        "adversarial": "ignored adversarial text",
        "data_type": "vanilla_benign",
    }

    normalised = download_wildjailbreak.normalise_record(record, idx=0)

    assert normalised == {
        "id": "row-1_vanilla_benign",
        "prompt": "benign-looking prompt",
        "data_type": "vanilla_benign",
        "source": "wildjailbreak",
    }


def test_normalise_record_uses_adversarial_for_adversarial_types():
    record = {
        "id": "row-2",
        "vanilla": "source vanilla prompt",
        "adversarial": "  jailbreak prompt  ",
        "data_type": "adversarial_harmful",
    }

    normalised = download_wildjailbreak.normalise_record(record, idx=1)

    assert normalised == {
        "id": "row-2_adversarial_harmful",
        "prompt": "jailbreak prompt",
        "data_type": "adversarial_harmful",
        "source": "wildjailbreak",
    }


def test_normalise_record_returns_none_for_empty_selected_prompt():
    record = {
        "id": "row-3",
        "vanilla": "kept only for context",
        "adversarial": "   ",
        "data_type": "adversarial_benign",
    }

    assert download_wildjailbreak.normalise_record(record, idx=2) is None


def test_normalise_record_raises_on_missing_data_type():
    with pytest.raises(ValueError, match="missing required field 'data_type'"):
        download_wildjailbreak.normalise_record({"vanilla": "prompt"}, idx=0)


def test_normalise_record_raises_on_unknown_data_type():
    with pytest.raises(ValueError, match="Unsupported WildJailbreak data_type"):
        download_wildjailbreak.normalise_record(
            {"vanilla": "prompt", "data_type": "mystery_type"},
            idx=0,
        )


def test_main_passes_current_dataset_loader_args_and_writes_normalised_rows(
    monkeypatch,
    tmp_path,
):
    output_path = tmp_path / "wildjailbreak.jsonl"
    captured: dict[str, object] = {}

    args = Namespace(
        output=str(output_path),
        hf_dataset="allenai/wildjailbreak",
        config_name="train",
        split="train",
        print_schema_only=False,
    )

    class FakeDataset:
        features = {"vanilla": "string", "adversarial": "string", "data_type": "string"}

        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return list(self)[idx]

        def __iter__(self):
            yield {
                "id": "vh",
                "vanilla": "harmful prompt",
                "adversarial": "",
                "data_type": "vanilla_harmful",
            }
            yield {
                "id": "ah",
                "vanilla": "harmful source",
                "adversarial": "obfuscated harmful prompt",
                "data_type": "adversarial_harmful",
            }

    def fake_load_dataset(*args_, **kwargs):
        captured["args"] = args_
        captured["kwargs"] = kwargs
        return FakeDataset()

    monkeypatch.setattr(download_wildjailbreak, "parse_args", lambda: args)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    download_wildjailbreak.main()

    assert captured["args"] == ("allenai/wildjailbreak", "train")
    assert captured["kwargs"] == {
        "split": "train",
        "delimiter": "\t",
        "keep_default_na": False,
        "trust_remote_code": False,
    }

    assert output_path.exists()
    assert read_jsonl(output_path) == [
        {
            "id": "vh_vanilla_harmful",
            "prompt": "harmful prompt",
            "data_type": "vanilla_harmful",
            "source": "wildjailbreak",
        },
        {
            "id": "ah_adversarial_harmful",
            "prompt": "obfuscated harmful prompt",
            "data_type": "adversarial_harmful",
            "source": "wildjailbreak",
        },
    ]
