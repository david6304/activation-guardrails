"""Smoke tests for main pipeline Python entrypoints."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from agguardrails.data import PromptExample, read_prompt_dataset
from agguardrails.eval import BinaryEvalResult
from agguardrails.features import ActivationDataset
from agguardrails.io import read_jsonl, read_jsonl as _read_jsonl, write_jsonl
from agguardrails.probes import ProbeResult
from scripts.main import (
    build_dataset as build_dataset_script,
    encode_sae_features as encode_sae_features_script,
    extract_activations as extract_activations_script,
    train_activation_probes as train_activation_probes_script,
    train_text_baseline as train_text_baseline_script,
)


def _write_main_config(path: Path, wildjailbreak_path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 123,
                "model": {"name": "google/gemma-2-9b-it", "dtype": "bfloat16"},
                "data": {
                    "wildjailbreak_path": str(wildjailbreak_path),
                    "n_vanilla_harmful": 8,
                    "n_vanilla_benign": 8,
                    "n_adversarial": 5,
                    "splits": {"train": 0.5, "val": 0.25, "test": 0.25},
                },
                "features": {"layers": [9, 20], "batch_size": 4, "max_length": 128},
                "probe": {"C": 1.0, "max_iter": 200},
                "baseline": {"max_features": 100, "C": 0.5},
                "eval": {"target_fpr": 0.01},
                "sae": {
                    "layers": [9, 20],
                    "width": 16384,
                    "release": "gemma-scope-9b-it-res",
                },
            }
        )
    )


def _wildjailbreak_records() -> list[dict[str, str]]:
    return (
        [
            {
                "id": f"vh{i}",
                "prompt": f"harmful {i}",
                "data_type": "vanilla_harmful",
                "source": "wildjailbreak",
            }
            for i in range(10)
        ]
        + [
            {
                "id": f"vb{i}",
                "prompt": f"benign {i}",
                "data_type": "vanilla_benign",
                "source": "wildjailbreak",
            }
            for i in range(10)
        ]
        + [
            {
                "id": f"ah{i}",
                "prompt": f"adversarial {i}",
                "data_type": "adversarial_harmful",
                "source": "wildjailbreak",
            }
            for i in range(7)
        ]
    )


def test_build_dataset_script_writes_vanilla_and_adversarial_outputs(tmp_path, monkeypatch):
    raw_path = tmp_path / "wildjailbreak.jsonl"
    config_path = tmp_path / "main.yaml"
    output_path = tmp_path / "main_prompts.jsonl"
    adversarial_output = tmp_path / "main_adversarial.jsonl"

    write_jsonl(raw_path, _wildjailbreak_records())
    _write_main_config(config_path, raw_path)

    monkeypatch.setattr(
        build_dataset_script,
        "parse_args",
        lambda: Namespace(
            config=str(config_path),
            output=str(output_path),
            adversarial_output=str(adversarial_output),
        ),
    )

    build_dataset_script.main()

    dataset = read_prompt_dataset(output_path)
    adversarial = read_prompt_dataset(adversarial_output)
    dataset_meta = _read_jsonl(output_path)

    assert len(dataset) == 16
    assert len(adversarial) == 5
    assert {example.split for example in dataset} == {"train", "val", "test"}
    assert all(example.label == 1 for example in adversarial)
    assert all(example.split == "test" for example in adversarial)
    assert {row["source"] for row in dataset_meta} == {"wildjailbreak"}
    assert output_path.with_suffix(".metadata.json").exists()
    assert adversarial_output.with_suffix(".metadata.json").exists()


def test_extract_activations_script_wires_requested_splits_and_adversarial(monkeypatch, tmp_path):
    config_path = tmp_path / "main.yaml"
    _write_main_config(config_path, tmp_path / "unused.jsonl")

    vanilla_dataset = [
        PromptExample("train-1", "prompt train", 1, "train", "wildjailbreak", "1", "vanilla_harmful"),
        PromptExample("test-1", "prompt test", 0, "test", "wildjailbreak", "2", "vanilla_benign"),
    ]
    adversarial_dataset = [
        PromptExample("adv-1", "prompt adv", 1, "test", "wildjailbreak", "3", "adversarial_harmful")
    ]
    calls: list[tuple[str, int]] = []
    saved: list[str] = []

    fake_model = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=41))

    def fake_read_prompt_dataset(path):
        if path == "adv.jsonl":
            return adversarial_dataset
        return vanilla_dataset

    def fake_split_prompt_dataset(dataset):
        assert dataset == vanilla_dataset
        return {"train": [vanilla_dataset[0]], "val": [], "test": [vanilla_dataset[1]]}

    def fake_extract_last_token_hidden_states(**kwargs):
        example_ids = [example.example_id for example in kwargs["examples"]]
        calls.append((kwargs.get("split_name", "unknown"), len(example_ids)))
        return ActivationDataset(
            features_by_layer={9: np.zeros((len(example_ids), 2), dtype=np.float32)},
            labels=np.zeros(len(example_ids), dtype=np.int64),
            example_ids=example_ids,
        )

    monkeypatch.setattr(
        extract_activations_script,
        "parse_args",
        lambda: Namespace(
            config=str(config_path),
            dataset="main.jsonl",
            adversarial_dataset="adv.jsonl",
            output_dir=str(tmp_path / "acts"),
            splits=["train", "test"],
        ),
    )
    monkeypatch.setattr(extract_activations_script, "load_model_and_tokenizer", lambda *args, **kwargs: (fake_model, object()))
    monkeypatch.setattr(extract_activations_script, "validate_layer_indices", lambda layers, model: None)
    monkeypatch.setattr(extract_activations_script, "read_prompt_dataset", fake_read_prompt_dataset)
    monkeypatch.setattr(extract_activations_script, "split_prompt_dataset", fake_split_prompt_dataset)
    monkeypatch.setattr(extract_activations_script, "extract_last_token_hidden_states", fake_extract_last_token_hidden_states)
    monkeypatch.setattr(
        extract_activations_script,
        "save_activation_dataset",
        lambda dataset, output_dir, split, config_path, model_name: saved.append(split),
    )

    extract_activations_script.main()

    assert saved == ["train", "test", "adversarial"]


def test_train_text_baseline_script_saves_pipeline_and_metrics(monkeypatch, tmp_path):
    config_path = tmp_path / "main.yaml"
    _write_main_config(config_path, tmp_path / "unused.jsonl")
    dataset = [
        PromptExample("train-1", "train harmful", 1, "train", "wildjailbreak", "1", "vanilla_harmful"),
        PromptExample("val-1", "val benign", 0, "val", "wildjailbreak", "2", "vanilla_benign"),
        PromptExample("test-1", "test harmful", 1, "test", "wildjailbreak", "3", "vanilla_harmful"),
    ]
    adversarial_dataset = [
        PromptExample("adv-1", "adv harmful", 1, "test", "wildjailbreak", "4", "adversarial_harmful")
    ]
    captured: dict[str, object] = {}

    val_result = BinaryEvalResult(0.8, 0.7, 0.0, 0.6, 1)
    test_result = BinaryEvalResult(0.75, 0.7, 0.0, 0.5, 1)
    fake_pipeline = SimpleNamespace(
        predict_proba=lambda texts: np.array([[0.1, 0.9] for _ in texts], dtype=np.float64),
    )
    fake_artifacts = SimpleNamespace(
        pipeline=fake_pipeline,
        val_result=val_result,
        test_result=test_result,
    )

    monkeypatch.setattr(
        train_text_baseline_script,
        "parse_args",
        lambda: Namespace(
            config=str(config_path),
            dataset="main.jsonl",
            adversarial_dataset="adv.jsonl",
            output_dir=str(tmp_path / "models"),
            metrics_dir=str(tmp_path / "metrics"),
        ),
    )
    monkeypatch.setattr(
        train_text_baseline_script,
        "read_prompt_dataset",
        lambda path: adversarial_dataset if path == "adv.jsonl" else dataset,
    )
    monkeypatch.setattr(
        train_text_baseline_script,
        "split_prompt_dataset",
        lambda ds: {"train": [ds[0]], "val": [ds[1]], "test": [ds[2]]},
    )
    monkeypatch.setattr(train_text_baseline_script, "fit_text_baseline", lambda **kwargs: fake_artifacts)
    monkeypatch.setattr(
        train_text_baseline_script,
        "save_artifact",
        lambda obj, path: Path(path).with_suffix(".joblib"),
    )
    monkeypatch.setattr(
        train_text_baseline_script,
        "save_metadata",
        lambda path, **kwargs: captured.setdefault("metadata", kwargs) or Path(path),
    )

    train_text_baseline_script.main()

    metadata = captured["metadata"]
    assert metadata["config_path"] == str(config_path)
    assert metadata["dataset_path"] == "main.jsonl"
    assert metadata["val_metrics"]["split"] == "val"
    assert metadata["test_metrics"]["split"] == "test"
    assert metadata["adversarial_dataset_path"] == "adv.jsonl"
    assert metadata["adversarial_metrics"]["split"] == "adversarial"
    assert metadata["adversarial_metrics"]["tpr_at_threshold"] == 1.0


def test_train_activation_probes_script_sweeps_layers_and_saves_metrics(monkeypatch, tmp_path):
    config_path = tmp_path / "main.yaml"
    _write_main_config(config_path, tmp_path / "unused.jsonl")
    captured: dict[str, object] = {"layers": [], "saved_paths": []}
    acts_dir = tmp_path / "acts"
    acts_dir.mkdir()
    (acts_dir / "adversarial_labels.npz").write_bytes(b"placeholder")

    train = ActivationDataset(
        features_by_layer={9: np.ones((2, 2), dtype=np.float32), 20: np.ones((2, 2), dtype=np.float32)},
        labels=np.array([0, 1], dtype=np.int64),
        example_ids=["a", "b"],
    )
    val = ActivationDataset(
        features_by_layer={9: np.ones((2, 2), dtype=np.float32), 20: np.ones((2, 2), dtype=np.float32)},
        labels=np.array([0, 1], dtype=np.int64),
        example_ids=["c", "d"],
    )
    test = ActivationDataset(
        features_by_layer={9: np.ones((2, 2), dtype=np.float32), 20: np.ones((2, 2), dtype=np.float32)},
        labels=np.array([0, 1], dtype=np.int64),
        example_ids=["e", "f"],
    )
    adversarial = ActivationDataset(
        features_by_layer={9: np.ones((2, 2), dtype=np.float32), 20: np.ones((2, 2), dtype=np.float32)},
        labels=np.array([1, 1], dtype=np.int64),
        example_ids=["g", "h"],
    )

    def fake_load_activation_split(*, input_dir, split, layers):
        assert layers == [9, 20]
        return {"train": train, "val": val, "test": test, "adversarial": adversarial}[split]

    def fake_fit_probe_for_layer(**kwargs):
        layer = kwargs["layer"]
        captured["layers"].append(layer)
        score = 0.6 if layer == 9 else 0.7
        fake_probe = SimpleNamespace(
            predict_proba=lambda x: np.array([[0.1, 0.9] for _ in range(len(x))], dtype=np.float64)
        )
        return ProbeResult(
            layer=layer,
            probe=fake_probe,
            val_result=BinaryEvalResult(score, 0.5, 0.0, score, 1),
            test_result=BinaryEvalResult(score - 0.1, 0.5, 0.0, score - 0.1, 1),
        )

    monkeypatch.setattr(
        train_activation_probes_script,
        "parse_args",
        lambda: Namespace(
            config=str(config_path),
            input_dir=str(acts_dir),
            output_dir=str(tmp_path / "probes"),
            metrics_dir=str(tmp_path / "metrics"),
        ),
    )
    monkeypatch.setattr(train_activation_probes_script, "load_activation_split", fake_load_activation_split)
    monkeypatch.setattr(train_activation_probes_script, "fit_probe_for_layer", fake_fit_probe_for_layer)
    monkeypatch.setattr(
        train_activation_probes_script,
        "save_artifact",
        lambda obj, path: captured["saved_paths"].append(str(path)),
    )
    monkeypatch.setattr(
        train_activation_probes_script,
        "save_metadata",
        lambda path, **kwargs: captured.setdefault("metadata", kwargs),
    )

    train_activation_probes_script.main()

    assert captured["layers"] == [9, 20]
    assert any("layer_9_probe" in path for path in captured["saved_paths"])
    assert any("layer_20_probe" in path for path in captured["saved_paths"])
    assert captured["metadata"]["best_layer"] == 20
    assert set(captured["metadata"]["per_layer"].keys()) == {"9", "20"}
    assert captured["metadata"]["best_adversarial_metrics"]["split"] == "adversarial"
    assert captured["metadata"]["per_layer"]["9"]["adversarial"]["tpr_at_threshold"] == 1.0


def test_encode_sae_features_script_encodes_requested_splits(monkeypatch, tmp_path):
    config_path = tmp_path / "main.yaml"
    _write_main_config(config_path, tmp_path / "unused.jsonl")
    captured: dict[str, object] = {"saved": [], "loaded": []}

    dataset = ActivationDataset(
        features_by_layer={
            9: np.ones((2, 3), dtype=np.float32),
            20: np.ones((2, 3), dtype=np.float32) * 2,
        },
        labels=np.array([0, 1], dtype=np.int64),
        example_ids=["a", "b"],
    )

    monkeypatch.setattr(
        encode_sae_features_script,
        "parse_args",
        lambda: Namespace(
            config=str(config_path),
            input_dir="acts",
            output_dir=str(tmp_path / "sae"),
            splits=["train"],
            batch_size=32,
            device="cpu",
            dtype="float32",
        ),
    )
    monkeypatch.setattr(
        encode_sae_features_script,
        "load_activation_split",
        lambda input_dir, split, layers: dataset,
    )

    def fake_load_pretrained_sae(**kwargs):
        captured["loaded"].append((kwargs["release"], kwargs["sae_id"]))
        return object()

    monkeypatch.setattr(
        encode_sae_features_script,
        "load_pretrained_sae",
        fake_load_pretrained_sae,
    )
    monkeypatch.setattr(
        encode_sae_features_script,
        "encode_with_sae",
        lambda sae, activations, batch_size: activations + 3,
    )
    monkeypatch.setattr(
        encode_sae_features_script,
        "save_artifact",
        lambda obj, path: captured["saved"].append(str(path)) or Path(path).with_suffix(".npz"),
    )
    monkeypatch.setattr(
        encode_sae_features_script,
        "save_metadata",
        lambda path, **kwargs: captured.setdefault("metadata", kwargs),
    )

    encode_sae_features_script.main()

    assert captured["loaded"] == [
        ("gemma-scope-9b-it-res", "layer_9/width_16k/canonical"),
        ("gemma-scope-9b-it-res", "layer_20/width_16k/canonical"),
    ]
    assert any("train_layer_9_sae_features" in path for path in captured["saved"])
    assert any("train_layer_20_sae_features" in path for path in captured["saved"])
    assert captured["metadata"]["split"] == "train"
    assert captured["metadata"]["layers"] == [9, 20]
