"""Smoke tests for main pipeline Python entrypoints."""

from __future__ import annotations

import json
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
    build_refusal_dataset as build_refusal_dataset_script,
    cache_sae_models as cache_sae_models_script,
    encode_sae_features as encode_sae_features_script,
    extract_activations as extract_activations_script,
    generate_responses as generate_responses_script,
    label_refusals as label_refusals_script,
    make_results_table as make_results_table_script,
    train_activation_probes as train_activation_probes_script,
    train_sae_probes as train_sae_probes_script,
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
                    "variant": "average_l0_14",
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
    captured: dict[str, object] = {"saved": [], "loaded": [], "metadata_calls": []}

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
            metrics_dir=str(tmp_path / "metrics"),
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
        lambda path, **kwargs: captured["metadata_calls"].append((str(path), kwargs))
        or Path(path),
    )

    encode_sae_features_script.main()

    assert captured["loaded"] == [
        ("gemma-scope-9b-it-res", "layer_9/width_16k/average_l0_14"),
        ("gemma-scope-9b-it-res", "layer_20/width_16k/average_l0_14"),
    ]
    assert any("train_layer_9_sae_features" in path for path in captured["saved"])
    assert any("train_layer_20_sae_features" in path for path in captured["saved"])
    assert len(captured["metadata_calls"]) == 2
    split_meta = captured["metadata_calls"][0]
    summary_meta = captured["metadata_calls"][1]
    assert split_meta[1]["split"] == "train"
    assert split_meta[1]["layers"] == [9, 20]
    assert split_meta[1]["variant"] == "average_l0_14"
    assert summary_meta[0].endswith("metrics/sae_encoding_summary.json")
    assert summary_meta[1]["layers"] == [9, 20]
    assert summary_meta[1]["variant"] == "average_l0_14"
    assert summary_meta[1]["splits"]["train"]["n_examples"] == 2


def test_cache_sae_models_script_loads_all_configured_saes(monkeypatch, tmp_path):
    config_path = tmp_path / "main.yaml"
    _write_main_config(config_path, tmp_path / "unused.jsonl")
    captured: dict[str, object] = {"loaded": []}

    monkeypatch.setattr(
        cache_sae_models_script,
        "parse_args",
        lambda: Namespace(
            config=str(config_path),
            device="cpu",
            dtype="float32",
        ),
    )
    monkeypatch.setattr(
        cache_sae_models_script,
        "load_pretrained_sae",
        lambda **kwargs: captured["loaded"].append((kwargs["release"], kwargs["sae_id"])),
    )

    cache_sae_models_script.main()

    assert captured["loaded"] == [
        ("gemma-scope-9b-it-res", "layer_9/width_16k/average_l0_14"),
        ("gemma-scope-9b-it-res", "layer_20/width_16k/average_l0_14"),
    ]


def test_train_sae_probes_script_sweeps_sae_layers_and_saves_metrics(monkeypatch, tmp_path):
    config_path = tmp_path / "main.yaml"
    _write_main_config(config_path, tmp_path / "unused.jsonl")
    captured: dict[str, object] = {"layers": [], "saved_paths": []}
    features_dir = tmp_path / "sae"
    features_dir.mkdir()
    (features_dir / "adversarial_labels.npz").write_bytes(b"placeholder")

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

    def fake_load_layer_feature_split(*, input_dir, split, layers, feature_name):
        assert layers == [9, 20]
        assert feature_name == "sae_features"
        return {"train": train, "val": val, "test": test, "adversarial": adversarial}[split]

    def fake_fit_probe_for_layer(**kwargs):
        layer = kwargs["layer"]
        captured["layers"].append(layer)
        score = 0.61 if layer == 9 else 0.74
        fake_probe = SimpleNamespace(
            predict_proba=lambda x: np.array([[0.2, 0.8] for _ in range(len(x))], dtype=np.float64)
        )
        return ProbeResult(
            layer=layer,
            probe=fake_probe,
            val_result=BinaryEvalResult(score, 0.5, 0.0, score, 1),
            test_result=BinaryEvalResult(score - 0.1, 0.5, 0.0, score - 0.1, 1),
        )

    monkeypatch.setattr(
        train_sae_probes_script,
        "parse_args",
        lambda: Namespace(
            config=str(config_path),
            input_dir=str(features_dir),
            output_dir=str(tmp_path / "sae_probes"),
            metrics_dir=str(tmp_path / "metrics"),
        ),
    )
    monkeypatch.setattr(
        train_sae_probes_script,
        "load_layer_feature_split",
        fake_load_layer_feature_split,
    )
    monkeypatch.setattr(train_sae_probes_script, "fit_probe_for_layer", fake_fit_probe_for_layer)
    monkeypatch.setattr(
        train_sae_probes_script,
        "save_artifact",
        lambda obj, path: captured["saved_paths"].append(str(path)),
    )
    monkeypatch.setattr(
        train_sae_probes_script,
        "save_metadata",
        lambda path, **kwargs: captured.setdefault("metadata", kwargs),
    )

    train_sae_probes_script.main()

    assert captured["layers"] == [9, 20]
    assert any("layer_9_sae_probe" in path for path in captured["saved_paths"])
    assert any("layer_20_sae_probe" in path for path in captured["saved_paths"])
    assert captured["metadata"]["best_layer"] == 20
    assert captured["metadata"]["release"] == "gemma-scope-9b-it-res"
    assert captured["metadata"]["width"] == 16384
    assert set(captured["metadata"]["per_layer"].keys()) == {"9", "20"}
    assert captured["metadata"]["best_adversarial_metrics"]["split"] == "adversarial"
    assert captured["metadata"]["per_layer"]["9"]["adversarial"]["tpr_at_threshold"] == 1.0


def test_main_results_table_script_writes_all_available_rows(monkeypatch, tmp_path):
    text_metrics = tmp_path / "text.json"
    probe_metrics = tmp_path / "probe.json"
    sae_metrics = tmp_path / "sae.json"
    output_path = tmp_path / "results.csv"
    captured: dict[str, object] = {}

    text_metrics.write_text(
        json.dumps(
            {
                "val_metrics": {"model_name": "tfidf_lr", "split": "val", "roc_auc": 0.9},
                "test_metrics": {"model_name": "tfidf_lr", "split": "test", "roc_auc": 0.8},
                "adversarial_metrics": {
                    "model_name": "tfidf_lr",
                    "split": "adversarial",
                    "tpr_at_threshold": 0.7,
                },
            }
        )
    )
    probe_metrics.write_text(
        json.dumps(
            {
                "best_layer": 20,
                "per_layer": {
                    "20": {
                        "val": {"model_name": "dense_probe", "split": "val", "layer": 20},
                        "test": {"model_name": "dense_probe", "split": "test", "layer": 20},
                    }
                },
            }
        )
    )
    sae_metrics.write_text(
        json.dumps(
            {
                "best_layer": 9,
                "per_layer": {
                    "9": {
                        "val": {"model_name": "sae_probe", "split": "val", "layer": 9},
                        "test": {"model_name": "sae_probe", "split": "test", "layer": 9},
                        "adversarial": {
                            "model_name": "sae_probe",
                            "split": "adversarial",
                            "layer": 9,
                        },
                    }
                },
            }
        )
    )

    monkeypatch.setattr(
        make_results_table_script,
        "parse_args",
        lambda: Namespace(
            config="configs/main/main.yaml",
            text_metrics=str(text_metrics),
            probe_metrics=str(probe_metrics),
            sae_probe_metrics=str(sae_metrics),
            output=str(output_path),
        ),
    )
    monkeypatch.setattr(
        make_results_table_script,
        "save_metadata",
        lambda path, **kwargs: captured.setdefault("metadata", kwargs),
    )

    make_results_table_script.main()

    assert output_path.exists()
    assert captured["metadata"]["n_rows"] == 8


# ---------------------------------------------------------------------------
# Refusal pipeline script tests
# ---------------------------------------------------------------------------

def _write_refusal_config(path: Path, advbench_path: Path, alpaca_path: Path) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "model": {"name": "google/gemma-2-9b-it", "dtype": "bfloat16"},
                "data": {
                    "dataset": "advbench_alpaca",
                    "advbench_path": str(advbench_path),
                    "alpaca_path": str(alpaca_path),
                    "n_harmful": 4,
                    "n_benign": 4,
                    "train_size": 2,
                    "val_size": 2,
                    "test_size": 4,
                },
                "generation": {
                    "max_new_tokens": 64,
                    "temperature": 0.0,
                    "batch_size": 2,
                    "checkpoint_every": 10,
                },
                "judge": {
                    "model": "self",
                    "max_new_tokens": 16,
                    "temperature": 0.0,
                    "batch_size": 2,
                },
                "features": {"layers": [9], "batch_size": 2, "max_length": 64},
                "probe": {"penalty": "l1", "C": 500, "max_iter": 100},
                "baseline": {"max_features": 100, "C": 1.0},
                "eval": {"target_fpr": 0.01},
                "sae": {
                    "layers": [9],
                    "width": 16384,
                    "variant": "average_l0_14",
                    "release": "gemma-scope-9b-it-res",
                },
            }
        )
    )


def _write_advbench_csv(path: Path, goals: list[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["goal", "target"])
        writer.writeheader()
        for goal in goals:
            writer.writerow({"goal": goal, "target": "Sure"})


def _write_alpaca_json(path: Path, instructions: list[str]) -> None:
    import json as _json
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [{"instruction": inst, "input": "", "output": "ok"} for inst in instructions]
    with path.open("w") as f:
        _json.dump(records, f)


def test_build_refusal_dataset_source_mode(tmp_path):
    advbench_path = tmp_path / "advbench" / "harmful_behaviors.csv"
    alpaca_path = tmp_path / "alpaca" / "alpaca_data.json"
    config_path = tmp_path / "refusal.yaml"
    output_path = tmp_path / "refusal_prompts.jsonl"

    _write_advbench_csv(advbench_path, [f"harmful {i}" for i in range(6)])
    _write_alpaca_json(alpaca_path, [f"benign {i}" for i in range(6)])
    _write_refusal_config(config_path, advbench_path, alpaca_path)

    monkeypatch_ns = SimpleNamespace()

    import scripts.main.build_refusal_dataset as brd
    # Can call main() directly since it does file I/O only (no GPU).
    import unittest.mock as mock
    with mock.patch.object(
        brd,
        "parse_args",
        return_value=Namespace(
            config=str(config_path),
            mode="source",
            labelled_responses="unused",
            source_dataset="unused",
            output=str(output_path),
        ),
    ):
        brd.main()

    from agguardrails.data import read_prompt_dataset
    dataset = read_prompt_dataset(output_path)
    assert len(dataset) == 8  # 4+4
    assert {e.split for e in dataset} == {"train", "val", "test"}


def test_build_refusal_dataset_relabel_mode(tmp_path):
    import json as _json
    advbench_path = tmp_path / "advbench" / "harmful_behaviors.csv"
    alpaca_path = tmp_path / "alpaca" / "alpaca_data.json"
    config_path = tmp_path / "refusal.yaml"
    source_path = tmp_path / "source.jsonl"
    labelled_path = tmp_path / "labelled.jsonl"
    output_path = tmp_path / "relabelled.jsonl"

    _write_advbench_csv(advbench_path, [f"harmful {i}" for i in range(6)])
    _write_alpaca_json(alpaca_path, [f"benign {i}" for i in range(6)])
    _write_refusal_config(config_path, advbench_path, alpaca_path)

    # Build source dataset first.
    import scripts.main.build_refusal_dataset as brd
    import unittest.mock as mock
    with mock.patch.object(
        brd,
        "parse_args",
        return_value=Namespace(
            config=str(config_path),
            mode="source",
            labelled_responses="unused",
            source_dataset="unused",
            output=str(source_path),
        ),
    ):
        brd.main()

    from agguardrails.data import read_prompt_dataset
    source = read_prompt_dataset(source_path)

    # Write fake labelled responses assigning refusal_label=1 to all.
    labelled_records = [
        {"example_id": e.example_id, "prompt": e.prompt, "response": "r",
         "source": e.source, "source_label": e.source_label, "label": e.label,
         "refusal_label": 1, "judge_output": "refusal"}
        for e in source
    ]
    labelled_path.write_text(
        "\n".join(_json.dumps(r) for r in labelled_records) + "\n"
    )

    with mock.patch.object(
        brd,
        "parse_args",
        return_value=Namespace(
            config=str(config_path),
            mode="relabel",
            labelled_responses=str(labelled_path),
            source_dataset=str(source_path),
            output=str(output_path),
        ),
    ):
        brd.main()

    relabelled = read_prompt_dataset(output_path)
    assert len(relabelled) == len(source)
    assert all(e.label == 1 for e in relabelled)


def test_generate_responses_script_wires_correctly(monkeypatch, tmp_path):
    advbench_path = tmp_path / "advbench" / "harmful_behaviors.csv"
    alpaca_path = tmp_path / "alpaca" / "alpaca_data.json"
    config_path = tmp_path / "refusal.yaml"
    dataset_path = tmp_path / "prompts.jsonl"
    output_path = tmp_path / "responses.jsonl"

    _write_advbench_csv(advbench_path, [f"harmful {i}" for i in range(6)])
    _write_alpaca_json(alpaca_path, [f"benign {i}" for i in range(6)])
    _write_refusal_config(config_path, advbench_path, alpaca_path)

    examples = [
        PromptExample("ex_0", "prompt 0", 1, "train", "advbench", "0", "harmful"),
        PromptExample("ex_1", "prompt 1", 0, "test", "alpaca", "1", "benign"),
    ]
    captured: dict = {}

    monkeypatch.setattr(
        generate_responses_script,
        "parse_args",
        lambda: Namespace(
            config=str(config_path),
            dataset=str(dataset_path),
            output=str(output_path),
        ),
    )
    monkeypatch.setattr(
        generate_responses_script,
        "load_model_and_tokenizer",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        generate_responses_script,
        "read_prompt_dataset",
        lambda path: examples,
    )
    monkeypatch.setattr(
        generate_responses_script,
        "generate_responses",
        lambda model, tokenizer, examples, **kwargs: (
            captured.__setitem__("kwargs", kwargs) or [
                {"example_id": e.example_id, "response": "r"} for e in examples
            ]
        ),
    )
    monkeypatch.setattr(
        generate_responses_script,
        "save_metadata",
        lambda path, **kwargs: captured.setdefault("metadata", kwargs),
    )

    generate_responses_script.main()

    assert captured["metadata"]["n_responses"] == 2
    assert captured["kwargs"]["max_new_tokens"] == 64
    assert captured["kwargs"]["temperature"] == 0.0


def test_label_refusals_script_writes_labelled_jsonl(monkeypatch, tmp_path):
    advbench_path = tmp_path / "advbench" / "harmful_behaviors.csv"
    alpaca_path = tmp_path / "alpaca" / "alpaca_data.json"
    config_path = tmp_path / "refusal.yaml"
    responses_path = tmp_path / "responses.jsonl"
    output_path = tmp_path / "labelled.jsonl"

    _write_advbench_csv(advbench_path, [f"harmful {i}" for i in range(6)])
    _write_alpaca_json(alpaca_path, [f"benign {i}" for i in range(6)])
    _write_refusal_config(config_path, advbench_path, alpaca_path)

    raw_records = [
        {"example_id": "h0", "prompt": "p", "response": "r", "source": "advbench",
         "source_label": "harmful", "label": 1},
        {"example_id": "b0", "prompt": "p", "response": "r", "source": "alpaca",
         "source_label": "benign", "label": 0},
    ]
    write_jsonl(responses_path, raw_records)

    labelled_output = [
        {**r, "refusal_label": i, "judge_output": ["refusal", "compliance"][i]}
        for i, r in enumerate(raw_records)
    ]

    monkeypatch.setattr(
        label_refusals_script,
        "parse_args",
        lambda: Namespace(
            config=str(config_path),
            responses=str(responses_path),
            output=str(output_path),
        ),
    )
    monkeypatch.setattr(
        label_refusals_script,
        "load_model_and_tokenizer",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(
        label_refusals_script,
        "label_refusals",
        lambda *args, **kwargs: labelled_output,
    )
    captured: dict = {}
    monkeypatch.setattr(
        label_refusals_script,
        "save_metadata",
        lambda path, **kwargs: captured.setdefault("metadata", kwargs),
    )

    label_refusals_script.main()

    written = read_jsonl(output_path)
    assert len(written) == 2
    assert written[0]["refusal_label"] == 0
    assert written[1]["refusal_label"] == 1
    assert captured["metadata"]["n_labelled"] == 2
