from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from agguardrails.activations import ActivationExample, save_activation_cache


def test_train_swim_probe_script_smoke(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    npz_path = tmp_path / "activations.npz"
    index_path = tmp_path / "activations.index.jsonl"
    model_path = tmp_path / "probe.pt"
    scores_path = tmp_path / "scores.jsonl"
    metrics_path = tmp_path / "metrics.json"
    examples = [
        _example("train-p1", "train", 1, 2.5),
        _example("train-p2", "train", 1, 2.0),
        _example("train-n1", "train", 0, -2.5),
        _example("train-n2", "train", 0, -2.0),
        _example("val-p1", "val", 1, 2.2),
        _example("val-n1", "val", 0, -2.2),
        _example("test-p1", "test", 1, 2.1),
        _example("test-n1", "test", 0, -2.1),
    ]
    save_activation_cache(
        examples,
        npz_path=npz_path,
        index_path=index_path,
        metadata={"model_id": "fixture"},
    )
    config_path.write_text(
        """
model:
  id: fixture-model
  revision: null
tokenizer:
  id: fixture-tokenizer
  revision: null
activation:
  artifact_dir: unused
swim_probe:
  artifact_dir: unused
  scores_path: unused.jsonl
  metrics_path: unused.json
  seed: 1
  window_size: 2
  softmax_temperature: 1.0
  ema_gamma: null
  train:
    epochs: 30
    learning_rate: 0.05
    weight_decay: 0.0
    batch_size: 2
  threshold:
    primary_max_fpr: 0.001
    secondary_max_fpr: 0.01
""",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/ccpp/train_swim_probe.py",
            "--config",
            str(config_path),
            "--activation-npz",
            str(npz_path),
            "--activation-index",
            str(index_path),
            "--model-output",
            str(model_path),
            "--scores-output",
            str(scores_path),
            "--metrics-output",
            str(metrics_path),
        ],
        check=True,
    )

    score_lines = scores_path.read_text(encoding="utf-8").splitlines()
    score_rows = [json.loads(line) for line in score_lines]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert model_path.exists()
    assert len(score_rows) == len(examples)
    assert "group_id" in score_rows[0]
    assert metrics["headline_level"] == "group_level"
    assert metrics["splits"]["test"]["roc_auc"] == 1.0
    assert metrics["row_level"]["splits"]["test"]["roc_auc"] == 1.0
    assert metrics["group_level"]["splits"]["test"]["roc_auc"] == 1.0
    assert "threshold" in metrics["thresholds"]["secondary"]
    assert metrics["thresholds"]["secondary"]["validation"]["tpr"] == 1.0
    assert "tpr_wilson_95" in metrics["thresholds"]["secondary"]["splits"]["test"]
    expected_warning = (
        "validation negatives are too few to resolve this FPR with one false positive"
    )
    actual_warning = metrics["thresholds"]["primary"]["fpr_resolution_warning"]
    assert actual_warning == expected_warning


def _example(
    example_id: str,
    split: str,
    label: int,
    value: float,
) -> ActivationExample:
    features = np.full((4, 3), value, dtype=np.float32)
    return ActivationExample(
        example_id=example_id,
        label=label,
        split=split,
        group_id=example_id,
        features=features,
        token_ids=np.arange(features.shape[0], dtype=np.int64),
        token_mask=np.ones(features.shape[0], dtype=bool),
        layers=[1],
        metadata={},
    )
