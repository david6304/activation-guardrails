from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_check_text_separability_script_blocks_near_ceiling(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    dataset_path = tmp_path / "dataset.jsonl"
    metrics_path = tmp_path / "text_metrics.json"
    config_path.write_text(
        f"""
dataset:
  normalized_path: {dataset_path}
  text_separability_gate:
    metrics_path: {metrics_path}
    max_allowed_roc_auc: 0.99
    text_fields:
      - user_text
      - assistant_text
""",
        encoding="utf-8",
    )
    rows = []
    for split in ["train", "val"]:
        rows.extend(
            [
                _row(f"{split}-p1", split, 1, "alpha positive marker"),
                _row(f"{split}-p2", split, 1, "alpha positive token"),
                _row(f"{split}-n1", split, 0, "omega negative marker"),
                _row(f"{split}-n2", split, 0, "omega negative token"),
            ]
        )
    dataset_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ccpp/check_text_separability.py",
            "--config",
            str(config_path),
        ],
        check=False,
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    assert result.returncode != 0
    assert metrics["gate"] == "failed"


def _row(example_id: str, split: str, label: int, assistant_text: str):
    return {
        "example_id": example_id,
        "group_id": example_id,
        "split": split,
        "label": label,
        "domain": "cbrn" if label else "science",
        "source_dataset": "fixture",
        "source_subset": "text",
        "user_text": "Shared chemistry prompt.",
        "assistant_text": assistant_text,
        "completion_source": "generated_uncensored",
        "faithfulness_tags": ["hard_negative"] if label == 0 else [],
    }

