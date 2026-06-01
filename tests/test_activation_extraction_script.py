from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from agguardrails.activations import activation_index_rows, load_activation_cache


def test_extract_streaming_activations_mock_mode(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.jsonl"
    output_npz = tmp_path / "activations.npz"
    output_index = tmp_path / "activations.index.jsonl"
    records = [
        {
            "example_id": "p1",
            "group_id": "gp",
            "split": "train",
            "label": 1,
            "domain": "cbrn",
            "source_dataset": "fixture",
            "source_subset": "positive",
            "user_text": "unsafe request",
            "assistant_text": "A direct non-refusal assistant completion placeholder.",
            "completion_source": "public",
        },
        {
            "example_id": "n1",
            "group_id": "gn",
            "split": "val",
            "label": 0,
            "domain": "science",
            "source_dataset": "fixture",
            "source_subset": "hard_negative",
            "user_text": "Explain safe chemistry lab decontamination.",
            "assistant_text": "Use PPE and follow the published lab protocol.",
            "completion_source": "public",
            "faithfulness_tags": ["hard_negative"],
        },
    ]
    dataset_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/ccpp/extract_streaming_activations.py",
            "--dataset-jsonl",
            str(dataset_path),
            "--output-npz",
            str(output_npz),
            "--output-index",
            str(output_index),
            "--mock",
        ],
        check=True,
    )

    examples, metadata = load_activation_cache(
        npz_path=output_npz,
        index_path=output_index,
    )
    index_rows = activation_index_rows(output_index)

    assert len(examples) == 2
    assert metadata["mock"] is True
    assert metadata["num_examples"] == 2
    assert index_rows[0]["feature_dim"] == 12
    assert examples[0].features.shape[1] == 12

