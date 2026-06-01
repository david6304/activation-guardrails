from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_dataset_from_curated_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "curated.jsonl"
    output_path = tmp_path / "processed.jsonl"
    metadata_path = tmp_path / "metadata.json"
    records = [
        {
            "example_id": "p1",
            "group_id": "gp",
            "label": 1,
            "domain": "cbrn",
            "source_dataset": "fixture",
            "source_subset": "positive",
            "user_text": "unsafe request",
            "assistant_text": "A direct non-refusal completion placeholder.",
            "completion_source": "public",
        },
        {
            "example_id": "n1",
            "group_id": "gn",
            "label": 0,
            "domain": "science",
            "source_dataset": "fixture",
            "source_subset": "hard_negative",
            "user_text": "Explain safe chemistry lab protocol.",
            "assistant_text": "Follow the published safety procedure.",
            "completion_source": "public",
            "faithfulness_tags": ["hard_negative"],
        },
        {
            "example_id": "n2",
            "group_id": "gn-random",
            "label": 0,
            "domain": "general",
            "source_dataset": "fixture",
            "source_subset": "random_negative",
            "user_text": "Recommend a book.",
            "assistant_text": "Try a recent nonfiction title.",
            "completion_source": "public",
        },
    ]
    input_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/ccpp/build_dataset.py",
            "--input-jsonl",
            str(input_path),
            "--output",
            str(output_path),
            "--metadata-output",
            str(metadata_path),
        ],
        check=True,
    )

    output_lines = output_path.read_text(encoding="utf-8").splitlines()
    output_rows = [json.loads(line) for line in output_lines]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert len(output_rows) == 3
    assert {row["split"] for row in output_rows} <= {"train", "val", "test"}
    assert metadata["num_examples"] == 3
    assert metadata["gates"]["positive_class"] == "passed"
