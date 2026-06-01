from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_generation_prompts_from_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "harmbench_rows.jsonl"
    output_path = tmp_path / "prompts.jsonl"
    metadata_path = tmp_path / "prompts.metadata.json"
    input_path.write_text(
        json.dumps(
            {
                "prompt": "Synthetic harmful-behavior prompt placeholder.",
                "category": "chemical_biological",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/ccpp/build_generation_prompts.py",
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
    rows = [json.loads(line) for line in output_lines]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert rows[0]["source_dataset"] == "walledai/HarmBench"
    assert "requires_generated_uncensored_completion" in rows[0]["faithfulness_tags"]
    assert metadata["status"] == "prompt_only_requires_completion_generation"
