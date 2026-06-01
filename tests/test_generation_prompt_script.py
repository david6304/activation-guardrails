from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_generation_prompts_from_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "harmbench_rows.jsonl"
    output_path = tmp_path / "prompts.jsonl"
    metadata_path = tmp_path / "prompts.metadata.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
dataset:
  normalized_path: unused.jsonl
  generation_prompts:
    path: {output_path}
    metadata_path: {metadata_path}
    allowed_domains:
      - chemical_biological
    min_total_prompts_for_report: 1
    min_unique_groups_for_report: 1
""",
        encoding="utf-8",
    )
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
            "--config",
            str(config_path),
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
    assert metadata["domain_counts_after_filter"] == {"chemical_biological": 1}
    assert metadata["reportable_size_gate"] == "passed"


def test_build_clearharm_generation_prompts_from_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "clearharm_rows.jsonl"
    output_path = tmp_path / "prompts.jsonl"
    metadata_path = tmp_path / "prompts.metadata.json"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
dataset:
  normalized_path: unused.jsonl
  generation_prompts:
    path: {output_path}
    metadata_path: {metadata_path}
    allowed_domains:
      - chemical_biological
    min_total_prompts_for_report: 1
    min_unique_groups_for_report: 1
""",
        encoding="utf-8",
    )
    input_path.write_text(
        json.dumps(
            {
                "clf_label": 1,
                "content": ["Synthetic CBRN harmful prompt placeholder."],
                "proxy_clf_label": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/ccpp/build_generation_prompts.py",
            "--config",
            str(config_path),
            "--dataset-id",
            "AlignmentResearch/ClearHarm",
            "--source-subset",
            "rep40",
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
    assert rows[0]["source_dataset"] == "AlignmentResearch/ClearHarm"
    assert rows[0]["domain"] == "chemical_biological"
    assert rows[0]["metadata"]["source_unique_prompt_hash"] == rows[0]["group_id"]
    assert metadata["domain_counts_after_filter"] == {"chemical_biological": 1}
    assert metadata["reportable_size_gate"] == "passed"
