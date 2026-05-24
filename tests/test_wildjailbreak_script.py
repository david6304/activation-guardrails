import json
import subprocess
import sys
from pathlib import Path

import yaml


def _row(data_type, index):
    return {
        "id": f"{data_type}-{index}",
        "vanilla": f"vanilla {data_type} {index}",
        "adversarial": f"adversarial {data_type} {index}",
        "completion": "",
        "tactics": '["encoding"]' if data_type.startswith("adversarial") else "",
        "data_type": data_type,
    }


def test_build_wildjailbreak_script_writes_outputs_from_local_jsonl(tmp_path):
    input_path = tmp_path / "wildjailbreak.jsonl"
    data_path = tmp_path / "normalized.jsonl"
    metadata_path = tmp_path / "metadata.json"
    config_path = tmp_path / "config.yaml"

    rows = [
        _row(data_type, index)
        for data_type in (
            "vanilla_harmful",
            "vanilla_benign",
            "adversarial_harmful",
            "adversarial_benign",
        )
        for index in range(4)
    ]
    with input_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    config = {
        "model": {"id": "google/gemma-2-9b-it", "revision": None},
        "dataset": {
            "id": "allenai/wildjailbreak",
            "revision": "local-test",
            "local_path": str(input_path),
        },
        "sampling": {
            "seed": 42,
            "per_data_type": {
                "vanilla_harmful": 4,
                "vanilla_benign": 4,
                "adversarial_harmful": 4,
                "adversarial_benign": 4,
            },
        },
        "splits": {
            "vanilla": {"train": 0.70, "val": 0.15, "test": 0.15},
            "adversarial_split": "transfer",
        },
        "outputs": {
            "data_path": str(data_path),
            "metadata_path": str(metadata_path),
        },
    }
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle)

    script_path = Path("scripts/ccpp/build_wildjailbreak_dataset.py")
    result = subprocess.run(
        [sys.executable, str(script_path), "--config", str(config_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote 16 examples" in result.stdout
    assert data_path.exists()
    assert metadata_path.exists()

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    assert metadata["config_path"] == str(config_path)
    assert metadata["dataset"]["revision"] == "local-test"
    assert metadata["splits"]["counts_by_split"]["transfer"] == 8
