"""Tests for cluster shell wrappers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_submit_main_job_prints_expected_sbatch_command():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_main_job.sh"
    env = {**os.environ, "USER": "pytest-user", "HOME": "/tmp/home"}

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-only",
            "--gpu-type",
            "any",
            "--stage",
            "all",
            "--adversarial-dataset",
            "data/processed/main_adversarial.jsonl",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--gres=gpu:1 --nodelist=crannog[01-02],landonia11" in result.stdout
    assert "run_main_pipeline.sh" in result.stdout
    assert "--stage all" in result.stdout
    assert "--model-cache /tmp/home/models" in result.stdout
    assert "--adversarial-dataset data/processed/main_adversarial.jsonl" in result.stdout


def test_submit_main_job_rejects_unknown_gpu_type():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_main_job.sh"

    result = subprocess.run(
        ["bash", str(script), "--gpu-type", "badgpu", "--print-only"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Unknown --gpu-type: badgpu" in result.stderr


def test_submit_mvp_job_prints_expected_sbatch_command():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_mvp_job.sh"
    env = {**os.environ, "USER": "pytest-user", "HOME": "/tmp/home"}

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-only",
            "--gpu-type",
            "a6000",
            "--stage",
            "report",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--gres=gpu:nvidia_rtx_a6000:1" in result.stdout
    assert "run_mvp_pipeline.sh" in result.stdout
    assert "--stage report" in result.stdout
    assert "--model-cache /tmp/home/models" in result.stdout


def test_submit_sae_encoding_job_prints_expected_sbatch_command():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_sae_encoding_job.sh"
    env = {**os.environ, "USER": "pytest-user", "HOME": "/tmp/home"}

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-only",
            "--gpu-type",
            "any",
            "--batch-size",
            "128",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--gres=gpu:1 --nodelist=crannog[01-02],landonia11" in result.stdout
    assert "run_sae_pipeline.sh" in result.stdout
    assert "--stage encode" in result.stdout
    assert "--batch-size 128" in result.stdout
    assert "--metrics-dir results/main/metrics" in result.stdout
    assert "--model-cache /tmp/home/models" in result.stdout


def test_submit_sae_job_prints_expected_sbatch_command():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_sae_job.sh"
    env = {**os.environ, "USER": "pytest-user", "HOME": "/tmp/home"}

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-only",
            "--gpu-type",
            "a6000",
            "--stage",
            "all",
            "--batch-size",
            "64",
            "--results-path",
            "results/main/sae_results.csv",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--gres=gpu:nvidia_rtx_a6000:1" in result.stdout
    assert "run_sae_pipeline.sh" in result.stdout
    assert "--stage all" in result.stdout
    assert "--probe-dir artifacts/models/main/sae_probes" in result.stdout
    assert "--results-path results/main/sae_results.csv" in result.stdout
    assert "--model-cache /tmp/home/models" in result.stdout
