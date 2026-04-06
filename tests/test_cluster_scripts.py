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
    assert "--nodes=1" in result.stdout
    assert "run_main_pipeline.sh" in result.stdout
    assert "--stage all" in result.stdout
    assert "--model-cache /tmp/home/models" in result.stdout
    assert (
        "--adversarial-dataset data/processed/main_adversarial.jsonl" in result.stdout
    )


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
    assert "--nodes=1" in result.stdout
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
    assert "--nodes=1" in result.stdout
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
    assert "--nodes=1" in result.stdout
    assert "run_sae_pipeline.sh" in result.stdout
    assert "--stage all" in result.stdout
    assert "--probe-dir artifacts/models/main/sae_probes" in result.stdout
    assert "--results-path results/main/sae_results.csv" in result.stdout
    assert "--model-cache /tmp/home/models" in result.stdout


def test_submit_refusal_job_prints_expected_sbatch_command():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_refusal_job.sh"
    env = {**os.environ, "USER": "pytest-user", "HOME": "/tmp/home"}

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-only",
            "--gpu-type",
            "any",
            "--stage",
            "activations",
            "--relabelled-dataset",
            "data/processed/refusal_ready.jsonl",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--gres=gpu:1 --nodelist=crannog[01-02],landonia11" in result.stdout
    assert "--nodes=1" in result.stdout
    assert "--time=12:00:00" in result.stdout
    assert "run_refusal_pipeline.sh" in result.stdout
    assert "--stage activations" in result.stdout
    assert "--relabelled-dataset data/processed/refusal_ready.jsonl" in result.stdout
    assert "--activation-dir artifacts/activations/refusal" in result.stdout
    assert "--model-cache /tmp/home/models" in result.stdout


def test_submit_refusal_source_job_omits_gpu_request():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_refusal_job.sh"
    env = {**os.environ, "USER": "pytest-user", "HOME": "/tmp/home"}

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-only",
            "--stage",
            "source",
            "--config",
            "configs/main/refusal_wildjailbreak.yaml",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--time=00:15:00" in result.stdout
    assert "run_refusal_pipeline.sh" in result.stdout
    assert "--stage source" in result.stdout
    assert "--config configs/main/refusal_wildjailbreak.yaml" in result.stdout
    assert "--gres=" not in result.stdout
    assert "--nodelist=" not in result.stdout


def test_submit_refusal_job_rejects_unknown_stage():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_refusal_job.sh"

    result = subprocess.run(
        ["bash", str(script), "--stage", "badstage", "--print-only"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "Unknown --stage: badstage" in result.stderr


def test_submit_cipher_transfer_job_prints_expected_sbatch_command():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_cipher_transfer_job.sh"
    env = {**os.environ, "USER": "pytest-user", "HOME": "/tmp/home"}

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-only",
            "--gpu-type",
            "any",
            "--stage",
            "encode",
            "--cipher",
            "rot13",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--gres=gpu:1 --nodelist=crannog[01-02],landonia11" in result.stdout
    assert "--nodes=1" in result.stdout
    assert "--time=01:00:00" in result.stdout
    assert "run_cipher_transfer_pipeline.sh" in result.stdout
    assert "--stage encode" in result.stdout
    assert "--cipher rot13" in result.stdout
    assert "--model-cache /tmp/home/models" in result.stdout


def test_submit_cipher_transfer_subset_fill_prints_expected_sbatch_command():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_cipher_transfer_job.sh"
    env = {**os.environ, "USER": "pytest-user", "HOME": "/tmp/home"}

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-only",
            "--stage",
            "subset_fill",
            "--cipher",
            "reverse",
            "--config",
            "configs/main/refusal_large_train.yaml",
            "--plain-dataset",
            "data/processed/refusal_large_train_prompts.jsonl",
            "--base-dir",
            "artifacts/cipher_transfer_large_train",
            "--existing-base-dir",
            "artifacts/cipher_transfer",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--time=02:00:00" in result.stdout
    assert "run_cipher_transfer_pipeline.sh" in result.stdout
    assert "--stage subset_fill" in result.stdout
    assert "--cipher reverse" in result.stdout
    assert "--config configs/main/refusal_large_train.yaml" in result.stdout
    assert (
        "--plain-dataset data/processed/refusal_large_train_prompts.jsonl"
        in result.stdout
    )
    assert "--base-dir artifacts/cipher_transfer_large_train" in result.stdout
    assert "--existing-base-dir artifacts/cipher_transfer" in result.stdout


def test_submit_cipher_transfer_job_requires_cipher():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_cipher_transfer_job.sh"

    result = subprocess.run(
        ["bash", str(script), "--print-only"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "--cipher is required" in result.stderr


def test_submit_transfer_eval_job_prints_expected_sbatch_command():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_transfer_eval_job.sh"
    env = {**os.environ, "USER": "pytest-user", "HOME": "/tmp/home"}

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-only",
            "--cipher-base-dir",
            "artifacts/cipher_transfer_large_train",
            "--output-dir",
            "results/refusal/cipher_transfer_large_train",
            "--ciphers",
            "reverse",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--time=00:30:00" in result.stdout
    assert "run_transfer_eval.sh" in result.stdout
    assert "--probe-dir artifacts/models/refusal/activation_probes_cv" in result.stdout
    assert "--sae-probe-dir artifacts/models/refusal/sae_probes_cv" in result.stdout
    assert "--plain-metrics-dir results/refusal/metrics_cv_large_train" in result.stdout
    assert "--cipher-base-dir artifacts/cipher_transfer_large_train" in result.stdout
    assert "--output-dir results/refusal/cipher_transfer_large_train" in result.stdout
    assert "--ciphers reverse" in result.stdout


def test_submit_probe_training_job_prints_target_aware_command():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_probe_training_job.sh"
    env = {**os.environ, "USER": "pytest-user", "HOME": "/tmp/home"}

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-only",
            "--config",
            "configs/main/refusal_wildjailbreak.yaml",
            "--dataset",
            "data/processed/refusal_wildjailbreak/refusal_labelled.jsonl",
            "--adversarial-dataset",
            "data/processed/refusal_wildjailbreak/refusal_labelled_adversarial.jsonl",
            "--label-key",
            "source_label",
            "--feature-dir",
            "artifacts/features/refusal_wildjailbreak/sae",
            "--with-latent-guard",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--time=02:00:00" in result.stdout
    assert "run_probe_training.sh" in result.stdout
    assert "--feature-dir artifacts/features/refusal_wildjailbreak/sae" in result.stdout
    assert (
        "--dataset data/processed/refusal_wildjailbreak/refusal_labelled.jsonl"
        in result.stdout
    )
    assert (
        "--adversarial-dataset "
        "data/processed/refusal_wildjailbreak/"
        "refusal_labelled_adversarial.jsonl" in result.stdout
    )
    assert "--label-key source_label" in result.stdout
    assert "--with-latent-guard" in result.stdout


def test_submit_wildjailbreak_c1_refusal_pipeline_prints_derived_command():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_wildjailbreak_c1.sh"
    env = {**os.environ, "USER": "pytest-user", "HOME": "/tmp/home"}

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-only",
            "--stage",
            "refusal-pipeline",
            "--run-tag",
            "wjb_c1_a6000",
            "--gpu-type",
            "a6000",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "run_tag=wjb_c1_a6000" in result.stdout
    assert (
        "vanilla_source_dataset="
        "data/processed/wjb_c1_a6000/refusal_prompts.jsonl" in result.stdout
    )
    assert "Command:" in result.stdout
    assert "submit_refusal_job.sh" in result.stdout
    assert "--stage all" in result.stdout
    assert "--gpu-type a6000" in result.stdout
    assert "--job-name wjb_c1_a6000-refusal-pipeline" in result.stdout
    assert "--activation-dir artifacts/activations/wjb_c1_a6000" in result.stdout


def test_submit_wildjailbreak_c1_harmfulness_train_prints_target_command():
    script = REPO_ROOT / "scripts" / "cluster" / "submit_wildjailbreak_c1.sh"
    env = {**os.environ, "USER": "pytest-user", "HOME": "/tmp/home"}

    result = subprocess.run(
        [
            "bash",
            str(script),
            "--print-only",
            "--stage",
            "harmfulness-train",
            "--run-tag",
            "wjb_harm",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert (
        "harmfulness_results_root=results/refusal/wjb_harm/harmfulness" in result.stdout
    )
    assert "submit_probe_training_job.sh" in result.stdout
    assert "--label-key source_label" in result.stdout
    assert "--dataset data/processed/wjb_harm/refusal_labelled.jsonl" in result.stdout
    assert (
        "--adversarial-dataset "
        "data/processed/wjb_harm/refusal_labelled_adversarial.jsonl" in result.stdout
    )
    assert (
        "--text-model-dir artifacts/models/wjb_harm/harmfulness/text_baseline"
        in result.stdout
    )
    assert (
        "--results-output results/refusal/wjb_harm/harmfulness/results_table.csv"
        in result.stdout
    )
    assert "--job-name wjb_harm-harmfulness-train" in result.stdout
