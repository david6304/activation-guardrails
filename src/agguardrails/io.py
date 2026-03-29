"""I/O helpers for JSONL datasets, numpy/sklearn/torch artifacts, and metadata."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

from agguardrails.utils import get_git_hash, timestamp_str


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file and return a list of dicts."""
    path = Path(path)
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, records: list[dict]) -> Path:
    """Write a list of dicts as JSONL. Creates parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Artifact save / load (numpy, sklearn/joblib, torch)
# ---------------------------------------------------------------------------


def save_artifact(obj: Any, path: str | Path) -> Path:
    """Save an artifact, dispatching on type.

    - numpy ndarray  -> .npz
    - torch Tensor   -> .pt
    - anything else  -> .joblib  (sklearn models, pipelines, etc.)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(obj, np.ndarray):
        np.savez_compressed(path.with_suffix(".npz"), data=obj)
        return path.with_suffix(".npz")

    if isinstance(obj, torch.Tensor):
        torch.save(obj, path.with_suffix(".pt"))
        return path.with_suffix(".pt")

    # Default: joblib (sklearn models, vectorisers, etc.)
    joblib.dump(obj, path.with_suffix(".joblib"))
    return path.with_suffix(".joblib")


def load_artifact(path: str | Path) -> Any:
    """Load an artifact, dispatching on file extension."""
    path = Path(path)
    suffix = path.suffix

    if suffix == ".npz":
        return np.load(path)["data"]

    if suffix == ".pt":
        return torch.load(path, weights_only=False)

    if suffix == ".joblib":
        return joblib.load(path)

    msg = f"Unsupported artifact extension: {suffix}"
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Metadata sidecar
# ---------------------------------------------------------------------------


def save_metadata(path: str | Path, **kwargs: Any) -> Path:
    """Write a JSON metadata sidecar alongside an artifact or result.

    Automatically includes git hash and timestamp.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "git_hash": get_git_hash(),
        "timestamp": timestamp_str(),
        **kwargs,
    }

    with path.open("w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return path
