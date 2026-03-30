"""Dataset loading, normalisation, sampling, and split generation.

MVP loaders (HarmBench, XSTest) are pilot-only — kept for reference.
Main experiment uses WildJailbreak loaders below.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split

from agguardrails.io import read_jsonl, write_jsonl


@dataclass(frozen=True)
class PromptExample:
    """Canonical prompt record used across the MVP pipeline."""

    example_id: str
    prompt: str
    label: int
    split: str
    source: str
    source_id: str
    source_label: str


def load_harmbench_examples(path: str | Path) -> list[dict[str, Any]]:
    """Load harmful prompts from the local HarmBench JSONL file."""
    records = read_jsonl(path)
    examples: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        content = record.get("content", [])
        if not content:
            continue

        prompt = content[0].strip()
        if not prompt:
            continue

        source_id = str(record.get("id", idx))
        examples.append(
            {
                "source": "harmbench",
                "source_id": source_id,
                "source_label": "harmful",
                "label": 1,
                "prompt": prompt,
            }
        )

    return examples


def load_xstest_examples(
    path: str | Path,
    *,
    label: str = "safe",
) -> list[dict[str, Any]]:
    """Load prompts from XSTest and keep only the requested label subset."""
    records = read_jsonl(path)
    examples: list[dict[str, Any]] = []

    for record in records:
        if record.get("label") != label:
            continue

        prompt = str(record.get("prompt", "")).strip()
        if not prompt:
            continue

        source_id = str(record.get("id", len(examples)))
        examples.append(
            {
                "source": "xstest",
                "source_id": source_id,
                "source_label": str(record["label"]),
                "label": 0,
                "prompt": prompt,
            }
        )

    return examples


def sample_examples(
    examples: list[dict[str, Any]],
    *,
    n: int,
) -> list[dict[str, Any]]:
    """Take the first *n* examples from a deterministic local source."""
    if n > len(examples):
        msg = f"Requested {n} examples but only {len(examples)} available."
        raise ValueError(msg)
    return examples[:n]


def make_stratified_splits(
    examples: list[dict[str, Any]],
    *,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> list[PromptExample]:
    """Assign stratified train/val/test splits to canonical prompt examples."""
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-8:
        msg = "train_size + val_size + test_size must sum to 1.0"
        raise ValueError(msg)

    labels = [example["label"] for example in examples]
    indices = list(range(len(examples)))

    train_idx, temp_idx = train_test_split(
        indices,
        train_size=train_size,
        stratify=labels,
        random_state=seed,
    )

    temp_fraction = val_size + test_size
    val_fraction_of_temp = val_size / temp_fraction
    temp_labels = [labels[idx] for idx in temp_idx]

    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_fraction_of_temp,
        stratify=temp_labels,
        random_state=seed,
    )

    split_lookup = {
        **{idx: "train" for idx in train_idx},
        **{idx: "val" for idx in val_idx},
        **{idx: "test" for idx in test_idx},
    }

    dataset: list[PromptExample] = []
    for idx, example in enumerate(examples):
        example_id = f"{example['source']}::{example['source_id']}"
        dataset.append(
            PromptExample(
                example_id=example_id,
                prompt=example["prompt"],
                label=example["label"],
                split=split_lookup[idx],
                source=example["source"],
                source_id=example["source_id"],
                source_label=example["source_label"],
            )
        )

    return dataset


def build_mvp_dataset(
    *,
    harmful_path: str | Path,
    benign_path: str | Path,
    benign_label: str,
    n_harmful: int,
    n_benign: int,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> list[PromptExample]:
    """Build the frozen MVP prompt dataset from local JSONL sources."""
    harmful = sample_examples(load_harmbench_examples(harmful_path), n=n_harmful)
    benign = sample_examples(
        load_xstest_examples(benign_path, label=benign_label),
        n=n_benign,
    )
    combined = harmful + benign
    return make_stratified_splits(
        combined,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )


def write_prompt_dataset(path: str | Path, dataset: list[PromptExample]) -> Path:
    """Write canonical prompt examples as JSONL."""
    records = [asdict(example) for example in dataset]
    return write_jsonl(path, records)


def read_prompt_dataset(path: str | Path) -> list[PromptExample]:
    """Read canonical prompt examples from JSONL."""
    return [PromptExample(**record) for record in read_jsonl(path)]


def split_prompt_dataset(
    dataset: list[PromptExample],
) -> dict[str, list[PromptExample]]:
    """Group canonical prompt examples by split name."""
    grouped: dict[str, list[PromptExample]] = {"train": [], "val": [], "test": []}
    for example in dataset:
        grouped.setdefault(example.split, []).append(example)
    return grouped


# ---------------------------------------------------------------------------
# WildJailbreak loaders (main experiment)
# ---------------------------------------------------------------------------
# Expected JSONL schema (produced by scripts/download_wildjailbreak.py):
#   {"id": str, "prompt": str, "data_type": "vanilla_harmful"|"vanilla_benign"|
#    "adversarial_harmful", "source": "wildjailbreak"}
#
# data_type → label mapping:
#   vanilla_harmful / adversarial_harmful → label=1
#   vanilla_benign                        → label=0


def load_wildjailbreak_examples(
    path: str | Path,
    *,
    data_type: str,
) -> list[dict[str, Any]]:
    """Load WildJailbreak prompts of a given data_type from a normalised JSONL.

    Args:
        path: Path to the normalised JSONL produced by scripts/download_wildjailbreak.py.
        data_type: One of ``"vanilla_harmful"``, ``"vanilla_benign"``,
            ``"adversarial_harmful"``.
    """
    valid_types = {"vanilla_harmful", "vanilla_benign", "adversarial_harmful"}
    if data_type not in valid_types:
        msg = f"data_type must be one of {valid_types}, got {data_type!r}"
        raise ValueError(msg)

    label = 0 if data_type == "vanilla_benign" else 1
    source_label = data_type

    records = read_jsonl(path)
    examples: list[dict[str, Any]] = []

    for idx, record in enumerate(records):
        if record.get("data_type") != data_type:
            continue
        prompt = str(record.get("prompt", "")).strip()
        if not prompt:
            continue
        source_id = str(record.get("id", idx))
        examples.append(
            {
                "source": "wildjailbreak",
                "source_id": source_id,
                "source_label": source_label,
                "label": label,
                "prompt": prompt,
            }
        )

    return examples


def build_main_dataset(
    *,
    wildjailbreak_path: str | Path,
    n_vanilla_harmful: int,
    n_vanilla_benign: int,
    train_size: float,
    val_size: float,
    test_size: float,
    seed: int,
) -> list[PromptExample]:
    """Build the main experiment train/val/test dataset from WildJailbreak vanilla split.

    Only vanilla examples are used for train/val/test. The adversarial test set
    is kept separate and built via ``build_adversarial_test_set()``.

    Args:
        wildjailbreak_path: Path to the normalised WildJailbreak JSONL.
        n_vanilla_harmful: Number of vanilla harmful examples to sample.
        n_vanilla_benign: Number of vanilla benign examples to sample.
        train_size: Fraction for training split.
        val_size: Fraction for validation split.
        test_size: Fraction for test split.
        seed: Random seed for stratified splitting.
    """
    harmful = sample_examples(
        load_wildjailbreak_examples(wildjailbreak_path, data_type="vanilla_harmful"),
        n=n_vanilla_harmful,
    )
    benign = sample_examples(
        load_wildjailbreak_examples(wildjailbreak_path, data_type="vanilla_benign"),
        n=n_vanilla_benign,
    )
    combined = harmful + benign
    return make_stratified_splits(
        combined,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
    )


def build_adversarial_test_set(
    *,
    wildjailbreak_path: str | Path,
    n_adversarial: int,
) -> list[PromptExample]:
    """Build the held-out adversarial test set from WildJailbreak.

    All examples are adversarial_harmful (label=1). No benign counterpart
    is included — this set tests detector robustness under obfuscation only.

    Returns PromptExample records all with split="test".
    """
    adversarial = sample_examples(
        load_wildjailbreak_examples(wildjailbreak_path, data_type="adversarial_harmful"),
        n=n_adversarial,
    )
    dataset: list[PromptExample] = []
    for example in adversarial:
        example_id = f"wildjailbreak::{example['source_id']}"
        dataset.append(
            PromptExample(
                example_id=example_id,
                prompt=example["prompt"],
                label=example["label"],
                split="test",
                source=example["source"],
                source_id=example["source_id"],
                source_label=example["source_label"],
            )
        )
    return dataset
