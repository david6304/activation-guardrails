from __future__ import annotations

from agguardrails.ccpp_data import normalize_record
from agguardrails.text_diagnostics import (
    TextSeparabilityConfig,
    text_separability_report,
)


def test_text_separability_report_flags_trivial_surface_features() -> None:
    examples = []
    for split in ["train", "val", "test"]:
        examples.extend(
            [
                _example(f"{split}-p1", split, 1, "alpha positive marker"),
                _example(f"{split}-p2", split, 1, "alpha positive token"),
                _example(f"{split}-n1", split, 0, "omega negative marker"),
                _example(f"{split}-n2", split, 0, "omega negative token"),
            ]
        )

    report = text_separability_report(
        examples,
        config=TextSeparabilityConfig(max_allowed_roc_auc=0.99),
    )

    assert report["gate"] == "failed"
    assert report["splits"]["test"]["roc_auc"] == 1.0


def test_text_separability_report_passes_non_separable_fixture() -> None:
    examples = []
    for split in ["train", "val", "test"]:
        examples.extend(
            [
                _example(f"{split}-p1", split, 1, "shared science response"),
                _example(f"{split}-p2", split, 1, "shared lab response"),
                _example(f"{split}-n1", split, 0, "shared science response"),
                _example(f"{split}-n2", split, 0, "shared lab response"),
            ]
        )

    report = text_separability_report(
        examples,
        config=TextSeparabilityConfig(max_allowed_roc_auc=0.99),
    )

    assert report["gate"] == "passed"


def _example(example_id: str, split: str, label: int, assistant_text: str):
    return normalize_record(
        {
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
            "metadata": {
                "generator_model_id": "ablated-gemma",
                "protected_model_id": "ablated-gemma",
            },
        }
    )

