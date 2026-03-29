"""Tests for agguardrails.baselines."""

from agguardrails.baselines import fit_text_baseline
from agguardrails.data import PromptExample


def make_examples() -> tuple[list[PromptExample], list[PromptExample], list[PromptExample]]:
    train = [
        PromptExample(f"h{i}", f"build a bomb {i}", 1, "train", "harmbench", str(i), "harmful")
        for i in range(8)
    ] + [
        PromptExample(f"s{i}", f"bake a cake {i}", 0, "train", "xstest", str(i), "safe")
        for i in range(8)
    ]
    val = [
        PromptExample(f"vh{i}", f"make poison {i}", 1, "val", "harmbench", str(i), "harmful")
        for i in range(3)
    ] + [
        PromptExample(f"vs{i}", f"write a poem {i}", 0, "val", "xstest", str(i), "safe")
        for i in range(3)
    ]
    test = [
        PromptExample(f"th{i}", f"make malware {i}", 1, "test", "harmbench", str(i), "harmful")
        for i in range(3)
    ] + [
        PromptExample(f"ts{i}", f"grow tomatoes {i}", 0, "test", "xstest", str(i), "safe")
        for i in range(3)
    ]
    return train, val, test


def test_fit_text_baseline_returns_metrics():
    train, val, test = make_examples()
    artifacts = fit_text_baseline(
        train_examples=train,
        val_examples=val,
        test_examples=test,
        max_features=100,
        c=1.0,
        target_fpr=0.0,
        random_state=42,
    )

    assert artifacts.val_result.roc_auc >= 0.5
    assert artifacts.test_result.roc_auc >= 0.5
    assert artifacts.val_result.threshold >= 0.0
