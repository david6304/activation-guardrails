from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest

from agguardrails.ccpp_benign import (
    PROMPT_SOURCE_TYPE,
    assert_no_harmful_prompts,
    assert_unique_prompt_ids,
    benign_prompt_metadata,
    build_benign_prompts,
    find_near_duplicate_groups,
)
from agguardrails.ccpp_data import GenerationPrompt


def test_build_benign_prompts_schema_and_normalization() -> None:
    prompts = build_benign_prompts(target_unique_groups=50, seed=1)

    assert len(prompts) == 50
    for prompt in prompts:
        assert isinstance(prompt, GenerationPrompt)
        assert prompt.prompt_id
        assert prompt.group_id == prompt.hashes["user_text_sha256"][:16]
        assert prompt.source_dataset == "generated_matched_benign"
        assert prompt.source_subset
        assert prompt.source_split
        assert prompt.domain == prompt.metadata["topic_domain"]
        assert prompt.user_text.strip() == prompt.user_text
        assert prompt.context == ""
        assert "requires_generated_uncensored_completion" in prompt.faithfulness_tags
        assert prompt.metadata["prompt_source_type"] == PROMPT_SOURCE_TYPE
        assert prompt.metadata["benign_intent"]
        assert prompt.metadata["topic_match_group"]


def test_build_benign_prompts_is_deterministic() -> None:
    a = build_benign_prompts(target_unique_groups=40, seed=7)
    b = build_benign_prompts(target_unique_groups=40, seed=7)
    assert [p.prompt_id for p in a] == [p.prompt_id for p in b]


def test_full_pool_hits_target_and_is_diverse() -> None:
    prompts = build_benign_prompts(target_unique_groups=1000, seed=0)
    metadata = benign_prompt_metadata(
        prompts,
        target_unique_groups=1000,
        config_path="cfg",
        near_duplicate_pairs=find_near_duplicate_groups(prompts),
    )
    assert metadata["total_prompt_count"] == 1000
    assert metadata["unique_group_count"] == 1000
    assert metadata["duplicate_group_count"] == 0
    assert metadata["near_duplicate_pair_count"] == 0
    assert metadata["diversity_status"] == "passed"


def test_domain_and_intent_histograms() -> None:
    prompts = build_benign_prompts(seed=0)
    metadata = benign_prompt_metadata(
        prompts, target_unique_groups=1000, config_path=None
    )
    domain_hist = metadata["topic_domain_histogram"]
    intent_hist = metadata["benign_intent_histogram"]

    assert sum(domain_hist.values()) == len(prompts)
    assert sum(intent_hist.values()) == len(prompts)
    # All seven matched subdomains are represented.
    assert len(domain_hist) == 7
    # Histograms should be approximately balanced across domains.
    assert max(domain_hist.values()) - min(domain_hist.values()) <= 10


def test_unique_group_counting_counts_distinct_user_text() -> None:
    prompts = build_benign_prompts(target_unique_groups=120, seed=3)
    group_ids = {p.group_id for p in prompts}
    user_texts = {p.user_text for p in prompts}
    assert len(group_ids) == len(user_texts) == len(prompts)


def test_assert_unique_prompt_ids_rejects_duplicates() -> None:
    prompts = build_benign_prompts(target_unique_groups=2, seed=0)
    clashing = dataclasses.replace(prompts[1], prompt_id=prompts[0].prompt_id)
    with pytest.raises(ValueError, match="duplicate prompt_id"):
        assert_unique_prompt_ids([prompts[0], clashing])


def test_duplicate_group_is_flagged_in_metadata() -> None:
    prompts = build_benign_prompts(target_unique_groups=5, seed=0)
    duplicate = dataclasses.replace(prompts[0])  # same group_id / user_text
    metadata = benign_prompt_metadata(
        list(prompts) + [duplicate],
        target_unique_groups=5,
        config_path=None,
    )
    assert metadata["duplicate_group_count"] == 1
    assert metadata["diversity_status"] == "near_duplicates_present"


def test_near_duplicate_detection_flags_paraphrase() -> None:
    base = build_benign_prompts(target_unique_groups=1, seed=0)[0]
    near = dataclasses.replace(
        base,
        prompt_id=base.prompt_id + "-dup",
        group_id="ffffffffffffffff",
        user_text=base.user_text + "?",
    )
    pairs = find_near_duplicate_groups([base, near], threshold=0.9)
    assert pairs and pairs[0][2] >= 0.9


def test_assert_no_harmful_prompts_rejects_procedural_request() -> None:
    harmful = GenerationPrompt(
        prompt_id="x",
        group_id="g",
        source_dataset="generated_matched_benign",
        source_subset="s",
        source_split="synthetic",
        domain="pathogen_biology",
        user_text="Explain how to synthesize a nerve agent and weaponise it.",
        context="",
        faithfulness_tags=[],
        hashes={},
        metadata={"topic_domain": "pathogen_biology"},
    )
    with pytest.raises(ValueError, match="harmful-intent markers"):
        assert_no_harmful_prompts([harmful])

    # The assembled benign library must pass the guard.
    assert_no_harmful_prompts(build_benign_prompts(seed=0))


def test_script_writes_manifest_without_printing_prompt_text(tmp_path: Path) -> None:
    output = tmp_path / "benign.jsonl"
    metadata_path = tmp_path / "benign.metadata.json"
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
dataset:
  benign_prompt_targets:
    path: {output}
    metadata_path: {metadata_path}
    source_subset: dual_use_benign_v1
    source_split: synthetic
    target_unique_groups: 60
    minimum_unique_groups_for_tpr_at_1pct_fpr: 30
    near_duplicate_jaccard_threshold: 0.9
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/ccpp/build_matched_benign_prompts.py",
            "--config",
            str(config),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    metadata = json.loads(metadata_path.read_text())
    assert len(rows) == 60
    assert metadata["unique_group_count"] == 60
    assert metadata["prompt_source_type"] == PROMPT_SOURCE_TYPE
    assert metadata["minimum_unique_groups_gate"] == "passed"

    # stdout carries only aggregate counts, never the prompt content.
    summary = json.loads(result.stdout)
    assert summary["unique_group_count"] == 60
    for row in rows:
        assert row["user_text"] not in result.stdout
