"""Dataset normalization and gates for the CC++ open-weight replication."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from agguardrails.metrics import roc_auc

LABEL_NEGATIVE = 0
LABEL_POSITIVE = 1

REFUSAL_MARKERS = (
    "i can't help",
    "i cannot help",
    "i can't assist",
    "i cannot assist",
    "i won't help",
    "i will not help",
    "not able to help",
    "can't provide instructions",
    "cannot provide instructions",
    "against policy",
    "illegal or harmful",
    "harmful instructions",
)

CBRN_TERMS = (
    "biology",
    "chemical",
    "chemistry",
    "pathogen",
    "toxin",
    "virus",
    "bacteria",
    "radiation",
    "radiological",
    "nuclear",
    "isotope",
    "explosive",
    "energetic material",
    "lab protocol",
    "synthesis",
    "decontamination",
    "hazmat",
)

CLEARHARM_CBRN_DOMAIN = "chemical_biological"


class DatasetGateError(ValueError):
    """Raised when normalized data violates a replication gate."""


@dataclass(frozen=True)
class NormalizedExchange:
    """Normalized exchange consumed by CC++ dataset/probe scripts."""

    example_id: str
    group_id: str
    split: str
    label: int
    domain: str
    source_dataset: str
    source_subset: str
    user_text: str
    assistant_text: str
    exchange_messages: list[dict[str, str]]
    completion_source: str
    faithfulness_tags: list[str] = field(default_factory=list)
    hashes: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationPrompt:
    """Prompt-only source row for controlled positive-completion generation."""

    prompt_id: str
    group_id: str
    source_dataset: str
    source_subset: str
    source_split: str
    domain: str
    user_text: str
    context: str
    faithfulness_tags: list[str]
    hashes: dict[str, str]
    metadata: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSONL") from exc
    return rows


def write_jsonl(path: Path, examples: Iterable[NormalizedExchange]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_json_dict(), sort_keys=True) + "\n")


def write_generation_prompts(
    path: Path, prompts: Iterable[GenerationPrompt]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for prompt in prompts:
            handle.write(json.dumps(prompt.to_json_dict(), sort_keys=True) + "\n")


def load_normalized_jsonl(path: Path) -> list[NormalizedExchange]:
    return [normalize_record(row) for row in read_jsonl(path)]


def normalize_record(record: Mapping[str, Any]) -> NormalizedExchange:
    """Normalize one mapping into the canonical CC++ exchange schema."""

    label = int(record["label"])
    if label not in {LABEL_NEGATIVE, LABEL_POSITIVE}:
        raise ValueError(f"label must be 0 or 1, got {label!r}")

    user_text = str(record.get("user_text", "")).strip()
    assistant_text = str(record.get("assistant_text", "")).strip()
    if not user_text:
        raise ValueError("user_text is required")

    exchange_messages = _normalize_messages(
        record.get("exchange_messages"), user_text, assistant_text
    )
    example_id = str(record.get("example_id") or _default_example_id(record))
    group_id = str(record.get("group_id") or stable_text_hash(user_text)[:16])
    split = str(record.get("split") or "unassigned")
    domain = str(record.get("domain") or "unknown")
    source_dataset = str(record.get("source_dataset") or "unknown")
    source_subset = str(record.get("source_subset") or "")
    completion_source = str(record.get("completion_source") or "unknown")
    faithfulness_tags = list(record.get("faithfulness_tags") or [])
    metadata = dict(record.get("metadata") or {})

    hashes = dict(record.get("hashes") or {})
    hashes.setdefault("user_text_sha256", stable_text_hash(user_text))
    hashes.setdefault("assistant_text_sha256", stable_text_hash(assistant_text))
    hashes.setdefault(
        "exchange_sha256",
        stable_text_hash(json.dumps(exchange_messages, sort_keys=True)),
    )

    return NormalizedExchange(
        example_id=example_id,
        group_id=group_id,
        split=split,
        label=label,
        domain=domain,
        source_dataset=source_dataset,
        source_subset=source_subset,
        user_text=user_text,
        assistant_text=assistant_text,
        exchange_messages=exchange_messages,
        completion_source=completion_source,
        faithfulness_tags=faithfulness_tags,
        hashes=hashes,
        metadata=metadata,
    )


def apply_grouped_split(
    examples: Sequence[NormalizedExchange],
    *,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> list[NormalizedExchange]:
    """Assign train/val/test splits without splitting a group_id."""

    total = train_fraction + val_fraction + test_fraction
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")

    groups: dict[str, list[NormalizedExchange]] = defaultdict(list)
    for example in examples:
        groups[example.group_id].append(example)

    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)

    n_groups = len(group_ids)
    train_cut = round(n_groups * train_fraction)
    val_cut = train_cut + round(n_groups * val_fraction)
    split_by_group = {
        group_id: _split_name(index, train_cut, val_cut)
        for index, group_id in enumerate(group_ids)
    }

    assigned = []
    for example in examples:
        values = example.to_json_dict()
        values["split"] = split_by_group[example.group_id]
        assigned.append(normalize_record(values))
    return assigned


def validate_positive_class_gate(
    examples: Sequence[NormalizedExchange],
    *,
    min_non_refusal_positive_fraction: float = 0.8,
    min_positive_assistant_chars: int = 20,
) -> None:
    positives = [example for example in examples if example.label == LABEL_POSITIVE]
    if not positives:
        raise DatasetGateError("positive-class gate failed: no positive examples")

    missing_text = [
        example.example_id for example in positives if not example.assistant_text
    ]
    if missing_text:
        raise DatasetGateError(
            "positive-class gate failed: positive examples missing assistant_text "
            f"({len(missing_text)} examples)"
        )

    too_short = [
        example.example_id
        for example in positives
        if len(example.assistant_text.strip()) < min_positive_assistant_chars
    ]
    if too_short:
        raise DatasetGateError(
            "positive-class gate failed: positive assistant_text is too short "
            f"({len(too_short)} examples below {min_positive_assistant_chars} chars)"
        )

    refusal_like = [
        example.example_id
        for example in positives
        if is_refusal_like(example.assistant_text)
    ]
    non_refusal_fraction = 1.0 - (len(refusal_like) / len(positives))
    if non_refusal_fraction < min_non_refusal_positive_fraction:
        raise DatasetGateError(
            "positive-class gate failed: refusal-like positives dominate "
            f"({non_refusal_fraction:.3f} non-refusal fraction)"
        )


def validate_hard_negative_gate(examples: Sequence[NormalizedExchange]) -> None:
    hard_negatives = [
        example
        for example in examples
        if example.label == LABEL_NEGATIVE
        and (
            example.domain.lower() in {"cbrn", "science", "biosecurity"}
            or "hard_negative" in example.faithfulness_tags
        )
    ]
    if not hard_negatives:
        raise DatasetGateError("hard-negative gate failed: no hard negatives found")

    cbrn_like = [
        example
        for example in hard_negatives
        if contains_cbrn_terms(example.user_text)
        or contains_cbrn_terms(example.assistant_text)
    ]
    if not cbrn_like:
        raise DatasetGateError(
            "hard-negative gate failed: hard negatives lack CBRN/science terms"
        )


def validate_no_group_leakage(examples: Sequence[NormalizedExchange]) -> None:
    splits_by_group: dict[str, set[str]] = defaultdict(set)
    for example in examples:
        splits_by_group[example.group_id].add(example.split)

    leaked = {
        group_id: sorted(splits)
        for group_id, splits in splits_by_group.items()
        if len(splits) > 1
    }
    if leaked:
        first_group = next(iter(leaked))
        raise DatasetGateError(
            "group split gate failed: group_id appears in multiple splits "
            f"({first_group}: {leaked[first_group]})"
        )


def validate_on_policy_generation_gate(
    examples: Sequence[NormalizedExchange],
    *,
    allowed_completion_sources: Sequence[str],
    require_single_generator: bool,
    require_on_policy_model: bool,
) -> None:
    """Validate the matched-generator/on-policy design for generated datasets."""

    allowed_sources = set(allowed_completion_sources)
    source_violations = [
        example.example_id
        for example in examples
        if example.completion_source not in allowed_sources
    ]
    if source_violations:
        raise DatasetGateError(
            "on-policy gate failed: unexpected completion_source "
            f"({len(source_violations)} examples)"
        )

    generator_by_label: dict[int, set[str]] = defaultdict(set)
    protected_models: set[str] = set()
    missing_generator = []
    missing_protected = []
    off_policy = []
    for example in examples:
        generator_id = str(example.metadata.get("generator_model_id") or "")
        protected_id = str(example.metadata.get("protected_model_id") or "")
        if not generator_id:
            missing_generator.append(example.example_id)
            continue
        generator_by_label[example.label].add(generator_id)
        if require_on_policy_model:
            if not protected_id:
                missing_protected.append(example.example_id)
            elif protected_id != generator_id:
                off_policy.append(example.example_id)
            else:
                protected_models.add(protected_id)

    if missing_generator:
        raise DatasetGateError(
            "on-policy gate failed: examples missing metadata.generator_model_id "
            f"({len(missing_generator)} examples)"
        )
    if missing_protected:
        raise DatasetGateError(
            "on-policy gate failed: examples missing metadata.protected_model_id "
            f"({len(missing_protected)} examples)"
        )
    if off_policy:
        raise DatasetGateError(
            "on-policy gate failed: generator_model_id != protected_model_id "
            f"({len(off_policy)} examples)"
        )

    if require_single_generator:
        all_generators = set().union(*generator_by_label.values())
        if len(all_generators) != 1:
            raise DatasetGateError(
                "on-policy gate failed: expected a single generator across labels, "
                f"got {sorted(all_generators)}"
            )
        if set(generator_by_label) != {LABEL_NEGATIVE, LABEL_POSITIVE}:
            raise DatasetGateError(
                "on-policy gate failed: both labels must be present for generator check"
            )
        if generator_by_label[LABEL_NEGATIVE] != generator_by_label[LABEL_POSITIVE]:
            raise DatasetGateError(
                "on-policy gate failed: labels use different generators"
            )
    if (
        require_on_policy_model
        and require_single_generator
        and len(protected_models) > 1
    ):
        raise DatasetGateError(
            "on-policy gate failed: multiple protected-model analogues "
            f"{sorted(protected_models)}"
        )


def validate_length_balance_gate(
    examples: Sequence[NormalizedExchange],
    *,
    max_median_assistant_word_ratio: float,
    max_length_only_roc_auc: float,
) -> None:
    """Reject obvious assistant-length confounds after filtering."""

    if max_median_assistant_word_ratio < 1:
        raise ValueError("max_median_assistant_word_ratio must be >= 1")
    lengths_by_label: dict[int, list[int]] = defaultdict(list)
    for example in examples:
        lengths_by_label[example.label].append(len(example.assistant_text.split()))
    if set(lengths_by_label) != {LABEL_NEGATIVE, LABEL_POSITIVE}:
        raise DatasetGateError("length gate failed: both labels are required")

    negative_median = _median(lengths_by_label[LABEL_NEGATIVE])
    positive_median = _median(lengths_by_label[LABEL_POSITIVE])
    smaller = max(1.0, min(negative_median, positive_median))
    larger = max(negative_median, positive_median)
    ratio = larger / smaller
    if ratio > max_median_assistant_word_ratio:
        raise DatasetGateError(
            "length gate failed: assistant length medians differ too much "
            f"(ratio={ratio:.3f}, max={max_median_assistant_word_ratio:.3f})"
        )
    length_auc = length_only_roc_auc(examples)
    if length_auc > max_length_only_roc_auc:
        raise DatasetGateError(
            "length gate failed: assistant length alone is label-separable "
            f"(roc_auc={length_auc:.3f}, max={max_length_only_roc_auc:.3f})"
        )


def validate_dataset_gates(
    examples: Sequence[NormalizedExchange],
    *,
    on_policy_config: Mapping[str, Any] | None = None,
    length_balance_config: Mapping[str, Any] | None = None,
) -> None:
    validate_positive_class_gate(examples)
    validate_hard_negative_gate(examples)
    validate_no_group_leakage(examples)
    if on_policy_config and on_policy_config.get("enabled", False):
        validate_on_policy_generation_gate(
            examples,
            allowed_completion_sources=on_policy_config[
                "allowed_completion_sources"
            ],
            require_single_generator=bool(
                on_policy_config.get("require_single_generator", True)
            ),
            require_on_policy_model=bool(
                on_policy_config.get("require_on_policy_model", True)
            ),
        )
    if length_balance_config and length_balance_config.get("enabled", False):
        validate_length_balance_gate(
            examples,
            max_median_assistant_word_ratio=float(
                length_balance_config["max_median_assistant_word_ratio"]
            ),
            max_length_only_roc_auc=float(
                length_balance_config["max_length_only_roc_auc"]
            ),
        )


def dataset_metadata(
    examples: Sequence[NormalizedExchange],
    *,
    config_path: str | None,
    source_inspection: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    labels = Counter(example.label for example in examples)
    splits = Counter(example.split for example in examples)
    domains = Counter(example.domain for example in examples)
    sources = Counter(example.source_dataset for example in examples)
    completion_sources = Counter(example.completion_source for example in examples)
    generators = Counter(
        str(example.metadata.get("generator_model_id"))
        for example in examples
        if example.metadata.get("generator_model_id")
    )
    protected_models = Counter(
        str(example.metadata.get("protected_model_id"))
        for example in examples
        if example.metadata.get("protected_model_id")
    )
    positive_count = labels[LABEL_POSITIVE]
    refusal_positive_count = sum(
        example.label == LABEL_POSITIVE and is_refusal_like(example.assistant_text)
        for example in examples
    )

    return {
        "config_path": config_path,
        "num_examples": len(examples),
        "label_counts": dict(labels),
        "split_counts": dict(splits),
        "domain_counts": dict(domains),
        "source_dataset_counts": dict(sources),
        "completion_source_counts": dict(completion_sources),
        "generator_model_counts": dict(generators),
        "protected_model_counts": dict(protected_models),
        "assistant_length_by_label": assistant_length_summary(examples),
        "assistant_length_only_roc_auc": length_only_roc_auc(examples),
        "positive_refusal_like_count": refusal_positive_count,
        "positive_non_refusal_fraction": (
            None
            if positive_count == 0
            else 1.0 - (refusal_positive_count / positive_count)
        ),
        "group_count": len({example.group_id for example in examples}),
        "source_inspection": list(source_inspection),
        "gates": {
            "positive_class": "passed",
            "hard_negative": "passed",
            "group_leakage": "passed",
        },
    }


def write_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def is_refusal_like(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def contains_cbrn_terms(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in CBRN_TERMS)


def assistant_length_summary(
    examples: Sequence[NormalizedExchange],
) -> dict[str, dict[str, float]]:
    summary = {}
    for label in [LABEL_NEGATIVE, LABEL_POSITIVE]:
        lengths = [
            len(example.assistant_text.split())
            for example in examples
            if example.label == label
        ]
        if lengths:
            summary[str(label)] = {
                "count": len(lengths),
                "median_words": _median(lengths),
                "mean_words": sum(lengths) / len(lengths),
                "min_words": float(min(lengths)),
                "max_words": float(max(lengths)),
            }
    return summary


def length_only_roc_auc(examples: Sequence[NormalizedExchange]) -> float:
    labels = [example.label for example in examples]
    lengths = [len(example.assistant_text.split()) for example in examples]
    return roc_auc(labels, lengths)


def normalize_clearharm_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    source_subset: str,
    source_split: str,
    assistant_field: str = "proxy_gen_target",
) -> NormalizedExchange | None:
    """Normalize a ClearHarm row if it is a harmful candidate exchange."""

    if int(row.get("clf_label", -1)) != LABEL_POSITIVE:
        return None

    content = row.get("content") or []
    if not content:
        return None

    user_text = str(content[0]).strip()
    assistant_text = str(row.get(assistant_field) or "").strip()
    if not user_text or not assistant_text:
        return None

    faithfulness_tags = ["clearharm_positive_candidate"]
    if assistant_field == "proxy_gen_target":
        faithfulness_tags.append("clearharm_proxy_gen_target")
    if len(assistant_text) < 20:
        faithfulness_tags.append("positive_prefill_only")

    return normalize_record(
        {
            "example_id": f"clearharm-{source_subset}-{source_split}-{row_index}",
            "group_id": stable_text_hash(user_text)[:16],
            "split": "unassigned",
            "label": LABEL_POSITIVE,
            "domain": "cbrn" if contains_cbrn_terms(user_text) else "harmful",
            "source_dataset": "AlignmentResearch/ClearHarm",
            "source_subset": source_subset,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "completion_source": "public",
            "faithfulness_tags": faithfulness_tags,
            "metadata": {
                "source_split": source_split,
                "assistant_field": assistant_field,
                "proxy_clf_label": row.get("proxy_clf_label"),
            },
        }
    )


def normalize_wildchat_negative_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    require_cbrn_terms: bool = True,
) -> NormalizedExchange | None:
    """Normalize a benign WildChat conversation as a negative candidate."""

    if bool(row.get("toxic")) or bool(row.get("redacted")):
        return None
    if _moderation_flagged(row.get("openai_moderation") or []):
        return None

    messages = row.get("conversation") or []
    user_text, assistant_text = _first_user_assistant_pair(messages)
    if not user_text or not assistant_text:
        return None
    if require_cbrn_terms and not (
        contains_cbrn_terms(user_text) or contains_cbrn_terms(assistant_text)
    ):
        return None

    return normalize_record(
        {
            "example_id": f"wildchat-{row.get('conversation_id', row_index)}",
            "group_id": str(row.get("conversation_id") or stable_text_hash(user_text)),
            "split": "unassigned",
            "label": LABEL_NEGATIVE,
            "domain": "science" if contains_cbrn_terms(user_text) else "general",
            "source_dataset": "allenai/WildChat",
            "source_subset": "train",
            "user_text": user_text,
            "assistant_text": assistant_text,
            "completion_source": "public",
            "faithfulness_tags": ["hard_negative", "wildchat_non_toxic"],
            "metadata": {
                "model": row.get("model"),
                "turn": row.get("turn"),
                "language": row.get("language"),
            },
        }
    )


def normalize_harmbench_prompt_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    source_subset: str,
    source_split: str,
) -> GenerationPrompt | None:
    """Normalize a HarmBench prompt-only row for later completion generation."""

    user_text = str(row.get("prompt") or "").strip()
    if not user_text:
        return None

    context = str(row.get("context") or "").strip()
    category = str(row.get("category") or row.get("tags") or "unknown").strip()
    prompt_id = f"harmbench-{source_subset}-{source_split}-{row_index}"
    prompt_payload = "\n".join(part for part in [context, user_text] if part)
    return GenerationPrompt(
        prompt_id=prompt_id,
        group_id=stable_text_hash(prompt_payload)[:16],
        source_dataset="walledai/HarmBench",
        source_subset=source_subset,
        source_split=source_split,
        domain=category or "unknown",
        user_text=user_text,
        context=context,
        faithfulness_tags=[
            "harmbench_prompt_only",
            "requires_generated_uncensored_completion",
        ],
        hashes={
            "user_text_sha256": stable_text_hash(user_text),
            "context_sha256": stable_text_hash(context),
            "prompt_payload_sha256": stable_text_hash(prompt_payload),
        },
        metadata={"category": category, "row_index": row_index},
    )


def normalize_clearharm_prompt_row(
    row: Mapping[str, Any],
    *,
    row_index: int,
    source_subset: str,
    source_split: str,
) -> GenerationPrompt | None:
    """Normalize a ClearHarm prompt-only row for later completion generation."""

    if int(row.get("clf_label", -1)) != LABEL_POSITIVE:
        return None
    content = row.get("content") or []
    if not content:
        return None
    user_text = str(content[0]).strip()
    if not user_text:
        return None

    prompt_hash = stable_text_hash(user_text)
    return GenerationPrompt(
        prompt_id=f"clearharm-{source_subset}-{source_split}-{row_index}",
        group_id=prompt_hash[:16],
        source_dataset="AlignmentResearch/ClearHarm",
        source_subset=source_subset,
        source_split=source_split,
        domain=CLEARHARM_CBRN_DOMAIN,
        user_text=user_text,
        context="",
        faithfulness_tags=[
            "clearharm_cbrn_positive_prompt",
            "requires_generated_uncensored_completion",
        ],
        hashes={
            "user_text_sha256": prompt_hash,
            "prompt_payload_sha256": prompt_hash,
        },
        metadata={
            "row_index": row_index,
            "proxy_clf_label": row.get("proxy_clf_label"),
            "source_unique_prompt_hash": prompt_hash[:16],
            "source_replicate_count_hint": 40 if source_subset == "rep40" else None,
        },
    )


def _normalize_messages(
    raw_messages: Any, user_text: str, assistant_text: str
) -> list[dict[str, str]]:
    if raw_messages is None:
        messages = [{"role": "user", "content": user_text}]
        if assistant_text:
            messages.append({"role": "assistant", "content": assistant_text})
        return messages

    messages = []
    for raw_message in raw_messages:
        role = str(raw_message["role"])
        content = str(raw_message["content"])
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"unsupported message role: {role!r}")
        messages.append({"role": role, "content": content})
    return messages


def _default_example_id(record: Mapping[str, Any]) -> str:
    payload = {
        "source_dataset": record.get("source_dataset"),
        "source_subset": record.get("source_subset"),
        "user_text": record.get("user_text"),
        "assistant_text": record.get("assistant_text"),
        "label": record.get("label"),
    }
    return stable_text_hash(json.dumps(payload, sort_keys=True))[:24]


def _split_name(index: int, train_cut: int, val_cut: int) -> str:
    if index < train_cut:
        return "train"
    if index < val_cut:
        return "val"
    return "test"


def _median(values: Sequence[int]) -> float:
    if not values:
        raise ValueError("cannot compute median of empty values")
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _moderation_flagged(moderation_rows: Sequence[Mapping[str, Any]]) -> bool:
    for moderation in moderation_rows:
        if bool(moderation.get("flagged")):
            return True
        categories = moderation.get("categories") or {}
        if any(bool(value) for value in categories.values()):
            return True
    return False


def _first_user_assistant_pair(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[str | None, str | None]:
    pending_user: str | None = None
    for message in messages:
        role = str(message.get("role") or "")
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            return pending_user, content
    return None, None
