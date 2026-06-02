"""On-policy completion generation for the CC++ vertical slice.

Turns prompt-only manifests (ClearHarm CBRN positives, matched dual-use benign
negatives) into normalized exchanges by completing each prompt with the selected
refusal-ablated protected-model analogue. The same model id is recorded as both
``generator_model_id`` and ``protected_model_id`` so the on-policy gate in
``agguardrails.ccpp_data`` passes.

The module exposes a backend-agnostic generator protocol: a ``MockGenerator`` for
CPU tests and a ``TransformersGenerator`` for the GPU run on the cluster.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from agguardrails.ccpp_data import (
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    GenerationPrompt,
    NormalizedExchange,
    normalize_record,
)

COMPLETION_SOURCE = "generated_uncensored"
BENIGN_PROMPT_SOURCE_TYPE = "matched_dual_use_benign"


@dataclass(frozen=True)
class DecodingParams:
    """Decoding configuration recorded in each exchange's metadata."""

    backend: str
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True
    seed: int = 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "do_sample": self.do_sample,
            "seed": self.seed,
        }


class Generator(Protocol):
    """Produces an assistant completion for one user turn."""

    def generate(self, user_text: str, context: str = "") -> str: ...


def infer_label(prompt: GenerationPrompt) -> int:
    """Infer the binary label of a prompt from its provenance metadata."""

    source_type = str(prompt.metadata.get("prompt_source_type") or "")
    if source_type == BENIGN_PROMPT_SOURCE_TYPE:
        return LABEL_NEGATIVE
    if prompt.source_dataset == "generated_matched_benign":
        return LABEL_NEGATIVE
    if prompt.source_dataset in {
        "AlignmentResearch/ClearHarm",
        "walledai/HarmBench",
    }:
        return LABEL_POSITIVE
    if any("benign" in tag for tag in prompt.faithfulness_tags):
        return LABEL_NEGATIVE
    if any("positive" in tag for tag in prompt.faithfulness_tags):
        return LABEL_POSITIVE
    raise ValueError(
        f"cannot infer label for prompt {prompt.prompt_id!r}; pass an explicit label"
    )


def _compose_user_turn(user_text: str, context: str) -> str:
    context = (context or "").strip()
    if context:
        return f"{context}\n\n{user_text.strip()}"
    return user_text.strip()


def build_exchange_record(
    prompt: GenerationPrompt,
    assistant_text: str,
    *,
    label: int,
    generator_model_id: str,
    protected_model_id: str,
    decoding: DecodingParams,
) -> NormalizedExchange:
    """Assemble one normalized exchange from a prompt and its completion."""

    if label not in {LABEL_NEGATIVE, LABEL_POSITIVE}:
        raise ValueError(f"label must be 0 or 1, got {label!r}")

    user_text = _compose_user_turn(prompt.user_text, prompt.context)
    tags = [
        tag
        for tag in prompt.faithfulness_tags
        if tag != "requires_generated_uncensored_completion"
    ]
    tags += ["generated_uncensored_completion", "on_policy_protected_model"]
    if label == LABEL_NEGATIVE:
        tags.append("hard_negative")

    metadata = dict(prompt.metadata)
    metadata.update(
        {
            "generator_model_id": generator_model_id,
            "protected_model_id": protected_model_id,
            "decoding": decoding.to_metadata(),
            "source_prompt_id": prompt.prompt_id,
            "source_split": prompt.source_split,
            "label_source": "inferred_from_manifest",
        }
    )

    return normalize_record(
        {
            "example_id": f"{prompt.prompt_id}-gen",
            "group_id": prompt.group_id,
            "split": "unassigned",
            "label": label,
            "domain": prompt.domain,
            "source_dataset": prompt.source_dataset,
            "source_subset": prompt.source_subset,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "completion_source": COMPLETION_SOURCE,
            "faithfulness_tags": tags,
            "hashes": dict(prompt.hashes),
            "metadata": metadata,
        }
    )


def generate_exchanges(
    prompts: Sequence[GenerationPrompt],
    generator: Generator,
    *,
    generator_model_id: str,
    protected_model_id: str,
    decoding: DecodingParams,
    label: int | None = None,
) -> list[NormalizedExchange]:
    """Complete every prompt and return normalized exchanges."""

    exchanges = []
    for prompt in prompts:
        prompt_label = label if label is not None else infer_label(prompt)
        assistant_text = generator.generate(prompt.user_text, prompt.context).strip()
        exchanges.append(
            build_exchange_record(
                prompt,
                assistant_text,
                label=prompt_label,
                generator_model_id=generator_model_id,
                protected_model_id=protected_model_id,
                decoding=decoding,
            )
        )
    return exchanges


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

_MOCK_SENTENCES = (
    "At a high level the key considerations are documentation and oversight.",
    "Established standards describe how this is monitored and validated.",
    "Reviewers focus on safety, governance, and clear reporting lines.",
    "Recognised guidance emphasises training and competency assessment.",
    "The relevant controls are layered and independently verified.",
    "Outcomes are tracked so that corrective actions can be confirmed.",
)


class MockGenerator:
    """Deterministic CPU stub used for tests and dry runs.

    Produces non-refusal, fixed-length placeholder text whose length does not
    correlate with the label, so the length-balance gate is not tripped. It does
    not emit any harmful or procedural content.
    """

    def __init__(self, seed: int = 0, num_sentences: int = 4) -> None:
        self.seed = seed
        self.num_sentences = num_sentences

    def generate(self, user_text: str, context: str = "") -> str:
        rng = random.Random(f"{self.seed}:{user_text}")
        sentences = [rng.choice(_MOCK_SENTENCES) for _ in range(self.num_sentences)]
        return " ".join(sentences)


class TransformersGenerator:
    """Hugging Face Transformers backend for the GPU run.

    Loads the refusal-ablated protected-model analogue once and completes each
    prompt with its chat template. Imports are lazy so CPU/test paths never need
    torch or transformers installed.
    """

    def __init__(
        self,
        model_id: str,
        *,
        decoding: DecodingParams,
        device_map: str = "auto",
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.decoding = decoding
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=device_map,
            dtype=getattr(torch, dtype),
        )
        self.model.eval()

    def generate(self, user_text: str, context: str = "") -> str:
        torch = self._torch
        torch.manual_seed(self.decoding.seed)
        content = _compose_user_turn(user_text, context)
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                inputs,
                max_new_tokens=self.decoding.max_new_tokens,
                do_sample=self.decoding.do_sample,
                temperature=self.decoding.temperature,
                top_p=self.decoding.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def build_generator(
    backend: str,
    *,
    model_id: str,
    decoding: DecodingParams,
) -> Generator:
    if backend == "mock":
        return MockGenerator(seed=decoding.seed)
    if backend == "transformers":
        return TransformersGenerator(model_id, decoding=decoding)
    raise ValueError(f"unsupported backend: {backend!r}")


def read_generation_prompts(
    rows: Sequence[Mapping[str, Any]],
) -> list[GenerationPrompt]:
    """Rehydrate ``GenerationPrompt`` objects from manifest JSONL rows."""

    prompts = []
    for row in rows:
        prompts.append(
            GenerationPrompt(
                prompt_id=str(row["prompt_id"]),
                group_id=str(row["group_id"]),
                source_dataset=str(row["source_dataset"]),
                source_subset=str(row.get("source_subset", "")),
                source_split=str(row.get("source_split", "")),
                domain=str(row.get("domain", "unknown")),
                user_text=str(row["user_text"]),
                context=str(row.get("context", "")),
                faithfulness_tags=list(row.get("faithfulness_tags") or []),
                hashes=dict(row.get("hashes") or {}),
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return prompts
