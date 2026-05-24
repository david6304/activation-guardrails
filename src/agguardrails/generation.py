"""Minimal model-generation helpers for response-cache construction."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any


def make_chat_messages(prompt: str) -> list[dict[str, str]]:
    """Build the prompt-only chat exchange cached before activation extraction."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    return [{"role": "user", "content": prompt}]


def real_generation_metadata(config: dict[str, Any]) -> dict[str, Any]:
    """Return non-secret runtime choices that affect real model generation."""
    generation_config = config.get("generation", {})
    runtime = dict(generation_config.get("runtime", {}))
    return {
        "backend": runtime.get("backend", "transformers"),
        "device_map": runtime.get("device_map", "auto"),
        "torch_dtype": runtime.get("torch_dtype", "auto"),
    }


class TransformersChatGenerator:
    """Lazy-loaded causal-LM chat generator.

    Keep this small and explicit; expensive model loading happens only through
    ``from_config`` and only after the CLI has checked the real-generation flag.
    """

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        generation_params: dict[str, Any],
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.generation_params = generation_params

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TransformersChatGenerator":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_config = dict(config.get("model", {}))
        tokenizer_config = dict(config.get("tokenizer") or model_config)
        generation_config = config.get("generation", {})
        runtime = dict(generation_config.get("runtime", {}))
        backend = runtime.get("backend", "transformers")
        if backend != "transformers":
            raise ValueError(f"Unsupported generation runtime backend: {backend}")

        model_id = model_config.get("id")
        tokenizer_id = tokenizer_config.get("id")
        if not model_id:
            raise ValueError("model.id is required for real generation")
        if not tokenizer_id:
            raise ValueError("tokenizer.id is required for real generation")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            revision=tokenizer_config.get("revision"),
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_config.get("revision"),
            device_map=runtime.get("device_map", "auto"),
            torch_dtype=runtime.get("torch_dtype", "auto"),
        )
        return cls(
            model=model,
            tokenizer=tokenizer,
            generation_params=dict(generation_config.get("params", {})),
        )

    def generate_response(self, prompt: str) -> str:
        import torch

        messages = make_chat_messages(prompt)
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            tokenize=True,
        )
        input_ids = _move_to_model_device(input_ids, self.model)
        with torch.inference_mode():
            output_ids = self.model.generate(input_ids, **self.generation_params)
        response_ids = output_ids[0, input_ids.shape[-1] :]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()


def generate_real_responses(
    examples: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> Iterator[tuple[dict[str, Any], list[dict[str, str]], str]]:
    """Yield real responses for selected examples.

    The empty-selection case intentionally returns without loading a model so
    resume/debug commands can still update metadata cheaply.
    """
    if not examples:
        return

    generator = TransformersChatGenerator.from_config(config)
    for example in examples:
        prompt = str(example["prompt"])
        messages = make_chat_messages(prompt)
        yield example, messages, generator.generate_response(prompt)


def _move_to_model_device(input_ids: Any, model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is None and hasattr(model, "get_input_embeddings"):
        embeddings = model.get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        device = getattr(weight, "device", None)
    if device is None:
        return input_ids
    return input_ids.to(device)
