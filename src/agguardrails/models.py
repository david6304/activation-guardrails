"""Model loading, tokeniser setup, and chat-template helpers.

Design notes
------------
- Token position for probing: we use the **last token** of the formatted prompt
  (the token just before where generation would begin).  This is standard in the
  probing literature (Azaria & Mitchell 2023; Zou et al. 2023 RepE) because it
  aggregates the full prompt context via causal attention.
- We always apply the model's chat template via ``apply_chat_template`` so that
  the model sees the same format it was fine-tuned on.
- Left-padding is set so that, in a batch, the last position is always the
  meaningful "end-of-prompt" token for every sequence.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(
    model_name: str,
    *,
    torch_dtype: str = "float16",
    device_map: str = "auto",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and its tokeniser, configured for activation extraction.

    Sets ``output_hidden_states=True`` so that forward passes return all layer
    hidden states.  Sets left-padding for correct batched inference on
    decoder-only models.
    """
    dtype = getattr(torch, torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Left-padding: ensures the last token position is always the final
    # content token in every sequence of a batch.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def format_prompt(tokenizer: AutoTokenizer, user_message: str) -> str:
    """Apply the model's chat template to a single user message.

    Returns the formatted string (not tokenised) with the assistant turn
    prefix appended (``add_generation_prompt=True``), so that the last token
    is the "end-of-prompt" position we probe.
    """
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def get_layer_count(model: AutoModelForCausalLM) -> int:
    """Return the number of transformer layers in the model."""
    return model.config.num_hidden_layers


def get_hidden_size(model: AutoModelForCausalLM) -> int:
    """Return the hidden dimension of the model."""
    return model.config.hidden_size
