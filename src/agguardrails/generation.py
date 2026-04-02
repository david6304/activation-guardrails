"""Batched response generation with checkpoint support.

Generates model responses to a list of prompts and writes them to a JSONL
checkpoint file.  Resumable: already-completed example IDs are skipped on
re-invocation, so a failed cluster job can pick up where it left off.

Output record schema:
    {
        "example_id": str,
        "prompt": str,
        "response": str,
        "source": str,
        "source_label": str,
        "label": int,   # source label (harmful=1, benign=0) — NOT refusal label
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from agguardrails.data import PromptExample
from agguardrails.io import write_jsonl
from agguardrails.models import format_prompt
from agguardrails.utils import batched


def _load_completed_ids(checkpoint_path: Path) -> set[str]:
    """Return the set of example_ids already written to *checkpoint_path*."""
    if not checkpoint_path.exists():
        return set()
    completed: set[str] = set()
    with checkpoint_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    completed.add(record["example_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return completed


def generate_responses(
    model,
    tokenizer,
    examples: list[PromptExample],
    *,
    max_new_tokens: int,
    batch_size: int,
    temperature: float,
    checkpoint_path: str | Path,
    checkpoint_every: int = 100,
) -> list[dict]:
    """Generate model responses for *examples*, writing checkpoints along the way.

    Args:
        model: Loaded causal LM (from ``load_model_and_tokenizer``).
        tokenizer: Matching tokeniser with left-padding configured.
        examples: Prompts to generate responses for.
        max_new_tokens: Maximum new tokens per response.
        batch_size: Number of prompts per forward pass.
        temperature: Sampling temperature.  Use 0.0 for greedy decoding.
        checkpoint_path: JSONL file for incremental output.  Existing records
            are loaded on entry; new records are appended.
        checkpoint_every: Flush to disk after this many new completions.

    Returns:
        List of response dicts for all examples (loaded + newly generated).
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    completed_ids = _load_completed_ids(checkpoint_path)
    pending = [e for e in examples if e.example_id not in completed_ids]

    # Load already-completed records into results list.
    results: list[dict] = []
    if checkpoint_path.exists():
        with checkpoint_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    if not pending:
        print(f"All {len(results)} responses already completed.", flush=True)
        return results

    print(
        f"Generating {len(pending)} responses "
        f"({len(completed_ids)} already done).",
        flush=True,
    )

    do_sample = temperature > 0.0
    new_records: list[dict] = []
    n_batches = (len(pending) + batch_size - 1) // batch_size

    with checkpoint_path.open("a") as checkpoint_file:
        for batch in tqdm(
            batched(pending, batch_size),
            total=n_batches,
            desc="generating",
            file=sys.stdout,
            disable=False,
        ):
            formatted = [format_prompt(tokenizer, e.prompt) for e in batch]
            encoded = tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)
            input_lengths = attention_mask.sum(dim=1)

            generate_kwargs: dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if do_sample:
                generate_kwargs["temperature"] = temperature

            with torch.inference_mode():
                output_ids = model.generate(**generate_kwargs)

            for i, example in enumerate(batch):
                # Slice only the newly generated tokens.
                new_tokens = output_ids[i, input_lengths[i]:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

                record = {
                    "example_id": example.example_id,
                    "prompt": example.prompt,
                    "response": response,
                    "source": example.source,
                    "source_label": example.source_label,
                    "label": example.label,
                }
                new_records.append(record)
                results.append(record)
                checkpoint_file.write(json.dumps(record, ensure_ascii=False) + "\n")

            if len(new_records) % checkpoint_every < batch_size:
                checkpoint_file.flush()

    print(f"Generation complete. Total: {len(results)} responses.", flush=True)
    return results
