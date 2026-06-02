#!/usr/bin/env python
"""Generate on-policy completions for a CC++ prompt manifest.

Reads a prompt-only manifest (ClearHarm CBRN positives or matched dual-use benign
negatives), completes each prompt with the selected refusal-ablated protected-
model analogue, and writes normalized exchanges ready for ``build_dataset.py``.

The same model id is recorded as ``generator_model_id`` and ``protected_model_id``
so the on-policy gate passes. Raw completions are written only to the output file;
stdout carries aggregate counts only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agguardrails.ccpp_data import (
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    read_jsonl,
)
from agguardrails.ccpp_generation import (
    DecodingParams,
    build_generator,
    iter_generate_exchanges,
    read_generation_prompts,
)
from agguardrails.progress import ProgressLogger, configure_logging

_LABEL_BY_NAME = {
    "auto": None,
    "positive": LABEL_POSITIVE,
    "negative": LABEL_NEGATIVE,
}


def main() -> int:
    args = parse_args()
    logger = configure_logging()

    rows = read_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    prompts = read_generation_prompts(rows)
    if not prompts:
        raise SystemExit(f"no prompts found in {args.manifest}")

    protected_model_id = args.protected_model_id or args.model_id
    decoding = DecodingParams(
        backend=args.backend,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.greedy,
        seed=args.seed,
    )
    logger.info("loading generator (%s) model=%s", args.backend, args.model_id)
    generator = build_generator(
        args.backend, model_id=args.model_id, decoding=decoding
    )
    logger.info(
        "generating %d completions (batch_size=%d) -> %s",
        len(prompts),
        args.batch_size,
        args.output,
    )

    progress = ProgressLogger(len(prompts), logger=logger, label="generated")
    label_counts: dict[int, int] = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for exchange in iter_generate_exchanges(
            prompts,
            generator,
            generator_model_id=args.model_id,
            protected_model_id=protected_model_id,
            decoding=decoding,
            label=_LABEL_BY_NAME[args.label],
            progress=progress,
            batch_size=args.batch_size,
        ):
            handle.write(json.dumps(exchange.to_json_dict(), sort_keys=True) + "\n")
            handle.flush()
            label_counts[exchange.label] = label_counts.get(exchange.label, 0) + 1

    num_exchanges = sum(label_counts.values())
    logger.info("done: %d exchanges written to %s", num_exchanges, args.output)
    print(
        json.dumps(
            {
                "num_exchanges": num_exchanges,
                "label_counts": label_counts,
                "model_id": args.model_id,
                "on_policy": args.model_id == protected_model_id,
                "output": str(args.output),
            }
        )
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-id",
        required=True,
        help="Refusal-ablated protected-model analogue (local path or HF id).",
    )
    parser.add_argument(
        "--protected-model-id",
        default=None,
        help="Defaults to --model-id to keep the run on-policy.",
    )
    parser.add_argument(
        "--backend", choices=["transformers", "mock"], default="transformers"
    )
    parser.add_argument(
        "--label", choices=list(_LABEL_BY_NAME), default="auto"
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Prompts per generate() call. Higher = faster on big GPUs; "
        "lower if you hit OOM.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
