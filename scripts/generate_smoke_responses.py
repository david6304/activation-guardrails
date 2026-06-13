#!/usr/bin/env python
"""Generate protected-model responses for the accepted smoke manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from agguardrails.response_generation import (
    DEFAULT_BASE_SEED,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DecodingSettings,
    TransformersGemmaBackend,
    generate_responses,
    inspect_resume,
    load_manifest,
)

DEFAULT_MODEL_PATH = "/home/s2296274/models/gemma-3-4b-it-heretic"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows, manifest_sha256 = load_manifest(args.manifest)
    tokenizer_path = args.tokenizer_path or args.model_path
    settings = DecodingSettings(
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    _, resume = inspect_resume(
        rows,
        output_path=args.output,
        manifest_sha256=manifest_sha256,
        base_seed=args.base_seed,
        settings=settings,
        model_path=args.model_path,
        tokenizer_path=tokenizer_path,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "classification": "exploratory_non_reportable_smoke",
                    "dry_run": True,
                    "manifest": str(args.manifest),
                    "output": str(args.output),
                    "model_path": args.model_path,
                    "tokenizer_path": tokenizer_path,
                    "decoding": settings.as_dict(),
                    **resume,
                },
                sort_keys=True,
            )
        )
        return 0

    backend = TransformersGemmaBackend(args.model_path, tokenizer_path)
    summary = generate_responses(
        rows,
        output_path=args.output,
        manifest_sha256=manifest_sha256,
        base_seed=args.base_seed,
        settings=settings,
        model_path=args.model_path,
        tokenizer_path=tokenizer_path,
        backend=backend,
    )
    print(json.dumps(summary, sort_keys=True))
    return int(summary["remaining_after_run"] != 0)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/wildjailbreak_smoke_manifest.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/responses/wildjailbreak_smoke_responses.jsonl"),
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--tokenizer-path",
        help="Defaults to --model-path; this path is loaded as a Gemma 3 tokenizer.",
    )
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and resume state without loading the model.",
    )
    args = parser.parse_args(argv)
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
