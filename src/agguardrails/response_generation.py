"""Generate resumable responses for the approved non-reportable smoke manifest."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from agguardrails.wildjailbreak_manifest import validate_manifest

RESPONSE_SCHEMA_VERSION = 1
CLASSIFICATION = "exploratory_non_reportable_smoke"
DEFAULT_BASE_SEED = 20260612
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_NEW_TOKENS = 4096
SEED_ALGORITHM = "sha256(base_seed + NUL + example_id), first 63 bits"


class ResponseGenerationError(ValueError):
    """Raised when response inputs or resume artifacts violate the contract."""


@dataclass(frozen=True)
class DecodingSettings:
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS

    def as_dict(self) -> dict[str, Any]:
        return {
            "do_sample": True,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "num_return_sequences": 1,
        }


@dataclass(frozen=True)
class GeneratedResponse:
    text: str
    prompt_token_count: int
    response_token_count: int
    termination_reason: str


class GenerationBackend(Protocol):
    model_identity: Mapping[str, Any]
    tokenizer_identity: Mapping[str, Any]

    def generate(
        self, prompt: str, *, seed: int, settings: DecodingSettings
    ) -> GeneratedResponse: ...


def load_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    """Load and validate the accepted manifest while retaining its byte identity."""

    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ResponseGenerationError(
                f"manifest line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(row, dict):
            raise ResponseGenerationError(
                f"manifest line {line_number} is not a JSON object"
            )
        rows.append(row)
    validate_manifest(rows)
    return rows, hashlib.sha256(raw).hexdigest()


def derive_example_seed(base_seed: int, example_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}\0{example_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def inspect_resume(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    manifest_sha256: str,
    base_seed: int,
    settings: DecodingSettings,
    model_path: str,
    tokenizer_path: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Validate current records and report completed, retryable, and pending work."""

    records = _load_existing_records(output_path)
    manifest_ids = {row["example_id"] for row in rows}
    unknown = records.keys() - manifest_ids
    if unknown:
        raise ResponseGenerationError(
            f"output contains {len(unknown)} example IDs outside the manifest"
        )

    expected_decoding = settings.as_dict()
    for example_id, record in records.items():
        expected = {
            "response_schema_version": RESPONSE_SCHEMA_VERSION,
            "classification": CLASSIFICATION,
            "example_id": example_id,
            "manifest_sha256": manifest_sha256,
            "base_seed": base_seed,
            "seed": derive_example_seed(base_seed, example_id),
            "seed_algorithm": SEED_ALGORITHM,
            "decoding": expected_decoding,
            "requested_model_path": model_path,
            "requested_tokenizer_path": tokenizer_path,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            raise ResponseGenerationError(
                f"output record for {example_id} is incompatible with this run"
            )
        status = record.get("status")
        if status == "success":
            valid_success = (
                isinstance(record.get("response"), str)
                and _is_nonnegative_int(record.get("prompt_token_count"))
                and _is_nonnegative_int(record.get("response_token_count"))
                and record.get("termination_reason") in {"eos", "length_cap", "other"}
            )
            if not valid_success:
                raise ResponseGenerationError(
                    f"successful output record for {example_id} is incomplete"
                )
        elif status == "failure":
            if "response" in record or not isinstance(record.get("failure"), dict):
                raise ResponseGenerationError(
                    f"failed output record for {example_id} is invalid"
                )
        else:
            raise ResponseGenerationError(
                f"output record for {example_id} has invalid status"
            )
        if (
            not isinstance(record.get("runtime_seconds"), (int, float))
            or not isinstance(record.get("model"), dict)
            or not isinstance(record.get("tokenizer"), dict)
        ):
            raise ResponseGenerationError(
                f"output record for {example_id} has incomplete provenance"
            )

    completed = sum(record["status"] == "success" for record in records.values())
    failures = len(records) - completed
    return records, {
        "manifest_examples": len(rows),
        "completed": completed,
        "retryable_failures": failures,
        "pending": len(rows) - len(records),
    }


def generate_responses(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    manifest_sha256: str,
    base_seed: int,
    settings: DecodingSettings,
    model_path: str,
    tokenizer_path: str,
    backend: GenerationBackend,
    log: Callable[[str], None] = print,
) -> dict[str, int]:
    """Generate one current record per example, preserving completed successes."""

    records, before = inspect_resume(
        rows,
        output_path=output_path,
        manifest_sha256=manifest_sha256,
        base_seed=base_seed,
        settings=settings,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    expected_model = dict(backend.model_identity)
    expected_tokenizer = dict(backend.tokenizer_identity)
    for example_id, record in records.items():
        if (
            record["model"] != expected_model
            or record["tokenizer"] != expected_tokenizer
        ):
            raise ResponseGenerationError(
                f"output record for {example_id} has a different backend identity"
            )
    generated = failed = skipped = 0
    for row in rows:
        example_id = row["example_id"]
        previous = records.get(example_id)
        if previous and previous["status"] == "success":
            skipped += 1
            continue

        seed = derive_example_seed(base_seed, example_id)
        started = time.perf_counter()
        try:
            result = backend.generate(row["prompt"], seed=seed, settings=settings)
            record = _base_record(
                row,
                manifest_sha256=manifest_sha256,
                base_seed=base_seed,
                seed=seed,
                settings=settings,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                backend=backend,
                runtime_seconds=time.perf_counter() - started,
            )
            record.update(
                {
                    "status": "success",
                    "response": result.text,
                    "prompt_token_count": result.prompt_token_count,
                    "response_token_count": result.response_token_count,
                    "termination_reason": result.termination_reason,
                }
            )
            generated += 1
        except Exception as exc:
            record = _base_record(
                row,
                manifest_sha256=manifest_sha256,
                base_seed=base_seed,
                seed=seed,
                settings=settings,
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                backend=backend,
                runtime_seconds=time.perf_counter() - started,
            )
            record.update(
                {
                    "status": "failure",
                    "failure": {
                        "stage": "generation",
                        "error_type": type(exc).__name__,
                    },
                    "prompt_token_count": None,
                    "response_token_count": None,
                    "termination_reason": None,
                }
            )
            failed += 1

        if previous is None:
            _append_record(output_path, record)
        else:
            records[example_id] = record
            _write_records_atomic(output_path, rows, records)
        records[example_id] = record
        log(f"{example_id} {record['status']}")

    completed = sum(record["status"] == "success" for record in records.values())
    return {
        **before,
        "generated": generated,
        "failed": failed,
        "skipped": skipped,
        "completed_after_run": completed,
        "remaining_after_run": len(rows) - completed,
    }


class TransformersGemmaBackend:
    """Lazy local-only Transformers backend for the protected Gemma 3 model."""

    def __init__(self, model_path: str, tokenizer_path: str) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            tokenizer_path, local_files_only=True
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            device_map="auto",
            dtype="auto",
            local_files_only=True,
        )
        self._model.eval()
        tokenizer = self._processor.tokenizer
        chat_template = self._processor.chat_template or tokenizer.chat_template
        if not chat_template:
            raise ResponseGenerationError("tokenizer has no chat template")

        config = self._model.config
        self.model_identity = {
            "path": model_path,
            "class": type(self._model).__name__,
            "model_type": config.model_type,
            "name_or_path": config.name_or_path,
            "commit_hash": getattr(config, "_commit_hash", None),
            "config_sha256": hashlib.sha256(
                json.dumps(config.to_dict(), sort_keys=True, default=str).encode()
            ).hexdigest(),
        }
        self.tokenizer_identity = {
            "path": tokenizer_path,
            "processor_class": type(self._processor).__name__,
            "tokenizer_class": type(tokenizer).__name__,
            "name_or_path": tokenizer.name_or_path,
            "commit_hash": getattr(tokenizer, "_commit_hash", None),
            "chat_template_sha256": hashlib.sha256(chat_template.encode()).hexdigest(),
        }

    def generate(
        self, prompt: str, *, seed: int, settings: DecodingSettings
    ) -> GeneratedResponse:
        from transformers import set_seed

        set_seed(seed)
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]
        with self._torch.inference_mode():
            sequences = self._model.generate(
                **inputs,
                do_sample=True,
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_new_tokens=settings.max_new_tokens,
                num_return_sequences=1,
                return_dict_in_generate=False,
            )
        response_ids = sequences[0, prompt_tokens:]
        response_token_count = int(response_ids.shape[-1])
        eos_ids = self._eos_token_ids()
        ended_with_eos = bool(response_token_count) and int(response_ids[-1]) in eos_ids
        if ended_with_eos:
            termination_reason = "eos"
        elif response_token_count >= settings.max_new_tokens:
            termination_reason = "length_cap"
        else:
            termination_reason = "other"
        return GeneratedResponse(
            text=self._processor.decode(response_ids, skip_special_tokens=True),
            prompt_token_count=int(prompt_tokens),
            response_token_count=response_token_count,
            termination_reason=termination_reason,
        )

    def _eos_token_ids(self) -> set[int]:
        values = (
            self._model.generation_config.eos_token_id,
            self._processor.tokenizer.eos_token_id,
        )
        eos_ids: set[int] = set()
        for value in values:
            if isinstance(value, int):
                eos_ids.add(value)
            elif value is not None:
                eos_ids.update(value)
        return eos_ids


def _base_record(
    row: Mapping[str, Any],
    *,
    manifest_sha256: str,
    base_seed: int,
    seed: int,
    settings: DecodingSettings,
    model_path: str,
    tokenizer_path: str,
    backend: GenerationBackend,
    runtime_seconds: float,
) -> dict[str, Any]:
    return {
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "classification": CLASSIFICATION,
        "example_id": row["example_id"],
        "group_id": row["group_id"],
        "split": row["split"],
        "harmfulness": row["harmfulness"],
        "prompt_type": row["prompt_type"],
        "data_type": row["data_type"],
        "source": row["source"],
        "manifest_sha256": manifest_sha256,
        "base_seed": base_seed,
        "seed": seed,
        "seed_algorithm": SEED_ALGORITHM,
        "decoding": settings.as_dict(),
        "requested_model_path": model_path,
        "requested_tokenizer_path": tokenizer_path,
        "model": dict(backend.model_identity),
        "tokenizer": dict(backend.tokenizer_identity),
        "runtime_seconds": runtime_seconds,
    }


def _load_existing_records(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ResponseGenerationError(
                    f"output line {line_number} is not valid JSON"
                ) from exc
            if not isinstance(record, dict) or not isinstance(
                record.get("example_id"), str
            ):
                raise ResponseGenerationError(
                    f"output line {line_number} has no valid example_id"
                )
            example_id = record["example_id"]
            if example_id in records:
                raise ResponseGenerationError(
                    f"output contains duplicate record for {example_id}"
                )
            records[example_id] = record
    return records


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_records_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = records.get(row["example_id"])
            if record is not None:
                handle.write(
                    json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n"
                )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
