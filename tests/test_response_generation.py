from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from agguardrails.response_generation import (
    CLASSIFICATION,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DecodingSettings,
    GeneratedResponse,
    ResponseGenerationError,
    TransformersGemmaBackend,
    derive_example_seed,
    generate_responses,
    inspect_resume,
)
from agguardrails.wildjailbreak_manifest import build_manifest

MODEL_PATH = "/home/s2296274/models/gemma-3-4b-it-heretic"


class FakeBackend:
    model_identity = {"class": "FakeModel", "path": MODEL_PATH}
    tokenizer_identity = {"class": "FakeTokenizer", "path": MODEL_PATH}

    def __init__(self, failures: set[str] | None = None) -> None:
        self.failures = failures or set()
        self.calls: list[tuple[str, int, DecodingSettings]] = []

    def generate(
        self, prompt: str, *, seed: int, settings: DecodingSettings
    ) -> GeneratedResponse:
        self.calls.append((prompt, seed, settings))
        if prompt in self.failures:
            raise RuntimeError(f"failure involving hidden prompt: {prompt}")
        return GeneratedResponse(
            text=f"generated response for {prompt}",
            prompt_token_count=7,
            response_token_count=4,
            termination_reason="eos",
        )


def test_generation_records_metadata_and_safely_resumes_failures(
    tmp_path: Path,
) -> None:
    rows = _small_rows()
    output_path = tmp_path / "responses.jsonl"
    logs: list[str] = []
    settings = DecodingSettings()
    first_backend = FakeBackend(failures={"sensitive prompt two"})

    first_summary = generate_responses(
        rows,
        output_path=output_path,
        manifest_sha256="manifest-hash",
        base_seed=31,
        settings=settings,
        model_path=MODEL_PATH,
        tokenizer_path=MODEL_PATH,
        backend=first_backend,
        log=logs.append,
    )
    first_records = _read_jsonl(output_path)
    completed_before_resume = first_records[0]

    assert first_summary["generated"] == 1
    assert first_summary["failed"] == 1
    assert len(first_records) == 2
    assert first_records[0]["classification"] == CLASSIFICATION
    assert first_records[0]["decoding"] == {
        "do_sample": True,
        "temperature": DEFAULT_TEMPERATURE,
        "top_p": DEFAULT_TOP_P,
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "num_return_sequences": 1,
    }
    assert first_records[0]["prompt_token_count"] == 7
    assert first_records[0]["response_token_count"] == 4
    assert first_records[0]["termination_reason"] == "eos"
    assert first_records[0]["model"] == first_backend.model_identity
    assert first_records[0]["tokenizer"] == first_backend.tokenizer_identity
    assert first_records[0]["seed"] == derive_example_seed(31, "example-1")
    assert first_records[1]["status"] == "failure"
    assert first_records[1]["failure"] == {
        "stage": "generation",
        "error_type": "RuntimeError",
    }
    assert first_records[1]["prompt_token_count"] is None
    assert first_records[1]["response_token_count"] is None
    assert first_records[1]["termination_reason"] is None
    assert "response" not in first_records[1]
    assert all("sensitive prompt" not in line for line in logs)
    assert "sensitive prompt two" not in output_path.read_text(encoding="utf-8")

    second_backend = FakeBackend()
    second_summary = generate_responses(
        rows,
        output_path=output_path,
        manifest_sha256="manifest-hash",
        base_seed=31,
        settings=settings,
        model_path=MODEL_PATH,
        tokenizer_path=MODEL_PATH,
        backend=second_backend,
        log=logs.append,
    )
    resumed_records = _read_jsonl(output_path)

    assert second_summary["skipped"] == 1
    assert second_summary["generated"] == 1
    assert second_summary["remaining_after_run"] == 0
    assert len(resumed_records) == 2
    assert len({record["example_id"] for record in resumed_records}) == 2
    assert resumed_records[0] == completed_before_resume
    assert [call[0] for call in second_backend.calls] == ["sensitive prompt two"]


def test_resume_rejects_duplicates_and_incompatible_contract(tmp_path: Path) -> None:
    rows = _small_rows()
    output_path = tmp_path / "responses.jsonl"
    settings = DecodingSettings()
    backend = FakeBackend()
    generate_responses(
        rows[:1],
        output_path=output_path,
        manifest_sha256="manifest-hash",
        base_seed=9,
        settings=settings,
        model_path=MODEL_PATH,
        tokenizer_path=MODEL_PATH,
        backend=backend,
        log=lambda _: None,
    )

    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(output_path.read_text(encoding="utf-8"))
    with pytest.raises(ResponseGenerationError, match="duplicate"):
        inspect_resume(
            rows,
            output_path=output_path,
            manifest_sha256="manifest-hash",
            base_seed=9,
            settings=settings,
            model_path=MODEL_PATH,
            tokenizer_path=MODEL_PATH,
        )

    output_path.write_text(
        json.dumps(
            {
                **_read_jsonl(output_path)[0],
                "base_seed": 10,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ResponseGenerationError, match="incompatible"):
        inspect_resume(
            rows,
            output_path=output_path,
            manifest_sha256="manifest-hash",
            base_seed=9,
            settings=settings,
            model_path=MODEL_PATH,
            tokenizer_path=MODEL_PATH,
        )


@pytest.mark.parametrize(
    ("response_ids", "max_new_tokens", "expected_reason"),
    [
        ([8, 2], 4, "eos"),
        ([8, 8, 8, 8], 4, "length_cap"),
    ],
)
def test_transformers_backend_uses_exact_chat_template_and_classifies_termination(
    monkeypatch: pytest.MonkeyPatch,
    response_ids: list[int],
    max_new_tokens: int,
    expected_reason: str,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel(response_ids)
    backend = TransformersGemmaBackend.__new__(TransformersGemmaBackend)
    backend._torch = torch
    backend._tokenizer = tokenizer
    backend._model = model
    seeded: list[int] = []
    monkeypatch.setattr("transformers.set_seed", seeded.append)

    result = backend.generate(
        "template-sensitive prompt",
        seed=1234,
        settings=DecodingSettings(max_new_tokens=max_new_tokens),
    )

    assert seeded == [1234]
    assert tokenizer.template_call == {
        "conversation": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "template-sensitive prompt"}],
            }
        ],
        "add_generation_prompt": True,
        "tokenize": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    assert "chat_template" not in tokenizer.template_call
    assert model.generate_call["do_sample"] is True
    assert model.generate_call["temperature"] == DEFAULT_TEMPERATURE
    assert model.generate_call["top_p"] == DEFAULT_TOP_P
    assert model.generate_call["max_new_tokens"] == max_new_tokens
    assert model.generate_call["num_return_sequences"] == 1
    assert model.generate_call["return_dict_in_generate"] is False
    assert result.prompt_token_count == 3
    assert result.response_token_count == len(response_ids)
    assert result.termination_reason == expected_reason


def test_transformers_backend_loads_text_tokenizer_without_image_processor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer()
    tokenizer.chat_template = "test chat template"
    tokenizer.name_or_path = MODEL_PATH
    model = FakeLoadedModel()
    tokenizer_calls: list[tuple[str, dict[str, object]]] = []
    model_calls: list[tuple[str, dict[str, object]]] = []

    def load_tokenizer(path: str, **kwargs):
        tokenizer_calls.append((path, kwargs))
        return tokenizer

    def load_model(path: str, **kwargs):
        model_calls.append((path, kwargs))
        return model

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        "transformers.AutoModelForImageTextToText.from_pretrained", load_model
    )

    backend = TransformersGemmaBackend(MODEL_PATH, MODEL_PATH)

    assert tokenizer_calls == [(MODEL_PATH, {"local_files_only": True})]
    assert model_calls == [
        (
            MODEL_PATH,
            {
                "device_map": "auto",
                "dtype": "auto",
                "local_files_only": True,
            },
        )
    ]
    assert model.eval_called is True
    assert backend.tokenizer_identity["tokenizer_class"] == "FakeTokenizer"
    assert "processor_class" not in backend.tokenizer_identity


def test_cli_argument_parsing_and_dry_run_do_not_load_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = _load_script()
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_rows, _ = build_manifest(_source_rows(), seed=17)
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    output_path = tmp_path / "responses.jsonl"
    monkeypatch.setattr(
        script,
        "TransformersGemmaBackend",
        lambda *_: pytest.fail("dry-run loaded the model"),
    )

    args = script.parse_args([])
    assert args.model_path == MODEL_PATH
    assert args.tokenizer_path is None
    assert args.temperature == DEFAULT_TEMPERATURE
    assert args.top_p == DEFAULT_TOP_P
    assert args.max_new_tokens == DEFAULT_MAX_NEW_TOKENS

    exit_code = script.main(
        [
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
            "--dry-run",
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert summary["dry_run"] is True
    assert summary["manifest_examples"] == 400
    assert summary["pending"] == 400
    assert summary["model_path"] == MODEL_PATH
    assert summary["tokenizer_path"] == MODEL_PATH
    assert not output_path.exists()


class FakeBatch(dict):
    def to(self, device: str) -> FakeBatch:
        assert device == "cpu"
        return self


class FakeTokenizer:
    def __init__(self) -> None:
        self.eos_token_id = 2
        self.template_call: dict[str, object] = {}

    def apply_chat_template(self, conversation, **kwargs):
        self.template_call = {"conversation": conversation, **kwargs}
        return FakeBatch(
            {
                "input_ids": torch.tensor([[4, 5, 6]]),
                "attention_mask": torch.tensor([[1, 1, 1]]),
            }
        )

    def decode(self, response_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return "decoded response"


class FakeModel:
    device = "cpu"
    generation_config = SimpleNamespace(eos_token_id=[2])

    def __init__(self, response_ids: list[int]) -> None:
        self.response_ids = response_ids
        self.generate_call: dict[str, object] = {}

    def generate(self, **kwargs):
        self.generate_call = kwargs
        prompt_ids = kwargs["input_ids"][0].tolist()
        return torch.tensor([prompt_ids + self.response_ids])


class FakeLoadedModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_type="gemma3",
            name_or_path=MODEL_PATH,
            _commit_hash=None,
            to_dict=lambda: {"model_type": "gemma3"},
        )
        self.eval_called = False

    def eval(self) -> None:
        self.eval_called = True


def _small_rows() -> list[dict[str, object]]:
    return [
        {
            "example_id": "example-1",
            "group_id": "group-1",
            "split": "train",
            "harmfulness": "harmful",
            "prompt_type": "vanilla",
            "data_type": "vanilla_harmful",
            "prompt": "sensitive prompt one",
            "source": {"row_index": 1},
        },
        {
            "example_id": "example-2",
            "group_id": "group-2",
            "split": "train",
            "harmfulness": "harmful",
            "prompt_type": "adversarial",
            "data_type": "adversarial_harmful",
            "prompt": "sensitive prompt two",
            "source": {"row_index": 2},
        },
    ]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "generate_smoke_responses.py"
    spec = importlib.util.spec_from_file_location("generate_smoke_responses", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label in ("harmful", "benign"):
        for group_index in range(100):
            vanilla = f"base request {label} {group_index}"
            rows.extend(
                [
                    {
                        "vanilla": vanilla,
                        "adversarial": "",
                        "completion": "omitted",
                        "data_type": f"vanilla_{label}",
                    },
                    {
                        "vanilla": vanilla,
                        "adversarial": f"derived request {label} {group_index}",
                        "completion": "omitted",
                        "data_type": f"adversarial_{label}",
                    },
                ]
            )
    return rows
