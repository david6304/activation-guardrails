import pytest
import torch

from agguardrails.generation import (
    TransformersChatGenerator,
    make_chat_messages,
    real_generation_metadata,
)


def test_make_chat_messages_builds_prompt_only_exchange():
    assert make_chat_messages("hello") == [{"role": "user", "content": "hello"}]


def test_make_chat_messages_rejects_empty_prompt():
    with pytest.raises(ValueError, match="prompt"):
        make_chat_messages(" ")


def test_real_generation_metadata_records_runtime_defaults_and_overrides():
    assert real_generation_metadata({"generation": {}}) == {
        "backend": "transformers",
        "device_map": "auto",
        "torch_dtype": "auto",
    }
    assert real_generation_metadata(
        {
            "generation": {
                "runtime": {
                    "backend": "transformers",
                    "device_map": "cpu",
                    "torch_dtype": "float32",
                },
            },
        }
    ) == {
        "backend": "transformers",
        "device_map": "cpu",
        "torch_dtype": "float32",
    }


def test_transformers_chat_generator_slices_prompt_tokens_from_decoded_response():
    class FakeTokenizer:
        def apply_chat_template(
            self,
            messages,
            *,
            add_generation_prompt,
            return_tensors,
            tokenize,
        ):
            assert messages == [{"role": "user", "content": "prompt"}]
            assert add_generation_prompt is True
            assert return_tensors == "pt"
            assert tokenize is True
            return torch.tensor([[10, 11]])

        def decode(self, token_ids, *, skip_special_tokens):
            assert token_ids.tolist() == [42, 43]
            assert skip_special_tokens is True
            return " generated response "

    class FakeModel:
        device = torch.device("cpu")

        def __init__(self):
            self.generation_kwargs = None

        def generate(self, input_ids, **kwargs):
            assert input_ids.tolist() == [[10, 11]]
            self.generation_kwargs = kwargs
            return torch.tensor([[10, 11, 42, 43]])

    model = FakeModel()
    generator = TransformersChatGenerator(
        model=model,
        tokenizer=FakeTokenizer(),
        generation_params={"max_new_tokens": 8, "do_sample": False},
    )

    assert generator.generate_response("prompt") == "generated response"
    assert model.generation_kwargs == {"max_new_tokens": 8, "do_sample": False}
