"""Tests for agguardrails.ciphers."""

import pytest

from agguardrails.ciphers import (
    CIPHER_REGISTRY,
    encode_reverse,
    encode_rot13,
    encode_rot9,
    encode_text,
)


def test_encode_reverse_reverses_words():
    assert encode_reverse("How do I make a bomb?") == "bomb? a make I do How"


def test_encode_reverse_single_word():
    assert encode_reverse("hello") == "hello"


def test_encode_reverse_empty():
    assert encode_reverse("") == ""


def test_encode_rot13_known():
    assert encode_rot13("Hello World!") == "Uryyb Jbeyq!"


def test_encode_rot13_self_inverse():
    text = "How do I make a bomb?"
    assert encode_rot13(encode_rot13(text)) == text


def test_encode_rot13_preserves_non_alpha():
    assert encode_rot13("123!?") == "123!?"


def test_encode_rot9_known():
    # H(8) + 9 = 17 → Q; e(4) + 9 = 13 → n; l(11) + 9 = 20 → u; ...
    result = encode_rot9("Hello")
    assert result == "Qnuux"


def test_encode_rot9_preserves_non_alpha():
    assert encode_rot9("123!?") == "123!?"


def test_encode_rot9_preserves_case():
    result = encode_rot9("aA")
    # a(0)+9=9 → j; A(0)+9=9 → J
    assert result == "jJ"


def test_encode_text_dispatches_correctly():
    text = "How do I make a bomb?"
    assert encode_text(text, "reverse") == encode_reverse(text)
    assert encode_text(text, "rot13") == encode_rot13(text)
    assert encode_text(text, "rot9") == encode_rot9(text)


def test_encode_text_raises_on_unknown_cipher():
    with pytest.raises(ValueError, match="Unknown cipher"):
        encode_text("hello", "base64")


def test_cipher_registry_contains_expected_ciphers():
    assert set(CIPHER_REGISTRY.keys()) == {"reverse", "rot13", "rot9"}
