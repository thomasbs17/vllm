import pytest
from unittest.mock import MagicMock, ANY

from vllm.entrypoints.openai.fim import StringTemplateFIMEncoder

def test_raises_error_when_tokenizer_lacks_convert_method():
    tokenizer = MagicMock()
    del tokenizer.convert_tokens_to_ids
    with pytest.raises(ValueError) as exc_info:
        StringTemplateFIMEncoder(tokenizer, "test", "template")
    assert "incompatible with 'test' FIM encoder" in str(exc_info.value)

def test_raises_error_when_special_token_is_unknown():
    tokenizer = MagicMock()
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.return_value = 0  # Matches unk_token_id
    special_tokens = ["<FIM>"]
    with pytest.raises(ValueError) as exc_info:
        StringTemplateFIMEncoder(tokenizer, "test", "template", special_tokens)
    assert "incompatible with 'test' FIM encoder" in str(exc_info.value)

def test_initializes_with_valid_special_tokens():
    tokenizer = MagicMock()
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.return_value = 1  # Different from unk_token_id
    special_tokens = ["<FIM>"]
    encoder = StringTemplateFIMEncoder(tokenizer, "test", "template", special_tokens)
    assert encoder is not None

def test_special_tokens_none_initializes_successfully():
    tokenizer = MagicMock()
    encoder = StringTemplateFIMEncoder(tokenizer, "test", "template", special_tokens=None)
    assert encoder is not None

def test_encode_with_suffix_formats_template_correctly():
    tokenizer = MagicMock()
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.return_value = 1
    template = "PREFIX{prefix}SUFFIX{suffix}"
    encoder = StringTemplateFIMEncoder(tokenizer, "test", template, ["PREFIX", "SUFFIX"])
    prefix = "hello"
    suffix = "world"
    expected_prompt = "PREFIXhelloSUFFIXworld"
    mock_input_ids = [1, 2, 3]
    tokenizer.return_value.input_ids = mock_input_ids
    result = encoder.encode_with_suffix(prefix, suffix)
    tokenizer.assert_called_once_with(expected_prompt, add_special_tokens=False)
    assert result == mock_input_ids

def test_tokenizer_called_without_special_tokens():
    tokenizer = MagicMock()
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.return_value = 1
    encoder = StringTemplateFIMEncoder(tokenizer, "test", "template", ["token"])
    encoder.encode_with_suffix("a", "b")
    tokenizer.assert_called_once_with(ANY, add_special_tokens=False)

def test_empty_prefix_and_suffix():
    tokenizer = MagicMock()
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.return_value = 1
    template = "{prefix}MID{suffix}"
    encoder = StringTemplateFIMEncoder(tokenizer, "test", template)
    tokenizer.return_value.input_ids = [123]
    result = encoder.encode_with_suffix("", "")
    tokenizer.assert_called_once_with("MID", add_special_tokens=False)
    assert result == [123]

def test_multiple_placeholders_in_template():
    tokenizer = MagicMock()
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.return_value = 1
    template = "{prefix} a {prefix} b {suffix} c {suffix}"
    encoder = StringTemplateFIMEncoder(tokenizer, "test", template)
    expected_prompt = "X a X b Y c Y"
    tokenizer.return_value.input_ids = [1, 2, 3]
    result = encoder.encode_with_suffix("X", "Y")
    tokenizer.assert_called_once_with(expected_prompt, add_special_tokens=False)
    assert result == [1, 2, 3]

def test_template_without_placeholders():
    tokenizer = MagicMock()
    encoder = StringTemplateFIMEncoder(tokenizer, "test", "fixed_template", None)
    tokenizer.return_value.input_ids = [5]
    result = encoder.encode_with_suffix("ignored", "ignored")
    tokenizer.assert_called_once_with("fixed_template", add_special_tokens=False)
    assert result == [5]

def test_special_characters_in_prompt():
    tokenizer = MagicMock()
    tokenizer.unk_token_id = 0
    tokenizer.convert_tokens_to_ids.return_value = 1
    template = "{prefix}\n{suffix}"
    encoder = StringTemplateFIMEncoder(tokenizer, "test", template)
    expected_prompt = "hello\nworld"
    tokenizer.return_value.input_ids = [10, 20]
    result = encoder.encode_with_suffix("hello", "world")
    tokenizer.assert_called_once_with(expected_prompt, add_special_tokens=False)
    assert result == [10, 20]