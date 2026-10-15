"""Regression coverage for the native BEAM-fork tokenizer dispatch."""

import json
from unittest.mock import Mock

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoTokenizer
from transformers.models.auto import tokenization_auto


@pytest.mark.parametrize("remote", [False, True])
def test_deepseek_tokenizer_dispatch(tmp_path, monkeypatch, remote):
    """Honor explicit remote code while preserving local tokenizer loading."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v3"}))
    tokenizer_config = {"tokenizer_class": "LlamaTokenizerFast"}
    if remote:
        tokenizer_config["auto_map"] = {
            "AutoTokenizer": ["tokenization_custom.TikTokenTokenizer", None]
        }
        remote_class = Mock()
        loader = Mock(return_value=remote_class)
        monkeypatch.setattr(tokenization_auto, "get_class_from_dynamic_module", loader)
    else:
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
        tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))

    result = AutoTokenizer.from_pretrained(tmp_path, trust_remote_code=True)

    if remote:
        assert result is remote_class.from_pretrained.return_value
        assert loader.call_args.args[0] == "tokenization_custom.TikTokenTokenizer"
        remote_class.register_for_auto_class.assert_called_once_with()
        assert (
            remote_class.from_pretrained.call_args.kwargs["trust_remote_code"] is True
        )
    else:
        assert result.encode("hello", add_special_tokens=False) == [1]
