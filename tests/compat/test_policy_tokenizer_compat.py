# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise tokenizer compatibility without importing Ray or any model weights."""

import ast
import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors
from tokenizers.models import WordLevel
import transformers
from transformers import AutoTokenizer
from transformers.models.auto import tokenization_auto


@pytest.fixture
def apply_patch():
    """Load the real dependency-light compatibility function in isolation."""
    source = Path(__file__).parents[2] / "nemo_rl/models/policy/__init__.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_patch_transformers_tokenizer_class_set"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[function.name]


@pytest.fixture
def hf518():
    """Require the real target implementation for upstream behavior checks."""
    from packaging.version import Version

    version = Version(transformers.__version__)
    if not Version("5.18.0") <= version < Version("5.19.0"):
        pytest.skip(
            "Actual HF5.18 implementation is required for this compatibility proof"
        )


def test_hf518_native_tokenizer_preserves_normalization_and_special_tokens(
    apply_patch, hf518, tmp_path
):
    """Keep the native backend and its existing JSON processor unchanged."""
    tokenizer = Tokenizer(
        WordLevel(
            {"<BOS>": 0, "<EOS>": 1, "<UNK>": 2, "hello": 3, "world": 4},
            unk_token="<UNK>",
        )
    )
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFKC(), normalizers.Lowercase()]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="<BOS> $A <EOS>", special_tokens=[("<BOS>", 0), ("<EOS>", 1)]
    )
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "deepseek_v3"}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "LlamaTokenizer",
                "bos_token": "<BOS>",
                "eos_token": "<EOS>",
                "unk_token": "<UNK>",
            }
        )
    )
    before_method = AutoTokenizer.from_pretrained
    before_blocklist = set(tokenization_auto.MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS)
    before_mapping = dict(tokenization_auto.TOKENIZER_MAPPING_NAMES)
    apply_patch()
    assert AutoTokenizer.from_pretrained == before_method
    assert (
        tokenization_auto.MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS == before_blocklist
    )
    assert tokenization_auto.TOKENIZER_MAPPING_NAMES == before_mapping
    actual = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    assert type(actual).__name__ == "TokenizersBackend"
    assert (
        actual.encode("ＨＥＬＬＯ world")
        == tokenizer.encode("ＨＥＬＬＯ world").ids
        == [0, 3, 4, 1]
    )
    assert (actual.bos_token_id, actual.eos_token_id, actual.unk_token_id) == (0, 1, 2)


def test_hf518_explicit_trusted_remote_tokenizer_without_tokenizer_json(
    apply_patch, hf518, tmp_path, monkeypatch
):
    """Use the upstream trusted auto_map despite DeepSeek's blocklist entry."""
    import transformers.dynamic_module_utils as dynamic

    monkeypatch.setattr(dynamic, "HF_MODULES_CACHE", str(tmp_path / "modules"))
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "deepseek_v3"}))
    (source / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "LocalRemoteTokenizer",
                "auto_map": {
                    "AutoTokenizer": ["tokenization_local.LocalRemoteTokenizer", None]
                },
            }
        )
    )
    (source / "tokenization_local.py").write_text(
        "from tokenizers import Tokenizer\n"
        "from tokenizers.models import WordLevel\n"
        "from transformers import PreTrainedTokenizerFast\n"
        "class LocalRemoteTokenizer(PreTrainedTokenizerFast):\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, *args, **kwargs):\n"
        "        assert kwargs['trust_remote_code'] is True\n"
        "        tokenizer = Tokenizer(WordLevel({'<UNK>': 0, 'hello': 1}, unk_token='<UNK>'))\n"
        "        return cls(tokenizer_object=tokenizer, unk_token='<UNK>')\n"
    )
    assert "deepseek_v3" in tokenization_auto.MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS
    apply_patch()
    actual = AutoTokenizer.from_pretrained(
        source, trust_remote_code=True, local_files_only=True
    )
    assert actual.__class__.__name__ == "LocalRemoteTokenizer"
    assert actual.__class__.__module__.startswith("transformers_modules.")
    assert actual.encode("hello") == [1]
    assert not (source / "tokenizer.json").exists()
    assert "deepseek_v3" in tokenization_auto.MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS


@pytest.mark.parametrize("version", ["5.12.1", "5.12.1+vendor.1"])
def test_legacy_fallback_retries_only_after_original_failure(
    apply_patch, monkeypatch, version
):
    """Retain the old successful-native-first and failed-load retry contract."""
    monkeypatch.setattr(transformers, "__version__", version)
    blocklist = {"deepseek_v3", "other"}
    mapping = {"deepseek_v3": "TokenizersBackend", "other": "Untouched"}
    monkeypatch.setattr(
        tokenization_auto, "MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS", blocklist
    )
    monkeypatch.setattr(tokenization_auto, "TOKENIZER_MAPPING_NAMES", mapping)
    calls = []
    result = object()

    def original(*args, **kwargs):
        calls.append((set(blocklist), dict(mapping), args, kwargs))
        if len(calls) == 1:
            raise ValueError("Synthetic native JSON missing")
        return result

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", original)
    apply_patch()
    assert AutoTokenizer.from_pretrained("local", trust_remote_code=True) is result
    assert len(calls) == 2
    assert "deepseek_v3" in calls[0][0] and "deepseek_v3" in calls[0][1]
    assert calls[1][0] == {"other"} and calls[1][1] == {"other": "Untouched"}
    assert calls[0][2:] == calls[1][2:]


def test_legacy_success_does_not_mutate_tokenizer_registries(apply_patch, monkeypatch):
    """A successful original tokenizer keeps DeepSeek's native protection."""
    monkeypatch.setattr(transformers, "__version__", "5.12.1")
    blocklist = {"deepseek_v3"}
    mapping = {"deepseek_v3": "TokenizersBackend"}
    monkeypatch.setattr(
        tokenization_auto, "MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS", blocklist
    )
    monkeypatch.setattr(tokenization_auto, "TOKENIZER_MAPPING_NAMES", mapping)
    result = object()
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *args, **kwargs: result
    )
    apply_patch()
    assert AutoTokenizer.from_pretrained("local") is result
    assert blocklist == {"deepseek_v3"}
    assert mapping == {"deepseek_v3": "TokenizersBackend"}


@pytest.mark.parametrize("version", ["5.13.0", "5.17.0", "5.19.0"])
def test_unverified_versions_keep_explicit_guard(apply_patch, monkeypatch, version):
    """An upgrade needs another targeted upstream behavior proof."""
    monkeypatch.setattr(transformers, "__version__", version)
    with pytest.raises(AssertionError, match="registry behavior"):
        apply_patch()
