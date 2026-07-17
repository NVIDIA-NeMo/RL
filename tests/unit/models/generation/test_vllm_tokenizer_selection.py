# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from nemo_rl.models.generation.vllm.vllm_worker import _resolve_tokenizer_name


def test_vllm_worker_uses_explicit_tokenizer_name() -> None:
    tokenizer_name = _resolve_tokenizer_name(
        {  # type: ignore[typeddict-item]
            "model_name": "org/base-model",
            "tokenizer_name": "org/instruct-tokenizer",
        }
    )

    assert tokenizer_name == "org/instruct-tokenizer"


def test_vllm_worker_defaults_tokenizer_to_model() -> None:
    tokenizer_name = _resolve_tokenizer_name(  # type: ignore[typeddict-item]
        {"model_name": "org/model"}
    )

    assert tokenizer_name == "org/model"
