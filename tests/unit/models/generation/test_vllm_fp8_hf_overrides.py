# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for merging fp8 kwargs with user-supplied hf_overrides.

The fp8 path returns a nested ``hf_overrides`` (holding ``quantization_config``).
A naive ``vllm_kwargs.update(fp8_kwargs)`` shallow-merges and clobbers any
user-supplied ``hf_overrides``. This exact bug was introduced, fixed (#1413),
silently reverted (#2188), and re-fixed (#2904). These tests pin the merge
behavior so it cannot regress a third time.
"""

from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm.patches import (
    _patch_vllm_mxfp8_speculative_draft_precision,
)
from nemo_rl.models.generation.vllm.vllm_worker import _merge_fp8_kwargs


def test_fp8_and_user_hf_overrides_coexist():
    """Both fp8's quantization_config and a user override survive the merge."""
    vllm_kwargs = {"hf_overrides": {"max_position_embeddings": 8192}}
    fp8_kwargs = {
        "quantization": "fp8",
        "kv_cache_dtype": "auto",
        "hf_overrides": {"quantization_config": {"weight_block_size": [128, 128]}},
    }

    _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)

    # fp8 quantization settings applied
    assert vllm_kwargs["quantization"] == "fp8"
    assert vllm_kwargs["kv_cache_dtype"] == "auto"
    # fp8's quantization_config survives ...
    assert vllm_kwargs["hf_overrides"]["quantization_config"] == {
        "weight_block_size": [128, 128]
    }
    # ... and so does the user-supplied override
    assert vllm_kwargs["hf_overrides"]["max_position_embeddings"] == 8192


def test_user_hf_overrides_take_precedence():
    """On key collision, the user-supplied hf_overrides value wins."""
    vllm_kwargs = {
        "hf_overrides": {
            "quantization_config": {
                "user": "wins",
                "ignore": ["model.layers.0.self_attn.*"],
            }
        }
    }
    fp8_kwargs = {
        "hf_overrides": {
            "quantization_config": {
                "fp8": "base",
                "ignore": ["lm_head"],
                "ignored_layers": ["lm_head"],
            }
        },
    }

    _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)

    assert vllm_kwargs["hf_overrides"]["quantization_config"] == {
        "fp8": "base",
        "user": "wins",
        "ignore": ["lm_head", "model.layers.0.self_attn.*"],
        "ignored_layers": ["lm_head"],
    }


def test_none_user_quantization_config_keeps_generated_config():
    vllm_kwargs = {"hf_overrides": {"quantization_config": None}}
    fp8_kwargs = {
        "hf_overrides": {
            "quantization_config": {
                "ignore": ["lm_head"],
                "ignored_layers": ["lm_head"],
            }
        }
    }

    _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)

    assert vllm_kwargs["hf_overrides"]["quantization_config"] == {
        "ignore": ["lm_head"],
        "ignored_layers": ["lm_head"],
    }


def test_non_mapping_user_quantization_config_is_rejected():
    vllm_kwargs = {"hf_overrides": {"quantization_config": "invalid"}}
    fp8_kwargs = {"hf_overrides": {"quantization_config": {"ignore": ["lm_head"]}}}

    with pytest.raises(
        ValueError, match="hf_overrides.quantization_config must be a mapping"
    ):
        _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)


def test_no_existing_hf_overrides():
    """fp8's hf_overrides apply cleanly when the user supplied none."""
    vllm_kwargs = {}
    fp8_kwargs = {
        "quantization": "fp8",
        "hf_overrides": {"quantization_config": {"weight_block_size": [128, 128]}},
    }

    _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)

    assert vllm_kwargs["hf_overrides"] == {
        "quantization_config": {"weight_block_size": [128, 128]}
    }


def test_none_hf_overrides_treated_as_empty():
    """A ``None`` hf_overrides (e.g. from config defaults) is handled as empty."""
    vllm_kwargs = {"hf_overrides": None}
    fp8_kwargs = {
        "hf_overrides": {"quantization_config": {"weight_block_size": [128, 128]}},
    }

    _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)

    assert vllm_kwargs["hf_overrides"] == {
        "quantization_config": {"weight_block_size": [128, 128]}
    }


def test_source_fp8_kwargs_not_mutated():
    """The merge must not mutate the caller's fp8_kwargs dict."""
    vllm_kwargs = {}
    fp8_kwargs = {
        "quantization": "fp8",
        "hf_overrides": {"quantization_config": {"weight_block_size": [128, 128]}},
    }

    _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)

    assert "hf_overrides" in fp8_kwargs


def _runtime_mxfp8_kwargs(speculative_config):
    vllm_kwargs = {"speculative_config": speculative_config}
    fp8_kwargs = {
        "quantization": "fp8",
        "hf_overrides": {
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MXFP8",
                "ignore": ["lm_head"],
            }
        },
    }

    _merge_fp8_kwargs(vllm_kwargs, fp8_kwargs)

    return vllm_kwargs


def _capture_vllm_0251_draft_model_config(monkeypatch, target_kwargs):
    from vllm.config import speculative as speculative_module
    from vllm.config.speculative import SpeculativeConfig

    captured = {}

    def fake_model_config(**kwargs):
        captured.update(kwargs)
        model_type = "deepseek_mtp" if kwargs["model"] == "target-model" else "qwen3"
        return SimpleNamespace(
            **kwargs,
            architectures=["DraftForCausalLM"],
            hf_config=SimpleNamespace(
                architectures=["DraftForCausalLM"],
                model_type=model_type,
            ),
            get_vocab_size=lambda: 32000,
            verify_with_parallel_config=lambda _parallel_config: None,
        )

    monkeypatch.setattr(speculative_module, "ModelConfig", fake_model_config)
    monkeypatch.setattr(
        SpeculativeConfig,
        "_verify_and_get_draft_tp",
        staticmethod(lambda *_args, **_kwargs: 1),
    )
    monkeypatch.setattr(
        SpeculativeConfig,
        "create_draft_parallel_config",
        staticmethod(lambda *_args, **_kwargs: SimpleNamespace()),
    )

    target_model_config = SimpleNamespace(
        model="target-model",
        tokenizer="target-tokenizer",
        tokenizer_mode="auto",
        trust_remote_code=False,
        allowed_local_media_path=None,
        allowed_media_domains=None,
        dtype="bfloat16",
        seed=0,
        tokenizer_revision=None,
        max_model_len=4096,
        enforce_eager=False,
        max_logprobs=20,
        config_format="auto",
        hf_text_config=SimpleNamespace(model_type="deepseek_v3"),
        get_vocab_size=lambda: 32000,
        **target_kwargs,
    )

    return SpeculativeConfig, target_model_config, captured


def test_external_draft_stays_bf16_with_target_runtime_mxfp8(monkeypatch):
    speculative_config = {
        "method": "draft_model",
        "model": "external-draft-model",
        "num_speculative_tokens": 3,
    }
    vllm_kwargs = _runtime_mxfp8_kwargs(speculative_config.copy())
    SpeculativeConfig, target_model_config, captured = (
        _capture_vllm_0251_draft_model_config(
            monkeypatch,
            {
                "quantization": "modelopt_mxfp8",
                "hf_overrides": vllm_kwargs["hf_overrides"],
            },
        )
    )
    _patch_vllm_mxfp8_speculative_draft_precision()

    speculative_config_obj = SpeculativeConfig(
        **vllm_kwargs["speculative_config"],
        target_model_config=target_model_config,
        target_parallel_config=SimpleNamespace(),
    )

    assert vllm_kwargs["speculative_config"] == speculative_config
    assert captured["quantization"] is None

    draft_hf_config = SimpleNamespace(
        architectures=["DraftForCausalLM"],
        model_type="qwen3",
    )
    captured["hf_overrides"](draft_hf_config)
    assert not hasattr(draft_hf_config, "quantization_config")
    assert speculative_config_obj.target_model_config is target_model_config


@pytest.mark.parametrize("method", ["mtp", "deepseek_mtp"])
def test_native_mtp_stays_bf16_with_target_runtime_mxfp8(monkeypatch, method: str):
    vllm_kwargs = _runtime_mxfp8_kwargs(
        {
            "method": method,
            "num_speculative_tokens": 1,
        }
    )
    SpeculativeConfig, target_model_config, captured = (
        _capture_vllm_0251_draft_model_config(
            monkeypatch,
            {
                "quantization": "modelopt_mxfp8",
                "hf_overrides": vllm_kwargs["hf_overrides"],
            },
        )
    )
    _patch_vllm_mxfp8_speculative_draft_precision()

    speculative_config_obj = SpeculativeConfig(
        **vllm_kwargs["speculative_config"],
        target_model_config=target_model_config,
        target_parallel_config=SimpleNamespace(),
    )

    assert captured["quantization"] is None
    assert target_model_config.quantization == "modelopt_mxfp8"
    assert speculative_config_obj.target_model_config is target_model_config


def test_native_mtp_respects_explicit_draft_quantization(monkeypatch):
    vllm_kwargs = _runtime_mxfp8_kwargs(
        {
            "method": "mtp",
            "num_speculative_tokens": 1,
            "quantization": "fp8",
        }
    )
    SpeculativeConfig, target_model_config, captured = (
        _capture_vllm_0251_draft_model_config(
            monkeypatch,
            {
                "quantization": "modelopt_mxfp8",
                "hf_overrides": vllm_kwargs["hf_overrides"],
            },
        )
    )
    _patch_vllm_mxfp8_speculative_draft_precision()

    speculative_config_obj = SpeculativeConfig(
        **vllm_kwargs["speculative_config"],
        target_model_config=target_model_config,
        target_parallel_config=SimpleNamespace(),
    )

    assert captured["quantization"] == "fp8"
    assert speculative_config_obj.target_model_config is target_model_config


def test_vllm_keeps_target_dict_overrides_out_of_draft_model():
    from vllm.config.speculative import SpeculativeConfig

    draft_hf_overrides = SpeculativeConfig.compose_draft_hf_overrides(
        {
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "MXFP8",
                "ignore": ["lm_head"],
            }
        }
    )

    assert draft_hf_overrides is SpeculativeConfig.hf_config_override
