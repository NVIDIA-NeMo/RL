# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from copy import deepcopy

import pytest

from nemo_rl.models.generation.vllm.config import (
    NATIVE_LORA_CONFIG_KEY,
    configure_vllm_lora_refit,
)


def _native_policy_config() -> dict:
    return {
        "precision": "bfloat16",
        "dtensor_cfg": {
            "enabled": True,
            "_v2": True,
            "automodel_kwargs": {"force_hf": True},
            "lora_cfg": {
                "enabled": True,
                "dim": 16,
                "alpha": 64,
            },
        },
        "megatron_cfg": {"enabled": False},
        "generation": {
            "backend": "vllm",
            "lora_refit_mode": "native",
            "refit_transport": None,
            "vllm_cfg": {
                "async_engine": False,
                "load_format": "dummy",
                "precision": "bfloat16",
            },
            "vllm_kwargs": {"additional_config": {"preserved": True}},
        },
    }


def test_native_lora_refit_materializes_vllm_adapter_config() -> None:
    policy_config = _native_policy_config()

    configure_vllm_lora_refit(policy_config)

    generation_config = policy_config["generation"]
    vllm_kwargs = generation_config["vllm_kwargs"]
    assert generation_config["vllm_cfg"]["load_format"] == "auto"
    assert vllm_kwargs["enable_lora"] is True
    assert vllm_kwargs["max_lora_rank"] == 16
    assert vllm_kwargs["max_loras"] == 1
    assert vllm_kwargs["max_cpu_loras"] == 1
    assert vllm_kwargs["lora_dtype"] == "bfloat16"
    assert vllm_kwargs["additional_config"]["preserved"] is True
    assert vllm_kwargs["additional_config"][NATIVE_LORA_CONFIG_KEY] == {
        "rank": 16,
        "alpha": 64,
    }


def test_native_lora_refit_allows_explicitly_disabled_speculative_decoding() -> None:
    policy_config = _native_policy_config()
    policy_config["generation"]["vllm_kwargs"]["speculative_config"] = {
        "num_speculative_tokens": 0,
        "method": "mtp",
    }

    configure_vllm_lora_refit(policy_config)

    assert policy_config["generation"]["vllm_kwargs"]["enable_lora"] is True


def test_native_lora_refit_sizes_cpu_cache_for_configured_adapter_capacity() -> None:
    policy_config = _native_policy_config()
    policy_config["generation"]["vllm_kwargs"]["max_loras"] = 2

    configure_vllm_lora_refit(policy_config)

    assert policy_config["generation"]["vllm_kwargs"]["max_cpu_loras"] == 2


def test_merged_lora_refit_is_an_explicit_unchanged_opt_in() -> None:
    policy_config = _native_policy_config()
    policy_config["generation"]["lora_refit_mode"] = "merged"
    original_vllm_kwargs = deepcopy(policy_config["generation"]["vllm_kwargs"])

    configure_vllm_lora_refit(policy_config)

    generation_config = policy_config["generation"]
    assert generation_config["vllm_cfg"]["load_format"] == "dummy"
    assert generation_config["vllm_kwargs"] == original_vllm_kwargs


def test_omitted_lora_refit_mode_defaults_to_native() -> None:
    policy_config = _native_policy_config()
    del policy_config["generation"]["lora_refit_mode"]

    configure_vllm_lora_refit(policy_config)

    generation_config = policy_config["generation"]
    assert generation_config["lora_refit_mode"] == "native"
    assert generation_config["vllm_kwargs"]["enable_lora"] is True


def test_lora_refit_rejects_invalid_mode() -> None:
    policy_config = _native_policy_config()
    policy_config["generation"]["lora_refit_mode"] = "invalid"

    with pytest.raises(ValueError, match="lora_refit_mode"):
        configure_vllm_lora_refit(policy_config)


def test_native_default_is_a_noop_when_lora_is_disabled() -> None:
    policy_config = _native_policy_config()
    policy_config["dtensor_cfg"]["lora_cfg"]["enabled"] = False
    original_policy_config = deepcopy(policy_config)

    configure_vllm_lora_refit(policy_config)

    assert policy_config == original_policy_config


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda cfg: cfg["dtensor_cfg"]["automodel_kwargs"].update(
                {"force_hf": False}
            ),
            "force_hf=true",
        ),
        (
            lambda cfg: cfg["dtensor_cfg"].update({"_v2": False}),
            "_v2=true",
        ),
        (
            lambda cfg: cfg["generation"].update({"refit_transport": "nixl"}),
            "refit_transport=null",
        ),
        (
            lambda cfg: cfg["generation"]["vllm_cfg"].update({"async_engine": True}),
            "async_engine=true",
        ),
        (
            lambda cfg: cfg["generation"]["vllm_cfg"].update(
                {"expose_http_server": True}
            ),
            "HTTP server",
        ),
        (
            lambda cfg: cfg["generation"].update({"real_quant": True}),
            "quantized rollout",
        ),
        (
            lambda cfg: cfg["generation"].update({"quant_cfg": "nvfp4"}),
            "quantized rollout",
        ),
        (
            lambda cfg: cfg["generation"]["vllm_kwargs"].update(
                {"speculative_config": {"num_speculative_tokens": 1}}
            ),
            "speculative decoding",
        ),
        (
            lambda cfg: cfg["generation"]["vllm_kwargs"].update({"enable_lora": False}),
            "enable_lora=true",
        ),
        (
            lambda cfg: cfg["generation"]["vllm_kwargs"].update({"max_lora_rank": 8}),
            "smaller than",
        ),
        (
            lambda cfg: cfg["generation"]["vllm_kwargs"].update(
                {"max_loras": 2, "max_cpu_loras": 1}
            ),
            "max_cpu_loras",
        ),
        (
            lambda cfg: cfg["generation"]["vllm_cfg"].update({"precision": "float16"}),
            "matching policy and vLLM precision",
        ),
        (
            lambda cfg: cfg.update({"precision": "float32"}),
            "supports bfloat16 or float16",
        ),
        (
            lambda cfg: cfg["generation"]["vllm_kwargs"].update(
                {"lora_dtype": "float16"}
            ),
            "does not match",
        ),
        (
            lambda cfg: cfg["dtensor_cfg"]["lora_cfg"].update({"use_dora": True}),
            "use_dora=true",
        ),
        (
            lambda cfg: cfg["dtensor_cfg"]["lora_cfg"].update(
                {"moe_rank_scaling": True}
            ),
            "moe_rank_scaling=true",
        ),
    ],
)
def test_native_lora_refit_rejects_unsupported_config(mutation, message) -> None:
    policy_config = _native_policy_config()
    mutation(policy_config)

    with pytest.raises(ValueError, match=message):
        configure_vllm_lora_refit(policy_config)


def test_native_lora_refit_rejects_megatron_policy() -> None:
    policy_config = _native_policy_config()
    policy_config["dtensor_cfg"] = {"enabled": False}
    policy_config["megatron_cfg"] = {
        "enabled": True,
        "peft": {"enabled": True, "dim": 16, "alpha": 64},
    }

    with pytest.raises(ValueError, match="DTensor v2"):
        configure_vllm_lora_refit(policy_config)
