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

import copy
from typing import Any

import pytest

from nemo_rl.models.generation.dynamo.config import DynamoConfig, RolePoolConfig

MEGATRON_CFG = {
    "enabled": True,
    "tensor_model_parallel_size": 2,
    "pipeline_model_parallel_size": 1,
    "context_parallel_size": 1,
}


def _megatron_pd_config() -> dict[str, Any]:
    return {
        "backend": "dynamo",
        "model_name": "Qwen/Qwen3-0.6B",
        "colocated": {
            "enabled": False,
            "resources": {"gpus_per_node": 8, "num_nodes": 1},
        },
        "refit_transport": None,
        "mcore_generation_config": {
            "enable_prefix_caching": True,
            "kv_cache_management_mode": "persist",
            "expose_http_server": False,
        },
        "dynamo_cfg": {
            "engine": "megatron",
            "startup_timeout_s": 5,
            "request_timeout_s": 30,
            "control_timeout_s": 30,
            "metrics_include_prefixes": None,
            "metrics_exclude_prefixes": None,
            "worker_args": {
                "tool_call_parser": None,
                "reasoning_parser": None,
                "exclude_tools_when_tool_choice_none": True,
                "enable_structural_tag": False,
                "structural_tag_scope": "auto",
                "structural_tag_schema": "auto",
                "custom_jinja_template": None,
                "endpoint_types": ["completions"],
                "extra_cli_args": [],
            },
            "frontend_args": {
                "tokenizer": "default",
                "tokenizer_cache": False,
                "tokenizer_cache_bytes": 50 * 1024 * 1024,
                "router_mode": "kv",
                "extra_cli_args": [],
            },
            "disaggregation": {
                "decode": {
                    "workers": 2,
                    "megatron_cfg": {"tensor_model_parallel_size": 2},
                },
                "prefill": {
                    "workers": 1,
                    "megatron_cfg": {"tensor_model_parallel_size": 4},
                    "mcore_generation_config": {"max_requests": 16},
                },
            },
        },
    }


def test_megatron_engine_validates_without_vllm_config_and_plans_decode_first():
    config = DynamoConfig.model_validate(_megatron_pd_config())
    assert config.is_megatron_engine
    assert config.vllm_cfg is None
    plan = config.role_plan(MEGATRON_CFG)
    assert [spec.name for spec in plan.engines] == ["decode-0", "decode-1", "prefill-0"]
    assert [spec.world_size for spec in plan.engines] == [2, 2, 4]
    assert plan.rank_offsets(8) == [8, 10, 12]
    assert plan.engines[2].engine_cfg["mcore_generation_config"] == {
        "max_requests": 16
    }
    with pytest.raises(ValueError, match="only defined for dynamo_cfg.engine='vllm'"):
        _ = config.engine_world_size


def test_megatron_engine_requires_disaggregation():
    cfg = _megatron_pd_config()
    del cfg["dynamo_cfg"]["disaggregation"]
    with pytest.raises(ValueError, match="requires dynamo_cfg.disaggregation"):
        DynamoConfig.model_validate(cfg)


def test_vllm_engine_rejects_disaggregation_for_now():
    cfg = _megatron_pd_config()
    cfg["dynamo_cfg"]["engine"] = "vllm"
    with pytest.raises(ValueError, match="not supported with dynamo_cfg.engine='vllm'"):
        DynamoConfig.model_validate(cfg)


def test_vllm_engine_still_requires_vllm_config():
    cfg = _megatron_pd_config()
    cfg["dynamo_cfg"]["engine"] = "vllm"
    del cfg["dynamo_cfg"]["disaggregation"]
    with pytest.raises(ValueError, match="vllm_cfg is required"):
        DynamoConfig.model_validate(cfg)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: c["mcore_generation_config"].update(enable_prefix_caching=False), "enable_prefix_caching"),
        (lambda c: c["mcore_generation_config"].update(kv_cache_management_mode="recompute"), "persist"),
        (lambda c: c["mcore_generation_config"].update(expose_http_server=True), "expose_http_server"),
        (lambda c: c.update(vllm_kwargs={"max_num_seqs": 8}), "vllm_kwargs must be empty"),
        (lambda c: c.update(refit_transport="mcore"), "refit_transport must be null"),
        (lambda c: c.pop("mcore_generation_config"), "mcore_generation_config is required"),
    ],
)
def test_megatron_engine_boundary_rules(mutate, match):
    cfg = _megatron_pd_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        DynamoConfig.model_validate(cfg)


def test_role_overrides_are_allowlisted():
    with pytest.raises(ValueError, match="may only override"):
        RolePoolConfig.model_validate(
            {"workers": 1, "megatron_cfg": {"num_layers": 4}}
        )
    with pytest.raises(ValueError, match="context_parallel_size=1"):
        RolePoolConfig.model_validate(
            {"workers": 1, "megatron_cfg": {"context_parallel_size": 2}}
        )
    with pytest.raises(ValueError, match="may only override"):
        RolePoolConfig.model_validate(
            {"workers": 1, "vllm_cfg": {"max_model_len": 4096}}
        )


def test_plan_requires_the_policy_megatron_config_for_megatron_engines():
    config = DynamoConfig.model_validate(_megatron_pd_config())
    with pytest.raises(ValueError, match="policy.megatron_cfg is required"):
        config.role_plan(None)


def test_plan_rejects_context_parallel_base_layout():
    config = DynamoConfig.model_validate(_megatron_pd_config())
    base = copy.deepcopy(MEGATRON_CFG)
    base["context_parallel_size"] = 2
    with pytest.raises(ValueError, match="context_parallel_size=1"):
        config.role_plan(base)
