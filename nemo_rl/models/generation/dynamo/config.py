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

from typing import Any, Literal, NotRequired, TypedDict

from nemo_rl.models.generation.interfaces import GenerationConfig


class DynamoSpecificArgs(TypedDict):
    """vLLM engine arguments passed through to dynamo.vllm."""

    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3"]
    precision: NotRequired[str]  # maps to vLLM --dtype
    load_format: NotRequired[str]
    enforce_eager: NotRequired[bool]
    hf_overrides: NotRequired[dict[str, Any]]
    extra_vllm_args: NotRequired[dict[str, Any]]


class DynamoCfg(TypedDict, total=False):
    """Dynamo infrastructure configuration."""

    frontend_http_port: int  # 0 = auto-assign
    router_mode: str  # "round-robin", "kv", "random", "least-loaded"
    etcd_port: int  # 0 = auto-assign
    etcd_peer_port: int  # 0 = auto-assign
    namespace: str
    enable_planner: bool  # Launch planner + VirtualConnectorClient for autoscaling
    initial_dp_size: int  # Workers at startup (must be <= cluster.world_size() // tp_size)


class DynamoVllmConfig(GenerationConfig):
    """GenerationConfig extended with Dynamo-specific settings.

    Uses key name "vllm_cfg" so that cfg["vllm_cfg"]["max_model_len"]
    works the same as VllmConfig for NeMo-Gym compatibility.
    """

    vllm_cfg: DynamoSpecificArgs
    dynamo_cfg: NotRequired[DynamoCfg]
    vllm_kwargs: NotRequired[dict[str, Any]]
