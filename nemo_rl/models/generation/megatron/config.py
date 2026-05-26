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

from typing import Literal, NotRequired, TypedDict

from nemo_rl.models.generation.interfaces import GenerationConfig


class MCoreGenerationConfig(TypedDict):
    """Megatron fields related only to inference.

    Any fields not declared here but declared in the training-side config can be overwritten.
    For example, Megatron inference might want `transformer_impl: "inference_optimized"`,
    while Megatron training might want `transformer_impl: "transformer_engine"`.
    """
    async_engine: NotRequired[bool]
    expose_http_server: NotRequired[bool]
    parsers: NotRequired[list[str]]

    buffer_size_gb: int
    block_size_tokens: int
    max_tokens: int

    num_cuda_graphs: int
    use_cuda_graphs_for_non_decode_steps: bool
    cuda_graph_impl: NotRequired[str]
    # Inference CUDA-graph scope. Options:
    # - 'none': inference runs in eager mode (no CUDA graphs).
    # - 'layer': graphs are owned at the per-layer boundary (TransformerLayer / MambaLayer).
    # - 'block': graphs are owned at the enclosing block (TransformerBlock / HybridBlock).
    # Only meaningful when cuda_graph_impl='local'.
    inference_cuda_graph_scope: NotRequired[str]

    use_flashinfer_sampling: NotRequired[bool]
    materialize_only_last_token_logits: bool
    enable_chunked_prefill: NotRequired[bool]
    enable_prefix_caching: NotRequired[bool]
    prefix_caching_coordinator_policy: NotRequired[str]

    refit_backend: Literal["gloo", "nvshmem"]
    num_speculative_tokens: int

    mamba_inference_ssm_states_dtype: NotRequired[str]
    mamba_inference_conv_states_dtype: NotRequired[str]
    # Fraction of the KV-cache buffer reserved for mamba states (SSM + conv).
    mamba_memory_ratio: NotRequired[float]

    # KV cache lifecycle across suspend/resume:
    # - "persist": cache stays allocated; CUDA graphs remain valid (default)
    # - "offload": cache is moved off-GPU between iterations
    #
    # The third mcore value, "recompute" (drop + rebuild on resume), must be set via
    # `grpo.async_grpo.recompute_kv_cache_after_weight_updates=true`.
    # TODO: Unify `kv_cache_management_mode` and `recompute_kv_cache_after_weight_updates`.
    kv_cache_management_mode: NotRequired[Literal["persist", "offload"]]

    logging_step_interval: NotRequired[int]

    inference_coordinator_port: NotRequired[int]


class MegatronGenerationConfig(GenerationConfig):
    """Generation config for Megatron Inference."""

    mcore_generation_config: MCoreGenerationConfig
