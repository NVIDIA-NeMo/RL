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

from typing import Any, Literal, NotRequired, Optional, TypedDict, cast

from nemo_rl.models.generation.interfaces import GenerationConfig
from nemo_rl.models.policy import PolicyConfig


class MCoreGenerationSpecificArgs(TypedDict):
    """Megatron fields related only to inference.

    Any fields not declared here but declared in the training-side config can be overwritten.
    For example, Megatron inference might want `transformer_impl: "inference_optimized"`,
    while Megatron training might want `transformer_impl: "transformer_engine"`.
    """

    expose_http_server: bool
    parsers: list[str]

    buffer_size_gb: int
    block_size_tokens: int
    max_tokens: int
    max_model_len: int

    num_cuda_graphs: int
    use_cuda_graphs_for_non_decode_steps: bool
    cuda_graph_impl: str
    # Inference CUDA-graph scope. Options:
    # - 'none': inference runs in eager mode (no CUDA graphs).
    # - 'layer': graphs are owned at the per-layer boundary (TransformerLayer / MambaLayer).
    # - 'block': graphs are owned at the enclosing block (TransformerBlock / HybridBlock).
    # Only meaningful when cuda_graph_impl='local'.
    inference_cuda_graph_scope: NotRequired[str]

    materialize_only_last_token_logits: bool
    enable_chunked_prefill: bool
    enable_prefix_caching: bool

    refit_backend: Literal["gloo", "nccl", "nvshmem"]
    num_speculative_tokens: int

    mamba_inference_ssm_states_dtype: NotRequired[str]
    mamba_inference_conv_states_dtype: NotRequired[str]

    # KV cache lifecycle across suspend/resume:
    # - "persist": cache stays allocated; CUDA graphs remain valid (default)
    # - "offload": cache is moved off-GPU between iterations
    #
    # The third mcore value, "recompute" (drop + rebuild on resume), must be set via
    # `grpo.async_grpo.recompute_kv_cache_after_weight_updates=true`.
    # TODO: Unify `kv_cache_management_mode` and `recompute_kv_cache_after_weight_updates`.
    kv_cache_management_mode: Literal["persist", "offload"]

    logging_step_interval: NotRequired[int]


class MCoreGenerationConfig(GenerationConfig):
    """Generation config for Megatron Inference."""

    mcore_generation_config: MCoreGenerationSpecificArgs


def merged_inference_megatron_cfg(policy_config: PolicyConfig) -> dict[str, Any]:
    """The `megatron_cfg` a dedicated inference model runs with."""
    generation_config = cast(MCoreGenerationConfig, policy_config["generation"])
    return {
        **cast(dict[str, Any], policy_config["megatron_cfg"]),
        **(generation_config.get("mcore_generation_config") or {}),
        "activation_checkpointing": False,
    }


def dedicated_inference_megatron_cfg(
    policy_config: PolicyConfig,
) -> Optional[dict[str, Any]]:
    """The `megatron_cfg` for a dedicated colocated inference model, or None.

    Colocated Megatron generation shares the training model unless the resolved
    inference layout or `transformer_impl` differs from training; then the worker
    builds a second model and reshards into it on every wake. Inference never
    uses CP, so CP is pinned to 1 (CP>1 training therefore always differs).

    Returns None when the resolved config matches training (dual-mode: generate
    directly on the shared training model).
    """
    inference_mcfg = merged_inference_megatron_cfg(policy_config)
    inference_mcfg["context_parallel_size"] = 1

    train_mcfg = cast(dict[str, Any], policy_config["megatron_cfg"])
    layout_keys = (
        "tensor_model_parallel_size",
        "pipeline_model_parallel_size",
        "expert_model_parallel_size",
        "expert_tensor_parallel_size",
        "context_parallel_size",
    )
    layout_differs = any(inference_mcfg[k] != train_mcfg[k] for k in layout_keys)
    impl_differs = inference_mcfg.get("transformer_impl") != train_mcfg.get(
        "transformer_impl"
    )
    if not (layout_differs or impl_differs):
        return None
    return inference_mcfg
