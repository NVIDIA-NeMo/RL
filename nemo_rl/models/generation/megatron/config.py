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


class MCoreGenerationSpecificArgs(TypedDict):
    """Megatron fields related only to inference.

    Any fields not declared here but declared in the training-side config can be overwritten.
    For example, Megatron inference might want `transformer_impl: "inference_optimized"`,
    while Megatron training might want `transformer_impl: "transformer_engine"`.
    """

    async_engine: bool
    expose_http_server: bool
    parsers: list[str]

    buffer_size_gb: int
    block_size_tokens: int
    max_tokens: int
    max_model_len: int

    # None disables CUDA-graph bucket construction; -1 selects automatic
    # sizing; positive values request a fixed maximum bucket count.
    num_cuda_graphs: int | None
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

    # "bridge" uses the shared packed-broadcast transport and Megatron Bridge
    # import mappings. "mcore" uses Megatron Core's native reshard/refit API.
    refit_impl: Literal["bridge", "mcore"]
    # Copy-service backend used only when refit_impl="mcore".
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
    # Whether MCore returns selected-token log-probs before or after sampling
    # processors. Policy recomputation uses raw model logits, so numerical
    # parity checks should select raw_logprobs explicitly.
    logprobs_mode: Literal["processed_logprobs", "raw_logprobs"]


class MCoreGenerationConfig(GenerationConfig):
    """Generation config for Megatron Inference."""

    # None uses Bridge packed broadcast or native MCore refit according to
    # refit_impl. nccl_reshard selects the non-colocated NCCL M-to-N transport.
    refit_transport: NotRequired[Literal["nccl_reshard"] | None]
    mcore_generation_config: MCoreGenerationSpecificArgs
