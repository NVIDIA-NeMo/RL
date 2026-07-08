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

from typing import Any, NotRequired, TypedDict

from nemo_rl.models.generation.interfaces import GenerationConfig


class TrtllmSpecificArgs(TypedDict):
    tensor_parallel_size: int
    gpu_memory_utilization: NotRequired[float]
    max_model_len: int
    precision: NotRequired[str]
    max_batch_size: NotRequired[int]
    max_num_tokens: NotRequired[int]
    # Use tensorrt_llm._torch.async_llm.AsyncLLM instead of the sync LLM.
    # Async mode allows true concurrent generation (one in-flight request per
    # prompt) and exposes llm.release()/llm.resume()/llm.update_weights() for
    # the upcoming colocated mode. Defaults to false (sync LLM, current behavior).
    async_engine: NotRequired[bool]
    # MoE expert parallelism. TRT-LLM splits the TP dimension on MoE layers
    # into moe_tp × moe_ep, so the constraint is
    #     moe_tensor_parallel_size * moe_expert_parallel_size == tensor_parallel_size
    # The outer worker count is unchanged (still TP × PP × DP) — these only
    # affect how MoE expert weights are partitioned inside each TP rank.
    moe_tensor_parallel_size: NotRequired[int]
    moe_expert_parallel_size: NotRequired[int]
    in_flight_weight_updates: NotRequired[bool]
    recompute_kv_cache_after_weight_updates: NotRequired[bool]
    default_chat_template_kwargs: NotRequired[dict[str, Any]]


class TrtllmConfig(GenerationConfig):
    trtllm_cfg: TrtllmSpecificArgs
    # Escape hatch for arbitrary TRT-LLM LLM/AsyncLLM constructor kwargs not
    # covered by TrtllmSpecificArgs (e.g. sampler_type, enable_attention_dp).
    # Spread into the engine constructor as `**trtllm_kwargs`.
    trtllm_kwargs: NotRequired[dict[str, Any]]
