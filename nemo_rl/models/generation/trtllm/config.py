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

from typing import Any, NotRequired, TypedDict

from nemo_rl.models.generation.interfaces import GenerationConfig


class TrtllmSpecificArgs(TypedDict):
    tensor_parallel_size: int
    model_name: NotRequired[str]
    gpu_memory_utilization: NotRequired[float]
    max_model_len: int
    precision: str
    max_batch_size: int
    max_num_tokens: int
    expose_http_server: NotRequired[bool]
    async_engine: NotRequired[bool]
    # MoE expert parallelism. TRT-LLM splits the TP dimension on MoE layers
    # into moe_tp × moe_ep, so the constraint is
    #     moe_tensor_parallel_size * moe_expert_parallel_size == tensor_parallel_size
    # The outer worker count is unchanged (still TP × PP × DP) — these only
    # affect how MoE expert weights are partitioned inside each TP rank.
    moe_tensor_parallel_size: NotRequired[int]
    moe_expert_parallel_size: NotRequired[int]
    # These mirror grpo.async_grpo.{in_flight_weight_updates,
    # recompute_kv_cache_after_weight_updates}. They are duplicated here because
    # TrtllmGeneration.update_weights_from_collective() reads the drain / kv-recompute
    # behavior from its generation config (self.cfg["trtllm_cfg"]) — the generation
    # backend does not receive the top-level master_config.grpo.async_grpo. Keep the
    # two in sync (the exemplar grpo_math_1B_trtllm.yaml interpolates them from
    # grpo.async_grpo so they cannot diverge).
    in_flight_weight_updates: NotRequired[bool]
    recompute_kv_cache_after_weight_updates: NotRequired[bool]
    default_chat_template_kwargs: NotRequired[dict[str, Any]]
    # TRT-LLM's registered parser names:
    #   "qwen3"       -> Qwen3ToolParser      (JSON format: {"name":..., "arguments":{...}})
    #   "qwen3_coder" -> Qwen3CoderToolParser  (XML format: <function=...>)
    tool_parser: NotRequired[str]
    reasoning_parser: NotRequired[str]
    # Enable per-iteration in-flight batching telemetry. When set, the async
    # worker constructs the engine with enable_iter_perf_stats=True and drains
    # get_stats_async() into per-worker inflight-batch-size timelines (the
    # TRT-LLM analog of vLLM's per_worker_inflight_batch_sizes). Requires
    # async_engine=True.
    enable_trtllm_metrics_logger: NotRequired[bool]
    # Poll interval (seconds) / get_stats_async timeout window for the metrics
    # logger. Only used when enable_trtllm_metrics_logger is True.
    trtllm_metrics_logger_interval: NotRequired[float]


class TrtllmConfig(GenerationConfig):
    trtllm_cfg: TrtllmSpecificArgs
    # Escape hatch for arbitrary TRT-LLM LLM/AsyncLLM constructor kwargs not
    # covered by TrtllmSpecificArgs (e.g. sampler_type, enable_attention_dp).
    # Spread into the engine constructor as `**trtllm_kwargs`.
    trtllm_kwargs: NotRequired[dict[str, Any]]
