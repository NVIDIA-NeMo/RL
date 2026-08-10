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


class TrtllmDisaggArgs(TypedDict):
    """Prefill/decode disaggregation.

    A *replica* is ``num_context_engines`` context engines plus
    ``num_generation_engines`` generation engines, fronted by one
    ``OpenAIDisaggServer`` that exposes the single URL NeMo-Gym talks to. The
    replica *count* is not configured: it follows from the inference cluster's
    size, the same way the DP-shard count does without disaggregation.

    Requires non-colocated generation: colocated sleeps the engines between
    rollouts, and a replica's context and generation engines must be resident
    together for the KV transceiver to work.
    """

    enabled: bool

    # Engines per replica. The two are independent, so the P:D ratio is free.
    num_context_engines: int
    num_generation_engines: int

    # Routing inside a replica, decided entirely by the disagg server.
    #
    # The context router must be *stateful* so a trajectory's turns return to
    # the engine holding its prefix -- that engine accumulates the prefix across
    # turns and only prefills the delta, so sending a later turn elsewhere
    # throws the work away.
    #
    # The generation router need not be: a generation engine receives KV freshly
    # from the context engine on every turn, so it has nothing worth returning
    # to, and a wrong load guess only costs transient skew. Keeping it stateless
    # also keeps placement local, with no coordinator process.
    ctx_router: str  # conversation | kv_cache_aware
    gen_router: str  # round_robin | load_balancing

    # Mapped onto TRT-LLM's CacheTransceiverConfig.
    # DEFAULT | UCX | NIXL | MOONCAKE | MPI
    cache_transceiver_backend: str
    max_tokens_in_buffer: NotRequired[int]

    # Per-role overrides merged over trtllm_cfg. Any trtllm_cfg key goes here --
    # tensor_parallel_size and the MoE split are the ones that usually differ,
    # and each role must satisfy moe_tp * moe_ep == its own TP. A role may also
    # carry its own ``trtllm_kwargs`` (including ``kv_cache_config``) when
    # prefill and decode want different engine tuning.
    ctx_trtllm_kwargs: NotRequired[dict[str, Any]]
    gen_trtllm_kwargs: NotRequired[dict[str, Any]]


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
    disaggregation: NotRequired[TrtllmDisaggArgs]
    default_chat_template_kwargs: NotRequired[dict[str, Any]]
    # TRT-LLM's registered parser names:
    #   "qwen3"       -> Qwen3ToolParser      (JSON format: {"name":..., "arguments":{...}})
    #   "qwen3_coder" -> Qwen3CoderToolParser  (XML format: <function=...>)
    tool_parser: NotRequired[str]
    reasoning_parser: NotRequired[str]


class TrtllmConfig(GenerationConfig):
    trtllm_cfg: TrtllmSpecificArgs
    # Escape hatch for arbitrary TRT-LLM LLM/AsyncLLM constructor kwargs not
    # covered by TrtllmSpecificArgs (e.g. sampler_type, enable_attention_dp).
    # Spread into the engine constructor as `**trtllm_kwargs`.
    trtllm_kwargs: NotRequired[dict[str, Any]]
