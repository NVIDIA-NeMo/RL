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

from typing import Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel

from nemo_rl.models.generation.interfaces import DisaggConfig, GenerationConfig


class TrtllmDisaggServerConfig(BaseModel, extra="allow"):
    """Forwarded verbatim to TRT-LLM's ``DisaggServerConfig`` -- nothing else.

    Not a :class:`DisaggConfig` subclass: the layout is a different thing, and
    inheriting would let this block declare layout keys nothing here reads.
    The ctx/gen spelling is TRT-LLM's; everything NeMo RL owns reads
    prefill/decode.
    """

    # ctx_router must be stateful or a trajectory's later turns reach an engine
    # without its prefix and re-prefill from scratch; the decode side gets KV
    # fresh each turn, so gen_router need not be. Literal because
    # ctx_router="round_robin" parses fine and silently drops that affinity.
    ctx_router: Literal["conversation", "kv_cache_aware"] = "conversation"
    gen_router: Literal["round_robin", "load_balancing"] = "load_balancing"

    # Relay prefill->decode prompt token ids as one base64 int32 string instead
    # of a 30k-int JSON array (DisaggServerConfig.gen_tokids_ctxbytes).
    gen_tokids_ctxbytes: bool = False
    # Strip the conversation history from the decode leg; the relayed token ids
    # carry the full prefix, so the decode adapter never needs the messages
    # (DisaggServerConfig.gen_strip_message_history).
    gen_strip_message_history: bool = False


class TrtllmEngineKnobs(TypedDict, total=False):
    """The settings a replica's prefill and decode engines may disagree on.

    Deliberately narrow: everything else in ``trtllm_cfg`` describes the model
    or the engine's contract with the rest of the system, and two engines of
    one replica differing on those would not serve one coherent policy. All
    optional -- a role block names only what differs, and the shared block's
    required keys fail loudly on their subscript read, not here.
    """

    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_batch_size: int
    max_num_tokens: int
    # TRT-LLM splits the TP dimension on MoE layers, so moe_tp * moe_ep must
    # equal *this engine's* tensor_parallel_size. Partitioning only: the worker
    # count stays TP x PP x DP.
    moe_tensor_parallel_size: int
    moe_expert_parallel_size: int


class TrtllmEngineArgs(TrtllmEngineKnobs):
    """One role's engine overrides: ``trtllm_cfg.{prefill,decode}_engine``."""

    # This role's AsyncLLM kwargs, merged over the replica-wide
    # generation.trtllm_kwargs -- how prefill and decode get different tuning.
    # No counterpart on the shared block: those already live one level up.
    trtllm_kwargs: NotRequired[dict[str, Any]]


class TrtllmSpecificArgs(TrtllmEngineKnobs):
    """``trtllm_cfg``: the per-engine knobs plus what every engine shares."""

    model_name: NotRequired[str]
    max_model_len: int
    precision: str
    expose_http_server: NotRequired[bool]
    async_engine: NotRequired[bool]
    # Duplicated from grpo.async_grpo because the generation backend never sees
    # master_config.grpo. The exemplar interpolates them from there, so they
    # cannot drift.
    in_flight_weight_updates: NotRequired[bool]
    recompute_kv_cache_after_weight_updates: NotRequired[bool]
    default_chat_template_kwargs: NotRequired[dict[str, Any]]
    # TRT-LLM's registered parser names:
    #   "qwen3"       -> Qwen3ToolParser      (JSON format: {"name":..., "arguments":{...}})
    #   "qwen3_coder" -> Qwen3CoderToolParser  (XML format: <function=...>)
    tool_parser: NotRequired[str]
    reasoning_parser: NotRequired[str]

    # Per-role engine overrides under PD disaggregation, merged over the
    # shared values above. Keys outside TrtllmEngineArgs are rejected at
    # startup by TrtllmGeneration._role_overrides.
    prefill_engine: NotRequired[TrtllmEngineArgs]
    decode_engine: NotRequired[TrtllmEngineArgs]

    disagg_server: NotRequired[TrtllmDisaggServerConfig]


class TrtllmConfig(GenerationConfig):
    trtllm_cfg: TrtllmSpecificArgs
    # Escape hatch for arbitrary TRT-LLM LLM/AsyncLLM constructor kwargs not
    # covered by TrtllmSpecificArgs (e.g. sampler_type, enable_attention_dp).
    # Spread into the engine constructor as `**trtllm_kwargs`, and shared by
    # every engine -- a single engine's are trtllm_cfg.{ctx,gen}.trtllm_kwargs,
    # which are merged over these.
    trtllm_kwargs: NotRequired[dict[str, Any]]


def _as_dict(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, BaseModel):
        return raw.model_dump(exclude_unset=True)
    return dict(raw)


def resolve_disagg_layout(config: TrtllmConfig) -> DisaggConfig:
    """``generation.disaggregation`` as its schema.

    Absent is the same as present-and-disabled: every field carries its default
    on :class:`DisaggConfig`, so callers read attributes unconditionally
    instead of re-deriving a default per key at the call site.
    """
    return DisaggConfig.model_validate(_as_dict(config.get("disaggregation")))


def resolve_disagg_server_config(config: TrtllmConfig) -> TrtllmDisaggServerConfig:
    """``trtllm_cfg.disagg_server`` as its schema.

    Kept separate from the layout rather than merged into one object: the two
    blocks answer different questions and go to different places -- the layout
    sizes the engine fleet NeMo RL creates, this is forwarded verbatim to
    TRT-LLM's ``DisaggServerConfig``.
    """
    raw = _as_dict(config["trtllm_cfg"].get("disagg_server"))
    # A layout key here reads as configuration but nothing would apply it, and
    # extra="allow" means pydantic would not complain either.
    misplaced = sorted(set(raw) & set(DisaggConfig.model_fields))
    if misplaced:
        raise ValueError(
            f"{misplaced} set in trtllm_cfg.disagg_server, which only carries "
            f"this backend's OpenAIDisaggServer settings. The disaggregation "
            f"layout belongs in generation.disaggregation."
        )
    return TrtllmDisaggServerConfig.model_validate(raw)
