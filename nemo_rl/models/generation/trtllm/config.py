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

from nemo_rl.models.generation.interfaces import DisaggConfig, GenerationConfig


class TrtllmDisaggServerConfig(DisaggConfig):
    """The layout plus TRT-LLM's ``OpenAIDisaggServer`` settings.

    The layout -- engines per replica, frontend count, frontend tokenization --
    is inherited from :class:`DisaggConfig`. What this adds is only what the
    frontend process needs: which routers it runs and how it thins the
    prefill->decode relay. Everything that configures an *engine* is engine
    config: the per-role overrides are ``trtllm_cfg.prefill_engine`` /
    ``trtllm_cfg.decode_engine``, and the KV cache transceiver is
    ``trtllm_kwargs.cache_transceiver_config`` like any other AsyncLLM argument.

    This backend additionally caps the inherited ``num_frontend_workers``:
    ``num_replicas * num_frontend_workers`` must be <= 256, the snowflake
    node_id space each frontend mints request ids from.

    Requires non-colocated generation: colocated sleeps the engines between
    rollouts, and a replica's prefill and decode engines must be resident
    together for the KV transceiver to work.
    """

    # Routing inside a replica, decided entirely by the disagg server.
    #
    # These four keep TRT-LLM's own spelling of the two legs (ctx/gen) because
    # they are passed straight through to DisaggServerConfig; everything NeMo RL
    # owns reads prefill/decode.
    #
    # The ctx router must be *stateful* so a trajectory's turns return to
    # the engine holding its prefix -- that engine accumulates the prefix across
    # turns and only prefills the delta, so sending a later turn elsewhere
    # throws the work away.
    #
    # The gen router need not be: a decode engine receives KV freshly
    # from the prefill engine on every turn, so it has nothing worth returning
    # to, and a wrong load guess only costs transient skew. Keeping it stateless
    # also keeps placement local, with no coordinator process.
    #
    # Both are ``Literal`` rather than ``str`` because a plausible-but-wrong
    # value is the dangerous case: ``ctx_router="round_robin"`` parses fine and
    # silently throws away the prefix affinity the prefill engines depend on.
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

    Deliberately narrow. Everything else in ``trtllm_cfg`` describes the *model*
    (``precision``, ``max_model_len``, the parsers) or the engine's contract
    with the rest of the system (``async_engine``, ``expose_http_server``, the
    refit flags); two engines of one replica differing on any of those would not
    serve one coherent policy. What is left is how wide the engine is, how its
    experts are split, and how much memory and batch it gets.

    Shared by the two blocks below, so neither has to restate them and they
    cannot drift. All optional: a role block names only what differs, and the
    shared block's required-in-practice keys (``tensor_parallel_size``,
    ``max_batch_size``, ``max_num_tokens``) are enforced by the exemplar YAML
    plus a direct subscript read that fails loudly, not by this type.
    """

    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_batch_size: int
    max_num_tokens: int
    # MoE expert parallelism. TRT-LLM splits the TP dimension on MoE layers
    # into moe_tp × moe_ep, so the constraint is
    #     moe_tensor_parallel_size * moe_expert_parallel_size == tensor_parallel_size
    # The outer worker count is unchanged (still TP × PP × DP) — these only
    # affect how MoE expert weights are partitioned inside each TP rank. Under
    # disaggregation it is checked against *this engine's* TP, since both the TP
    # and the split can differ between prefill and decode.
    moe_tensor_parallel_size: int
    moe_expert_parallel_size: int


class TrtllmEngineArgs(TrtllmEngineKnobs):
    """One role's engine overrides: ``trtllm_cfg.{prefill,decode}_engine``."""

    # Raw TRT-LLM constructor kwargs for this role's engines, merged over the
    # replica-wide ``generation.trtllm_kwargs``. This is how prefill and decode
    # get different engine tuning -- a separate ``kv_cache_config``, say.
    #
    # The shared block has no counterpart on purpose: its engine kwargs already
    # have a home one level up at ``generation.trtllm_kwargs``, and a second
    # spelling under ``trtllm_cfg`` would be two places to look for the same
    # thing.
    trtllm_kwargs: NotRequired[dict[str, Any]]


class TrtllmSpecificArgs(TrtllmEngineKnobs):
    """``trtllm_cfg``: the per-engine knobs plus what every engine shares."""

    model_name: NotRequired[str]
    max_model_len: int
    precision: str
    expose_http_server: NotRequired[bool]
    async_engine: NotRequired[bool]
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


def resolve_trtllm_disagg_config(config: TrtllmConfig) -> TrtllmDisaggServerConfig:
    """Merge the shared layout with this backend's disaggregation mechanics.

    The two are configured apart on purpose: ``generation.disaggregation`` is
    the layout any backend would need (engine counts, frontends), while
    ``trtllm_cfg.disagg_server`` is what the frontend process needs (its
    routers, its relay thinning). Callers want one object, so the two are
    merged here -- the split is a property of the config file, not of the code
    reading it.

    Absent is the same as present-and-disabled: every field carries its default
    on :class:`TrtllmDisaggServerConfig`, so callers read attributes
    unconditionally instead of re-deriving a default per key at the call site.
    """

    def _as_dict(raw: Any) -> dict[str, Any]:
        if raw is None:
            return {}
        if isinstance(raw, DisaggConfig):
            return raw.model_dump(exclude_unset=True)
        return dict(raw)

    layout = _as_dict(config.get("disaggregation"))
    mechanics = _as_dict(config["trtllm_cfg"].get("disagg_server"))
    # A key in both would make the winner depend on merge order, which is
    # exactly the kind of silent mismatch the split is meant to prevent.
    both = sorted(set(layout) & set(mechanics))
    if both:
        raise ValueError(
            f"{both} set in both generation.disaggregation and "
            f"trtllm_cfg.disagg_server. The first holds the backend-agnostic "
            f"layout, the second this backend's mechanics; keep each key in "
            f"exactly one."
        )
    return TrtllmDisaggServerConfig.model_validate({**layout, **mechanics})
