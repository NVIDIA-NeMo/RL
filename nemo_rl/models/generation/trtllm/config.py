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

from typing import Any, Literal, NotRequired, TypedDict, cast

from pydantic import BaseModel

from nemo_rl.models.generation.interfaces import DisaggConfig, GenerationConfig


class TrtllmDisaggServerConfig(BaseModel, extra="forbid"):
    """The ``DisaggServerConfig`` fields NeMo RL forwards to TRT-LLM.

    ``extra="forbid"``: ``build_config`` forwards only the fields declared
    here, so any other ``DisaggServerConfig`` key (``max_retries``, ...) would
    pass validation and then silently do nothing.

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
    # "auto" | "dummy", mirroring vllm_cfg.load_format. Set by
    # configure_generation_config, not by the recipe.
    load_format: NotRequired[str]
    # TRT-LLM splits the TP dimension on MoE layers, so moe_tp * moe_ep must
    # equal *this engine's* tensor_parallel_size. Partitioning only: the worker
    # count stays TP x PP x DP.
    moe_tensor_parallel_size: int
    moe_expert_parallel_size: int


class TrtllmEngineArgs(TrtllmEngineKnobs):
    """One role's engine overrides: ``trtllm_cfg.{prefill,decode}_engine``."""

    # This role's AsyncLLM kwargs, deep-merged over the replica-wide
    # generation.trtllm_kwargs -- how prefill and decode get different tuning.
    # No counterpart on the shared block: those already live one level up.
    trtllm_kwargs: NotRequired[dict[str, Any]]


class TrtllmSpecificArgs(TrtllmEngineKnobs):
    """``trtllm_cfg``: the per-engine knobs plus what every engine shares."""

    model_name: NotRequired[str]
    max_model_len: int
    precision: str
    # With precision="fp8", use MXFP8 (UE8M0 1x32 scales) instead of 128x128 block-FP8.
    is_mx: NotRequired[bool]
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

    # Per-role engine overrides under PD disaggregation. As written, a role
    # names only what differs from the shared values above, and keys outside
    # TrtllmEngineArgs are rejected at startup. resolve_role_overrides then
    # replaces each with that role's complete engine config, which is all
    # anything downstream reads.
    prefill_engine: NotRequired[TrtllmEngineArgs]
    decode_engine: NotRequired[TrtllmEngineArgs]

    disagg_server: NotRequired[TrtllmDisaggServerConfig]


class TrtllmConfig(GenerationConfig):
    trtllm_cfg: TrtllmSpecificArgs
    # Escape hatch for arbitrary TRT-LLM LLM/AsyncLLM constructor kwargs not
    # covered by TrtllmSpecificArgs (e.g. sampler_type, enable_attention_dp).
    # Spread into the engine constructor as `**trtllm_kwargs`, and shared by
    # every engine -- a single engine's are trtllm_cfg.{prefill,decode}_engine.trtllm_kwargs,
    # which are deep-merged over these.
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


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """``update`` over ``base``, nested dicts merged key by key; inputs untouched."""
    merged = dict(base)
    for key, value in update.items():
        if isinstance(merged.get(key), dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_role_overrides(config: TrtllmConfig) -> TrtllmConfig:
    """Resolve ``trtllm_cfg.{prefill,decode}_engine`` into complete engine configs.

    The one place the "a role wins over the shared value" rule is applied. As
    written, a role block names only what differs (``TrtllmEngineArgs``).
    Afterwards each is that role's whole engine config: the shared
    ``trtllm_cfg`` with the role's keys on top, and ``trtllm_kwargs`` deep-merged
    over ``generation.trtllm_kwargs``, so a role's nested block (say
    ``kv_cache_config: {max_tokens: N}``) adds to the shared one instead of
    replacing it. Everything downstream reads the resolved block and merges
    nothing itself.

    A role key outside ``TrtllmEngineArgs`` is rejected: the rest of
    ``trtllm_cfg`` describes the model or the engine's contract with the rest of
    the system, so a role differing on one would not serve the same policy as
    its replica-mate. Silently dropping a knob someone wrote down would be
    worse than refusing to start.

    Returns ``config`` unchanged without disaggregation, where the role blocks
    are inert. Idempotent: a resolved block only repeats the shared values
    outside ``TrtllmEngineArgs``, and those pass the check.
    """
    if not resolve_disagg_layout(config).enabled:
        return config

    # A plain dict here: the role blocks are addressed by computed key.
    trtllm_cfg = cast(dict[str, Any], config["trtllm_cfg"])
    shared = {
        k: v
        for k, v in trtllm_cfg.items()
        if k not in ("prefill_engine", "decode_engine", "disagg_server")
    }
    allowed = TrtllmEngineArgs.__required_keys__ | TrtllmEngineArgs.__optional_keys__
    resolved = dict(trtllm_cfg)
    for role in ("prefill", "decode"):
        overrides = dict(trtllm_cfg.get(f"{role}_engine") or {})
        unsupported = sorted(
            k for k in set(overrides) - allowed if overrides[k] != shared.get(k)
        )
        if unsupported:
            raise ValueError(
                f"trtllm_cfg.{role}_engine may only override {sorted(allowed)}, "
                f"got {unsupported}. Settings shared by every engine of the "
                f"replica belong on trtllm_cfg itself, and engine kwargs shared "
                f"by both roles on generation.trtllm_kwargs."
            )
        resolved[f"{role}_engine"] = {
            **shared,
            **overrides,
            "trtllm_kwargs": _deep_merge(
                dict(config.get("trtllm_kwargs") or {}),
                dict(overrides.get("trtllm_kwargs") or {}),
            ),
        }
    return cast(TrtllmConfig, {**config, "trtllm_cfg": resolved})


def role_engine_config(
    config: TrtllmConfig, role: Literal["prefill", "decode"]
) -> dict[str, Any]:
    """The whole engine config of ``role``. ``config`` must be role-resolved.

    See :func:`resolve_role_overrides`.
    """
    return cast(dict[str, Any], config["trtllm_cfg"])[f"{role}_engine"]


def role_tensor_parallel_size(
    config: TrtllmConfig, role: Literal["prefill", "decode"]
) -> int:
    """TP of one engine of ``role``. ``config`` must be role-resolved."""
    return int(role_engine_config(config, role)["tensor_parallel_size"])


def replica_width(config: TrtllmConfig) -> int:
    """GPUs one rollout replica spans.

    Without disaggregation a replica is one engine, ``tensor_parallel_size``
    wide (PP is always 1). Under it, a replica is its prefill engines plus its
    decode engines -- the unit that must stay inside one NVLink domain, since
    the KV cache crosses from prefill to decode on every turn.
    """
    layout = resolve_disagg_layout(config)
    if not layout.enabled:
        return int(config["trtllm_cfg"]["tensor_parallel_size"])
    return layout.num_prefill_engines * role_tensor_parallel_size(
        config, "prefill"
    ) + layout.num_decode_engines * role_tensor_parallel_size(config, "decode")


def resolve_disagg_server_config(config: TrtllmConfig) -> TrtllmDisaggServerConfig:
    """``trtllm_cfg.disagg_server`` as its schema.

    Kept separate from the layout rather than merged into one object: the two
    blocks answer different questions and go to different places -- the layout
    sizes the engine fleet NeMo RL creates, this is forwarded to TRT-LLM's
    ``DisaggServerConfig``.
    """
    raw = _as_dict(config["trtllm_cfg"].get("disagg_server"))
    # Caught ahead of extra="forbid" only to point at where the key belongs.
    misplaced = sorted(set(raw) & set(DisaggConfig.model_fields))
    if misplaced:
        raise ValueError(
            f"{misplaced} set in trtllm_cfg.disagg_server, which only carries "
            f"this backend's OpenAIDisaggServer settings. The disaggregation "
            f"layout belongs in generation.disaggregation."
        )
    return TrtllmDisaggServerConfig.model_validate(raw)
