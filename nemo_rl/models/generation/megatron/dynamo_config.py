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

"""Configuration for serving Megatron prefill/decode engines behind Dynamo.

``policy.generation.dynamo: true`` on a ``backend: megatron`` config is the
switch. ``policy.generation.dynamo_cfg`` tunes the frontend, the timeouts and
the per-role engine layouts; every field carries its default. The replica shape
comes from the backend-neutral ``policy.generation.disaggregation`` block.
"""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, Field, PositiveFloat, PositiveInt

from nemo_rl.models.generation.interfaces import DisaggConfig

# What one engine serves. Megatron engines behind Dynamo always run
# disaggregated: an aggregated Megatron engine is plain ``backend: megatron``.
MegatronDynamoRole = Literal["prefill", "decode"]

# Dynamo component each role registers under: the frontend's prefill router
# sends the prefill leg to "prefill" and the decode leg to "backend".
DYNAMO_COMPONENT_BY_ROLE: dict[str, str] = {"prefill": "prefill", "decode": "backend"}


class EngineLayoutOverrides(BaseModel, extra="forbid"):
    """Parallel layout a prefill or decode engine uses instead of the shared one.

    A field left ``None`` keeps the shared layout for that dimension. The KV
    handoff reshards the cache between the two sides, so prefill and decode
    engines may span different numbers of GPUs.
    """

    tensor_model_parallel_size: PositiveInt | None = None
    pipeline_model_parallel_size: PositiveInt | None = None


class MegatronDynamoFrontendArgs(BaseModel, extra="forbid"):
    """Arguments for the managed ``dynamo.frontend`` process."""

    # "kv" routes each request to the prefill engine holding most of its
    # prefix. No "direct": it needs a worker ID on every request, and NeMo RL's
    # rollouts name none.
    router_mode: Literal[
        "round-robin",
        "random",
        "power-of-two",
        "kv",
        "least-loaded",
        "device-aware-weighted",
    ] = "kv"
    extra_cli_args: list[str] = Field(default_factory=list)


class MegatronDynamoCfg(BaseModel, extra="forbid"):
    """``policy.generation.dynamo_cfg`` for Megatron engines behind Dynamo."""

    # Bounds etcd/NATS/frontend start-up and every engine's registration.
    startup_timeout_s: PositiveFloat = 600.0
    # Deadline for one rollout through the frontend. None waits as long as
    # generation takes, as the native generate_async does.
    request_timeout_s: PositiveFloat | None = None
    # Bounds every control call to an engine: refit, pause, resume, shutdown.
    control_timeout_s: PositiveFloat = 600.0
    frontend_args: MegatronDynamoFrontendArgs = Field(
        default_factory=MegatronDynamoFrontendArgs
    )
    # Per-role layouts, as trtllm_cfg keeps its own prefill_engine / decode_engine.
    prefill_engine: EngineLayoutOverrides = Field(default_factory=EngineLayoutOverrides)
    decode_engine: EngineLayoutOverrides = Field(default_factory=EngineLayoutOverrides)

    def role_overrides(self, role: MegatronDynamoRole) -> dict[str, int]:
        """Layout keys an engine of ``role`` sets on top of the shared layout."""
        overrides = self.prefill_engine if role == "prefill" else self.decode_engine
        return {
            key: value
            for key, value in overrides.model_dump().items()
            if value is not None
        }


def dynamo_enabled(generation_cfg: Mapping[str, Any]) -> bool:
    """Whether ``policy.generation.dynamo`` is on."""
    return bool(generation_cfg.get("dynamo"))


def megatron_dynamo_cfg(generation_cfg: Mapping[str, Any]) -> MegatronDynamoCfg:
    """The validated ``dynamo_cfg`` block; an absent block means all defaults."""
    return MegatronDynamoCfg.model_validate(generation_cfg.get("dynamo_cfg") or {})


def disaggregation_cfg(generation_cfg: Mapping[str, Any]) -> DisaggConfig:
    """The validated backend-neutral ``disaggregation`` block."""
    return DisaggConfig.model_validate(generation_cfg.get("disaggregation") or {})


def engine_width(policy_config: Mapping[str, Any], role: MegatronDynamoRole) -> int:
    """Ranks one engine of ``role`` spans.

    Engines are dedicated inference policies, so their width is the layout
    those policies are built with: ``policy.megatron_cfg`` with the
    ``mcore_generation_config`` overrides applied, then the role's
    ``prefill_engine`` / ``decode_engine`` overrides. This is the only place
    that computes an engine width.

    Raises:
        ValueError: When the role's expert-parallel grid does not fit inside
            one engine.
    """
    # Deferred: the Megatron config module pulls in the policy config types.
    from nemo_rl.models.generation.megatron.config import (
        merged_inference_megatron_cfg,
    )

    generation_cfg = policy_config["generation"]
    overrides = megatron_dynamo_cfg(generation_cfg).role_overrides(role)
    # The same merged layout the role's workers are built with (its
    # context_parallel_size is pinned to 1 by the merge).
    layout = merged_inference_megatron_cfg(
        policy_config,  # type: ignore[arg-type]
        overrides=overrides,
    )
    tp = layout["tensor_model_parallel_size"]
    pp = layout["pipeline_model_parallel_size"]
    width = int(tp * pp * layout["context_parallel_size"])
    # MCore lays the expert grid (ETP x EP x PP) over the same ranks and
    # requires the world size to be divisible by it
    # (megatron/core/parallel_state.py, initialize_model_parallel). One engine
    # is sized by the dense layout, so an expert grid wider than that cannot be
    # built; refuse it here instead of at model build.
    etp = layout["expert_tensor_parallel_size"]
    ep = layout["expert_model_parallel_size"]
    if width % (etp * ep * pp) != 0:
        layout_key = (
            f"policy.generation.dynamo_cfg.{role}_engine or "
            "policy.generation.mcore_generation_config"
            if overrides
            else "policy.generation.mcore_generation_config"
        )
        raise ValueError(
            f"Megatron {role} engines behind Dynamo span {width} ranks "
            f"(tensor_model_parallel_size={tp} x pipeline_model_parallel_size={pp}), "
            f"which is not divisible by expert_tensor_parallel_size={etp} x "
            f"expert_model_parallel_size={ep} x pipeline_model_parallel_size={pp}. "
            f"Set {layout_key} so that tensor_model_parallel_size is a multiple "
            "of expert_tensor_parallel_size x expert_model_parallel_size."
        )
    return width


def engine_widths(policy_config: Mapping[str, Any]) -> dict[MegatronDynamoRole, int]:
    """Ranks one engine spans, per role: ``{"prefill": w_p, "decode": w_d}``."""
    return {
        "prefill": engine_width(policy_config, "prefill"),
        "decode": engine_width(policy_config, "decode"),
    }


def replica_width(
    disaggregation: DisaggConfig, widths: Mapping[MegatronDynamoRole, int]
) -> int:
    """GPUs one prefill/decode replica spans.

    The Dynamo frontend routes across the whole fleet, so a replica only fixes
    the P:D ratio and keeps each group's engines next to each other.
    """
    return (
        disaggregation.num_prefill_engines * widths["prefill"]
        + disaggregation.num_decode_engines * widths["decode"]
    )


def engine_roles(
    disaggregation: DisaggConfig,
    inference_world_size: int,
    widths: Mapping[MegatronDynamoRole, int],
) -> list[MegatronDynamoRole]:
    """Each engine's role, in launch order, for this many GPUs.

    Replicas are tiled prefill engines first, as TRT-LLM disaggregation lays
    them out. Launch order is also refit rank order.
    """
    per_replica = replica_width(disaggregation, widths)
    if inference_world_size % per_replica != 0:
        raise ValueError(
            f"Megatron engines behind Dynamo have {inference_world_size} inference "
            f"GPUs, which is not a multiple of one replica's {per_replica} GPUs "
            "(the sum of the replica's prefill and decode engine widths)."
        )
    replica: list[MegatronDynamoRole] = [
        "prefill" if engine < disaggregation.num_prefill_engines else "decode"
        for engine in range(
            disaggregation.num_prefill_engines + disaggregation.num_decode_engines
        )
    ]
    return replica * (inference_world_size // per_replica)


def validate_megatron_dynamo_generation(generation_cfg: Mapping[str, Any]) -> None:
    """What prefill/decode Megatron engines behind Dynamo can and cannot carry.

    Raises:
        ValueError: For a config whose shape the engines cannot serve.
    """
    assert generation_cfg["backend"] == "megatron" and dynamo_enabled(generation_cfg)
    megatron_dynamo_cfg(generation_cfg)
    disaggregation = disaggregation_cfg(generation_cfg)
    if not disaggregation.enabled:
        raise ValueError(
            "policy.generation.dynamo=true requires "
            "policy.generation.disaggregation.enabled=true; aggregated Megatron "
            "engines behind Dynamo are not supported (leave dynamo false)"
        )
    if disaggregation.num_prefill_engines < 1 or disaggregation.num_decode_engines < 1:
        raise ValueError(
            "policy.generation.disaggregation needs at least one prefill and one "
            "decode engine per replica when policy.generation.dynamo=true"
        )
    # One managed frontend routes the whole fleet and tokenizes itself.
    if disaggregation.num_frontend_workers != 1 or disaggregation.frontend_tokenize:
        raise ValueError(
            "policy.generation.disaggregation.num_frontend_workers must be 1 and "
            "frontend_tokenize false when policy.generation.dynamo=true; the "
            "managed Dynamo frontend routes every replica"
        )
    colocated = generation_cfg.get("colocated")
    if colocated is not None and colocated["enabled"]:
        raise ValueError(
            "policy.generation.colocated.enabled must be false when "
            "policy.generation.dynamo=true; the engines are dedicated inference "
            "policies"
        )
    if generation_cfg.get("refit_transport") is not None:
        raise ValueError(
            "policy.generation.refit_transport must be null when "
            "policy.generation.dynamo=true; the engines refit through the "
            "collective transport"
        )
    for quantization_field in ("quant_cfg", "real_quant"):
        if generation_cfg.get(quantization_field):
            raise ValueError(
                f"policy.generation.{quantization_field} is not supported when "
                "policy.generation.dynamo=true"
            )
    mcore = generation_cfg["mcore_generation_config"]
    if not mcore["enable_prefix_caching"]:
        raise ValueError(
            "mcore_generation_config.enable_prefix_caching must be true for "
            "Megatron prefill/decode engines: decode admission reuses the "
            "prefix-cache allocator"
        )
    if mcore["kv_cache_management_mode"] != "persist":
        raise ValueError(
            "mcore_generation_config.kv_cache_management_mode must be 'persist' "
            "for Megatron prefill/decode engines; offload and recompute "
            "invalidate the registered KV transfer buffers"
        )
    if mcore["expose_http_server"]:
        raise ValueError(
            "mcore_generation_config.expose_http_server must be false when "
            "policy.generation.dynamo=true; the Dynamo frontend is the HTTP surface"
        )
