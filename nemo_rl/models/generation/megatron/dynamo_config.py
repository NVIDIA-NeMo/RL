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
switch. ``policy.generation.dynamo_cfg`` is the same block as for vLLM
(``nemo_rl.models.generation.dynamo.config.DynamoCfg``): it tunes the frontend
and the timeouts, and every field carries its default. The replica shape comes
from the backend-neutral ``policy.generation.disaggregation`` block, and each role's
engine layout from ``mcore_generation_config.prefill_engine`` / ``decode_engine``.
"""

from collections.abc import Mapping
from typing import Any

from nemo_rl.models.generation.disagg_layout import (
    ROLES,
    Role,
    plan_engines,
    replica_width_from,
    resolve_disagg_layout,
)
from nemo_rl.models.generation.dynamo.config import dynamo_cfg, dynamo_enabled
from nemo_rl.models.generation.megatron.config import (
    EngineLayoutOverrides,
    merged_inference_megatron_cfg,
)

# How the engines take new weights. None is NeMo RL's packed collective: train
# rank zero exports HF tensors and broadcasts them to every engine rank, which
# imports them into Megatron. "mcore" is Megatron-Core's native refit, where
# each training rank sends its own shards straight to the engine ranks that
# hold them, as the aggregated Megatron engines do.
DYNAMO_REFIT_TRANSPORTS: tuple[str | None, ...] = (None, "mcore")
# Copy services the native refit may use behind Dynamo.
DYNAMO_MCORE_REFIT_BACKENDS: tuple[str, ...] = ("nccl", "gloo")


def role_overrides(generation_cfg: Mapping[str, Any], role: Role) -> dict[str, int]:
    """Keys an engine of ``role`` sets on top of ``mcore_generation_config``.

    Read from ``mcore_generation_config.{role}_engine``; an absent block keeps
    the shared values.
    """
    raw = generation_cfg["mcore_generation_config"].get(f"{role}_engine")
    overrides = EngineLayoutOverrides.model_validate(raw or {})
    return {
        key: value for key, value in overrides.model_dump().items() if value is not None
    }


def engine_width(policy_config: Mapping[str, Any], role: Role) -> int:
    """Ranks one engine of ``role`` spans.

    Engines are dedicated inference policies, so their width is the layout
    those policies are built with: ``policy.megatron_cfg`` with the
    ``mcore_generation_config`` overrides applied, then the role's
    ``mcore_generation_config.prefill_engine`` / ``decode_engine`` overrides.
    This is the only place
    that computes an engine width.

    Raises:
        ValueError: When the role's expert-parallel grid does not fit inside
            one engine.
    """
    overrides = role_overrides(policy_config["generation"], role)
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
            f"policy.generation.mcore_generation_config.{role}_engine or "
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


def engine_widths(policy_config: Mapping[str, Any]) -> dict[Role, int]:
    """Ranks one engine spans, per role: ``{"prefill": w_p, "decode": w_d}``."""
    return {
        "prefill": engine_width(policy_config, "prefill"),
        "decode": engine_width(policy_config, "decode"),
    }


def replica_width(policy_config: Mapping[str, Any]) -> int:
    """GPUs one prefill/decode replica of Megatron engines spans.

    The Dynamo frontend routes across the whole fleet, so a replica only fixes
    the P:D ratio and keeps each group's engines next to each other. It is not
    an NVLink unit: ``MegatronGeneration.nvlink_domain_span`` is the widest
    engine.
    """
    widths = engine_widths(policy_config)
    return replica_width_from(
        resolve_disagg_layout(policy_config["generation"]),
        prefill_width=widths["prefill"],
        decode_width=widths["decode"],
    )


def plan_megatron_engines(
    policy_config: Mapping[str, Any], world_size: int
) -> tuple[list[Role], list[int]]:
    """Role and width of every engine, in launch (refit rank) order.

    Each replica contributes its prefill engines, then its decode engines, as
    ``disagg_layout.plan_engines`` lays them out; the replica count follows
    from ``world_size``.

    Args:
        policy_config: The full ``policy`` config, already checked by
            :func:`validate_megatron_dynamo_generation`.
        world_size: GPUs in the inference cluster.

    Returns:
        ``(roles, widths)``, one entry per engine.

    Raises:
        ValueError: When the engines do not tile ``world_size`` GPUs.
    """
    layout = resolve_disagg_layout(policy_config["generation"])
    # validate_megatron_dynamo_generation refuses aggregated engines, so the
    # planner never returns its "aggregated" role here.
    assert layout.enabled, "Megatron engines behind Dynamo are always disaggregated"
    widths = engine_widths(policy_config)
    roles, per_engine_widths, _ = plan_engines(
        world_size,
        layout=layout,
        # Unused with disaggregation enabled.
        aggregated_width=widths["prefill"],
        prefill_width=widths["prefill"],
        decode_width=widths["decode"],
        backend_cfg_key="policy.generation.mcore_generation_config",
    )
    planned_roles: list[Role] = []
    for role in roles:
        assert role == "prefill" or role == "decode", role
        planned_roles.append(role)
    return planned_roles, per_engine_widths


def validate_megatron_dynamo_generation(generation_cfg: Mapping[str, Any]) -> None:
    """What prefill/decode Megatron engines behind Dynamo can and cannot carry.

    Raises:
        ValueError: For a config whose shape the engines cannot serve.
    """
    assert generation_cfg["backend"] == "megatron" and dynamo_enabled(generation_cfg)
    # The same schema as vLLM behind Dynamo: one managed frontend for both.
    dynamo_cfg(generation_cfg)
    for role in ROLES:
        role_overrides(generation_cfg, role)
    disaggregation = resolve_disagg_layout(generation_cfg)
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
    refit_transport = generation_cfg.get("refit_transport")
    if refit_transport not in DYNAMO_REFIT_TRANSPORTS:
        raise ValueError(
            "policy.generation.refit_transport must be null (packed collective) "
            "or 'mcore' (native Megatron-Core refit) when "
            f"policy.generation.dynamo=true, got {refit_transport!r}"
        )
    refit_backend = generation_cfg["mcore_generation_config"].get("refit_backend")
    if refit_transport == "mcore" and refit_backend not in DYNAMO_MCORE_REFIT_BACKENDS:
        # nccl_m2n and nvshmem are left out: neither has been run against
        # engines of different widths, and nvshmem also needs a pre-transfer
        # initialization step these engines do not have.
        raise ValueError(
            "policy.generation.mcore_generation_config.refit_backend must be "
            f"one of {list(DYNAMO_MCORE_REFIT_BACKENDS)} when "
            "policy.generation.dynamo=true and refit_transport='mcore', got "
            f"{refit_backend!r}"
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
