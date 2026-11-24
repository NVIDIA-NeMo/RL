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
    resolve_disagg_layout,
)
from nemo_rl.models.generation.dynamo.config import dynamo_cfg, dynamo_enabled
from nemo_rl.models.generation.megatron.config import (
    ENGINE_ROLE_KEYS,
    EngineLayoutOverrides,
    merged_inference_megatron_cfg,
)


def role_overrides(generation_cfg: Mapping[str, Any], role: Role) -> dict[str, int]:
    """Layout keys an engine of ``role`` sets on top of the shared layout.

    Read from ``mcore_generation_config.{role}_engine``; an absent block keeps
    the shared layout.
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


def reject_megatron_dynamo_outside_single_controller(
    generation_config: Mapping[str, Any],
) -> None:
    """Refuse Megatron-behind-Dynamo keys on an entrypoint that cannot serve them.

    Only the SingleController builds Megatron engines behind Dynamo and runs
    ``MegatronGeneration.validate_settings``; on the other entrypoints these keys
    would be ignored and the run would train plain Megatron generation.

    Raises:
        NotImplementedError: ``backend: megatron`` with ``dynamo: true``.
        ValueError: ``mcore_generation_config.prefill_engine`` / ``decode_engine``
            with ``dynamo`` off.
    """
    if generation_config["backend"] != "megatron":
        return
    if dynamo_enabled(generation_config):
        raise NotImplementedError(
            "policy.generation.dynamo=true with backend=megatron runs only under "
            "the SingleController (examples/run_grpo_single_controller.py); use "
            "that entrypoint or remove policy.generation.dynamo"
        )
    reject_role_layouts_without_dynamo(generation_config)


def reject_role_layouts_without_dynamo(generation_config: Mapping[str, Any]) -> None:
    """Refuse per-role engine layouts on a Megatron config with ``dynamo`` off.

    Only Megatron engines behind Dynamo read them; without it they are dropped
    from the inference layout without a word.

    Raises:
        ValueError: ``mcore_generation_config.prefill_engine`` / ``decode_engine``
            is set.
    """
    mcore_generation_config = generation_config.get("mcore_generation_config") or {}
    if any(key in mcore_generation_config for key in ENGINE_ROLE_KEYS):
        raise ValueError(
            "policy.generation.mcore_generation_config.prefill_engine / "
            "decode_engine size Megatron engines behind Dynamo and are ignored "
            "without it. Set policy.generation.dynamo=true, or remove them."
        )


def validate_megatron_dynamo_generation(generation_cfg: Mapping[str, Any]) -> None:
    """What a prefill/decode fleet of Megatron engines can and cannot carry.

    Raises:
        ValueError: For a config whose shape the fleet cannot serve.
    """
    assert generation_cfg["backend"] == "megatron" and dynamo_enabled(generation_cfg)
    # The same schema as vLLM behind Dynamo: one managed frontend for both.
    dynamo_cfg(generation_cfg)
    for role in ROLES:
        role_overrides(generation_cfg, role)
    if "_dynamo_role" in generation_cfg:
        raise ValueError(
            "policy.generation._dynamo_role is set per engine by the Dynamo engine "
            "fleet; remove it from the config"
        )
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
