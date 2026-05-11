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

from typing import Any

from nemo_rl.models.policy import PolicyConfig

# will need to update this for other layer types
_FLEXTRON_MLP_LAYER_TYPES = frozenset(("E",))
_FLEXTRON_EMB_LAYER_TYPES = frozenset(("M", "E", "*"))


def _validate_flextron_config(config: PolicyConfig, model_cfg: Any) -> None:
    """Validate and resolve deterministic Flextron route config."""
    megatron_cfg = config["megatron_cfg"]
    flex_routers = megatron_cfg.get("flex_routers", None)
    sampling_rates = megatron_cfg.get("flextron_sampling_rates", None)
    if flex_routers == []:
        flex_routers = None
    if sampling_rates == []:
        sampling_rates = None

    if flex_routers is None and sampling_rates is None:
        return
    if flex_routers is None:
        raise ValueError(
            "policy.megatron_cfg.flextron_sampling_rates requires "
            "policy.megatron_cfg.flex_routers."
        )
    if sampling_rates is None:
        raise ValueError(
            "policy.megatron_cfg.flex_routers requires "
            "policy.megatron_cfg.flextron_sampling_rates."
        )
    if not isinstance(flex_routers, list):
        raise TypeError("policy.megatron_cfg.flex_routers must be a list.")

    resolved_sampling_rates = _validate_flextron_sampling_rates(
        sampling_rates, expected_len=1 + len(flex_routers)
    )
    _validate_flextron_generation_backend(config, resolved_sampling_rates)
    if not flex_routers:
        model_cfg.flex_routers = []
        model_cfg.flextron_sampling_rates = resolved_sampling_rates
        return

    main_layer_pattern = _get_flextron_main_layer_pattern(model_cfg)
    ffn_hidden_size = _get_flextron_model_dimension(model_cfg, "ffn_hidden_size")
    hidden_size = _get_flextron_model_dimension(model_cfg, "hidden_size")

    resolved_routers = []
    for router_idx, router in enumerate(flex_routers):
        if not isinstance(router, dict):
            raise TypeError(
                f"policy.megatron_cfg.flex_routers[{router_idx}] must be a dict."
            )
        if "mlp_int_list" not in router or "emb_int_list" not in router:
            raise ValueError(
                f"policy.megatron_cfg.flex_routers[{router_idx}] must define "
                "mlp_int_list and emb_int_list."
            )
        resolved_routers.append(
            {
                "mlp_int_list": _resolve_flextron_int_list(
                    router["mlp_int_list"],
                    field_path=f"policy.megatron_cfg.flex_routers[{router_idx}].mlp_int_list",
                    max_value=ffn_hidden_size,
                    main_layer_pattern=main_layer_pattern,
                    eligible_layer_types=_FLEXTRON_MLP_LAYER_TYPES,
                ),
                "emb_int_list": _resolve_flextron_int_list(
                    router["emb_int_list"],
                    field_path=f"policy.megatron_cfg.flex_routers[{router_idx}].emb_int_list",
                    max_value=hidden_size,
                    main_layer_pattern=main_layer_pattern,
                    eligible_layer_types=_FLEXTRON_EMB_LAYER_TYPES,
                ),
            }
        )

    model_cfg.flex_routers = resolved_routers
    model_cfg.flextron_sampling_rates = resolved_sampling_rates


def _validate_flextron_generation_backend(
    config: PolicyConfig, sampling_rates: list[float]
) -> None:
    """Reject backend combinations that cannot honor routed Flextron masks."""
    generation_cfg = config.get("generation", None)
    if generation_cfg is None:
        return
    if generation_cfg.get("backend") != "vllm":
        return
    if any(rate > 0 for rate in sampling_rates[1:]):
        raise ValueError(
            "policy.generation.backend='vllm' cannot be used with nonzero "
            "nested Flextron sampling rates. Use backend='megatron' or set all "
            "policy.megatron_cfg.flextron_sampling_rates[1:] values to 0."
        )


def _validate_flextron_sampling_rates(
    sampling_rates: Any, expected_len: int
) -> list[float]:
    """Validate Flextron sampling rates and return them as floats."""
    if not isinstance(sampling_rates, list):
        raise TypeError("policy.megatron_cfg.flextron_sampling_rates must be a list.")
    if len(sampling_rates) != expected_len:
        raise ValueError(
            "policy.megatron_cfg.flextron_sampling_rates must have length "
            f"{expected_len} (1 + len(policy.megatron_cfg.flex_routers)); got "
            f"{len(sampling_rates)}."
        )

    resolved_rates = []
    for rate_idx, rate in enumerate(sampling_rates):
        if isinstance(rate, bool) or not isinstance(rate, (float, int)):
            raise TypeError(
                "policy.megatron_cfg.flextron_sampling_rates"
                f"[{rate_idx}] must be a number."
            )
        if rate < 0:
            raise ValueError(
                "policy.megatron_cfg.flextron_sampling_rates"
                f"[{rate_idx}] must be non-negative."
            )
        resolved_rates.append(float(rate))

    if sum(resolved_rates) <= 0.0:
        raise ValueError(
            "policy.megatron_cfg.flextron_sampling_rates must sum to a value > 0."
        )
    return resolved_rates


def _get_flextron_main_layer_pattern(model_cfg: Any) -> str:
    """Return the main decoder hybrid pattern without MTP or pipe separators."""
    hybrid_layer_pattern = getattr(model_cfg, "hybrid_layer_pattern", None)
    if not hybrid_layer_pattern:
        raise ValueError(
            "policy.megatron_cfg.flex_routers requires model_cfg.hybrid_layer_pattern."
        )

    main_pattern = hybrid_layer_pattern.split("/", maxsplit=1)[0].replace("|", "")
    if not main_pattern:
        raise ValueError(
            "policy.megatron_cfg.flex_routers requires a non-empty main "
            "hybrid layer pattern."
        )
    return main_pattern


def _get_flextron_model_dimension(model_cfg: Any, attr_name: str) -> int:
    """Read and validate a positive model dimension needed by Flextron."""
    value = getattr(model_cfg, attr_name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            f"policy.megatron_cfg.flex_routers requires model_cfg.{attr_name} "
            "to be a positive integer."
        )
    return value


def _resolve_flextron_int_list(
    value: Any,
    *,
    field_path: str,
    max_value: int,
    main_layer_pattern: str,
    eligible_layer_types: frozenset[str],
) -> list[int]:
    """Expand and project one Flextron route dimension field."""
    if isinstance(value, bool):
        raise TypeError(f"{field_path} must be an int or list[int].")
    if isinstance(value, int):
        _validate_flextron_int_value(value, field_path=field_path, max_value=max_value)
        per_layer_values = [value] * len(main_layer_pattern)
    elif isinstance(value, list):
        if len(value) != len(main_layer_pattern):
            raise ValueError(
                f"{field_path} must be an int or a list with one value per main "
                f"hybrid layer ({len(main_layer_pattern)}); got {len(value)} values."
            )
        per_layer_values = []
        for value_idx, item in enumerate(value):
            item_path = f"{field_path}[{value_idx}]"
            _validate_flextron_int_value(
                item, field_path=item_path, max_value=max_value
            )
            per_layer_values.append(item)
    else:
        raise TypeError(f"{field_path} must be an int or list[int].")

    return [
        item
        for item, layer_type in zip(per_layer_values, main_layer_pattern)
        if layer_type in eligible_layer_types
    ]


def _validate_flextron_int_value(
    value: Any, *, field_path: str, max_value: int
) -> None:
    """Validate one Flextron integer dimension value."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_path} must contain integers.")
    if value < 0 or value > max_value:
        raise ValueError(f"{field_path} must be in [0, {max_value}]; got {value}.")
