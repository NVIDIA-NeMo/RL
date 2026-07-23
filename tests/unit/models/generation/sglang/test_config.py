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

from collections.abc import Callable
from typing import Any, cast

import pytest
from pydantic import ValidationError

from nemo_rl.models.generation.sglang.config import (
    SGLangConfig,
    normalize_sglang_config,
)


def _valid_config() -> SGLangConfig:
    return cast(
        SGLangConfig,
        {
            "sglang_cfg": {
                "model_path": "public/example-model",
                "random_seed": 42,
                "tp_size": 2,
                "dp_size": 1,
                "pp_size": 1,
                "ep_size": 1,
                "skip_server_warmup": True,
                "custom_server_arg": "preserved",
                "sglang_server_config": {
                    "needs_offload": True,
                    "cpu_weight_backup": False,
                    "sglang_server_concurrency": 64,
                    "pause_generation_mode": "retract",
                    "num_gpus": 4,
                    "num_gpus_per_engine": 2,
                },
            }
        },
    )


def _inner(config: SGLangConfig) -> dict[str, Any]:
    return cast(dict[str, Any], config["sglang_cfg"])


def test_normalize_materializes_centralized_defaults_and_preserves_passthrough():
    config = _valid_config()

    normalized = normalize_sglang_config(config)

    inner = _inner(config)
    assert normalized.quantization.scheme == "bf16"
    assert inner["quantization"] == {
        "scheme": "bf16",
        "weight_block_size": None,
        "scale_fmt": None,
        "modules_to_not_convert": [],
        "extra_high_precision_layers_hf": [],
        "num_layers_at_start_in_bf16": 0,
        "num_layers_at_end_in_bf16": 0,
        "converted_model_path": None,
        "cache_root": None,
    }
    assert inner["context_length"] is None
    assert inner["use_fault_tolerance"] is False
    assert inner["rollout_health_check_interval"] is None
    assert inner["rollout_health_check_timeout"] is None
    assert inner["rollout_health_check_first_wait"] is None
    assert inner["refit_timeout_s"] == 1800
    assert inner["sglang_router_config"]["use_external_router"] is False
    assert inner["custom_server_arg"] == "preserved"


@pytest.mark.parametrize(
    ("colocated_inference", "expected_mode"),
    [(True, "ipc"), (False, "broadcast")],
)
def test_weight_transfer_mode_is_derived_from_topology(
    colocated_inference: bool,
    expected_mode: str,
):
    config = _valid_config()

    normalize_sglang_config(
        config,
        colocated_inference=colocated_inference,
    )

    assert (
        _inner(config)["sglang_server_config"]["weight_transfer_mode"] == expected_mode
    )


def test_matching_weight_transfer_hint_is_accepted():
    config = _valid_config()
    _inner(config)["sglang_server_config"]["weight_transfer_mode"] = "broadcast"

    normalize_sglang_config(config, colocated_inference=False)

    assert _inner(config)["sglang_server_config"]["weight_transfer_mode"] == "broadcast"


def test_normalization_is_idempotent_across_setup_and_constructor_boundaries():
    config = _valid_config()

    normalize_sglang_config(config, colocated_inference=False)
    normalize_sglang_config(config)

    assert _inner(config)["sglang_server_config"]["weight_transfer_mode"] == "broadcast"
    assert _inner(config)["quantization"]["scheme"] == "bf16"


def test_conflicting_weight_transfer_hint_fails_before_startup():
    config = _valid_config()
    _inner(config)["sglang_server_config"]["weight_transfer_mode"] = "ipc"

    with pytest.raises(
        ValueError,
        match=r"conflicts with colocated\.enabled=False.*requires 'broadcast'",
    ):
        normalize_sglang_config(config, colocated_inference=False)


@pytest.mark.parametrize(
    "missing_key",
    [
        "model_path",
        "random_seed",
        "tp_size",
        "dp_size",
        "pp_size",
        "ep_size",
        "skip_server_warmup",
    ],
)
def test_directly_consumed_runtime_keys_are_required(missing_key: str):
    config = _valid_config()
    del _inner(config)[missing_key]

    with pytest.raises(ValidationError, match=missing_key):
        normalize_sglang_config(config)


@pytest.mark.parametrize(
    "missing_key",
    [
        "needs_offload",
        "cpu_weight_backup",
        "sglang_server_concurrency",
        "pause_generation_mode",
        "num_gpus",
        "num_gpus_per_engine",
    ],
)
def test_directly_consumed_server_keys_are_required(missing_key: str):
    config = _valid_config()
    del _inner(config)["sglang_server_config"][missing_key]

    with pytest.raises(ValidationError, match=missing_key):
        normalize_sglang_config(config)


@pytest.mark.parametrize(
    "missing_key",
    [
        "rollout_health_check_interval",
        "rollout_health_check_timeout",
        "rollout_health_check_first_wait",
    ],
)
def test_fault_tolerance_requires_every_health_setting(missing_key: str):
    config = _valid_config()
    inner = _inner(config)
    inner.update(
        {
            "use_fault_tolerance": True,
            "rollout_health_check_interval": 10,
            "rollout_health_check_timeout": 5,
            "rollout_health_check_first_wait": 0,
        }
    )
    del inner[missing_key]

    with pytest.raises(ValueError, match=missing_key):
        normalize_sglang_config(config)


def test_fault_tolerance_accepts_explicit_nonnegative_first_wait():
    config = _valid_config()
    _inner(config).update(
        {
            "use_fault_tolerance": True,
            "rollout_health_check_interval": 10,
            "rollout_health_check_timeout": 5,
            "rollout_health_check_first_wait": 0,
        }
    )

    normalize_sglang_config(config)

    assert _inner(config)["rollout_health_check_first_wait"] == 0


@pytest.mark.parametrize("timeout_s", [0, -1])
def test_refit_timeout_must_be_positive(timeout_s: float):
    config = _valid_config()
    _inner(config)["refit_timeout_s"] = timeout_s

    with pytest.raises(ValidationError, match="refit_timeout_s"):
        normalize_sglang_config(config)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda inner: inner.update(pp_size=2),
            "pp_size must be 1",
        ),
        (
            lambda inner: inner["sglang_server_config"].update(num_gpus_per_engine=1),
            "num_gpus_per_engine must equal tp_size",
        ),
        (
            lambda inner: inner["sglang_server_config"].update(num_gpus=3),
            "num_gpus must be divisible",
        ),
    ],
)
def test_invalid_engine_topology_fails_before_startup(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
):
    config = _valid_config()
    mutate(_inner(config))

    with pytest.raises(ValueError, match=message):
        normalize_sglang_config(config)


def test_external_router_requires_complete_endpoint():
    config = _valid_config()
    _inner(config)["sglang_router_config"] = {
        "use_external_router": True,
        "sglang_router_ip": "127.0.0.1",
    }

    with pytest.raises(ValueError, match="sglang_router_port"):
        normalize_sglang_config(config)
