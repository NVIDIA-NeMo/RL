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
    assert inner["engine_startup_timeout_s"] == 1800
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


@pytest.mark.parametrize(
    "timeout_key",
    ["engine_startup_timeout_s", "refit_timeout_s"],
)
@pytest.mark.parametrize("timeout_s", [0, -1])
def test_operation_timeout_must_be_positive(timeout_key: str, timeout_s: float):
    config = _valid_config()
    _inner(config)[timeout_key] = timeout_s

    with pytest.raises(ValidationError, match=timeout_key):
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


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda inner: inner.update(ep_size=3),
            "must divide tp_size",
        ),
        (
            lambda inner: inner.update(enable_ep_moe=True),
            "does not exist in the pinned SGLang",
        ),
        (
            lambda inner: inner.update(enable_deepep_moe=True),
            "does not exist in the pinned SGLang",
        ),
    ],
)
def test_invalid_moe_parallelism_fails_before_startup(
    mutate: Callable[[dict[str, Any]], None],
    message: str,
):
    config = _valid_config()
    mutate(_inner(config))

    with pytest.raises(ValueError, match=message):
        normalize_sglang_config(config)


@pytest.mark.parametrize(
    "mutate",
    [
        # ep_size must divide tp_size when no a2a backend is in play.
        lambda inner: inner.update(ep_size=2),
        # SGLang's _handle_a2a_moe overrides ep_size to tp_size for every
        # non-"none" backend, so an ep_size it will overwrite must not be
        # rejected here.
        lambda inner: inner.update(moe_a2a_backend="deepep"),
        lambda inner: inner.update(ep_size=2, moe_a2a_backend="deepep"),
        lambda inner: inner.update(moe_a2a_backend="deepep", deepep_mode="low_latency"),
        lambda inner: inner.update(ep_size=2, ep_num_redundant_experts=2),
        # triton_kernel's ep_size==1 assertion lives in SGLang's GPT-OSS
        # model-specific branch, so it must not be enforced generically.
        lambda inner: inner.update(moe_runner_backend="triton_kernel", ep_size=2),
    ],
)
def test_valid_moe_parallelism_is_accepted(mutate: Callable[[dict[str, Any]], None]):
    config = _valid_config()
    mutate(_inner(config))

    normalize_sglang_config(config)


@pytest.mark.parametrize("field", ["moe_a2a_backend", "deepep_mode"])
def test_moe_enum_typos_are_rejected(field: str):
    """SGLang turns these into MoeA2ABackend(...) / DeepEPMode(...) deep inside
    engine startup, where a typo costs a full allocation to discover."""
    config = _valid_config()
    _inner(config)[field] = "depep"

    with pytest.raises(ValidationError, match=field):
        normalize_sglang_config(config)


def test_moe_parallelism_is_unset_by_default():
    """Forwarded only when configured: these ServerArgs names have moved
    between SGLang releases, so a default nobody asked for would make the
    launcher fail on a version that merely renamed a knob we do not use."""
    config = _valid_config()

    normalize_sglang_config(config)

    inner = _inner(config)
    for key in ("moe_a2a_backend", "ep_num_redundant_experts", "deepep_mode"):
        assert inner[key] is None


def test_moe_parallelism_keys_are_forwarded_to_serverargs():
    """Guard the passthrough itself: schema acceptance means nothing if
    `_compute_server_args` never sends the key. `enable_ep_moe` was exactly
    that failure -- schema-visible, never forwarded, silently ignored."""
    import ast
    from pathlib import Path

    from nemo_rl.models.generation.sglang import config as sglang_config

    # Read the worker beside the config module rather than importing it:
    # sglang_worker pulls in the SGLang stack, which the unit env lacks.
    source = Path(sglang_config.__file__).with_name("sglang_worker.py").read_text()
    builder = next(
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "_compute_server_args"
    )
    # Only count keys inside a loop that actually assigns into kwargs, so a
    # dead list of names cannot satisfy this test.
    forwarded: set[str] = set()
    for loop in (n for n in ast.walk(builder) if isinstance(n, ast.For)):
        assigns_kwargs = any(
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Name)
            and node.value.id == "kwargs"
            for stmt in loop.body
            for node in ast.walk(stmt)
        )
        if not assigns_kwargs or not isinstance(loop.iter, ast.List):
            continue
        forwarded.update(
            element.value
            for element in loop.iter.elts
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        )

    for key in ("moe_a2a_backend", "ep_num_redundant_experts", "deepep_mode"):
        assert key in forwarded, f"{key} is configurable but never forwarded"
