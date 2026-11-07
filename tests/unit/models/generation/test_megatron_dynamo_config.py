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

"""Megatron prefill/decode engines behind Dynamo: the config switch and widths."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from nemo_rl.models.generation import resolve_generation_class
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.megatron import MegatronGeneration
from nemo_rl.models.generation.megatron.dynamo_generation import (
    MegatronDynamoGeneration,
)
from nemo_rl.models.generation.megatron.dynamo_config import (
    MegatronDynamoCfg,
    disaggregation_cfg,
    dynamo_enabled,
    engine_roles,
    engine_width,
    engine_widths,
    megatron_dynamo_cfg,
    replica_width,
    validate_megatron_dynamo_generation,
)

pytestmark = pytest.mark.mcore

_DELETE = object()
MEGATRON_POLICY = {
    "model_name": "Qwen/Qwen3-0.6B",
    "megatron_cfg": {
        "enabled": True,
        "tensor_model_parallel_size": 2,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "expert_model_parallel_size": 1,
    },
}


def _megatron_config(**overrides) -> dict:
    """A ``backend: megatron`` generation config serving P/D engines behind Dynamo."""
    config = {
        "backend": "megatron",
        "dynamo": True,
        "model_name": "Qwen/Qwen3-0.6B",
        "dynamo_cfg": {},
        "refit_transport": None,
        "disaggregation": {
            "enabled": True,
            "num_prefill_engines": 1,
            "num_decode_engines": 2,
        },
        "mcore_generation_config": {
            "enable_prefix_caching": True,
            "kv_cache_management_mode": "persist",
            "expose_http_server": False,
        },
        "colocated": {"enabled": False},
    }
    config.update(overrides)
    return config


def _policy(generation: dict | None = None) -> dict:
    return {**MEGATRON_POLICY, "generation": generation or _megatron_config()}


def _set(*path, value):
    """Mutator that assigns (or deletes) one nested key of a config dict."""

    def mutate(config: dict) -> None:
        node = config
        for key in path[:-1]:
            node = node[key]
        if value is _DELETE:
            del node[path[-1]]
        else:
            node[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    ("generation_cfg", "expected"),
    [
        ({"backend": "megatron", "dynamo": True}, True),
        ({"backend": "megatron", "dynamo": False}, False),
        ({"backend": "megatron"}, False),
        ({}, False),
    ],
    ids=["on", "off", "absent", "empty"],
)
def test_dynamo_enabled_reads_the_flag(generation_cfg, expected) -> None:
    assert dynamo_enabled(generation_cfg) is expected


def test_dynamo_cfg_defaults_and_rejects_unknown_keys() -> None:
    cfg = megatron_dynamo_cfg({})
    assert cfg == MegatronDynamoCfg()
    assert cfg.frontend_args.router_mode == "kv" and cfg.request_timeout_s is None
    assert cfg.role_overrides("prefill") == {} and cfg.role_overrides("decode") == {}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        megatron_dynamo_cfg({"dynamo_cfg": {"engine": "megatron"}})
    with pytest.raises(ValidationError, match="extra_forbidden"):
        megatron_dynamo_cfg(
            {"dynamo_cfg": {"decode_engine": {"context_parallel_size": 2}}}
        )


@pytest.mark.parametrize("size", [0, -2])
@pytest.mark.parametrize(
    "key", ["tensor_model_parallel_size", "pipeline_model_parallel_size"]
)
def test_engine_layout_overrides_must_be_positive(key, size) -> None:
    with pytest.raises(ValidationError, match="greater_than"):
        megatron_dynamo_cfg({"dynamo_cfg": {"prefill_engine": {key: size}}})


@pytest.mark.parametrize("backend", ["vllm", "sglang", "trtllm", "dynamo"])
def test_dynamo_switch_is_refused_for_non_megatron_backends(backend) -> None:
    with pytest.raises(ValueError, match=f"does not apply to backend='{backend}'"):
        resolve_generation_class({"backend": backend, "dynamo": True})


def test_dynamo_switch_resolves_to_the_sibling_class() -> None:
    assert resolve_generation_class(_megatron_config()) is MegatronDynamoGeneration
    assert issubclass(MegatronDynamoGeneration, GenerationInterface)
    assert not issubclass(MegatronDynamoGeneration, MegatronGeneration)
    plain = {**_megatron_config(), "dynamo": False}
    assert resolve_generation_class(plain) is MegatronGeneration


def test_engines_take_their_width_from_the_policy_layout() -> None:
    policy = _policy()
    assert engine_widths(policy) == {"prefill": 2, "decode": 2}
    # The same mcore_generation_config overrides the engine workers are built with.
    policy["generation"]["mcore_generation_config"]["tensor_model_parallel_size"] = 4
    widths = engine_widths(policy)
    assert widths == {"prefill": 4, "decode": 4}
    disaggregation = disaggregation_cfg(policy["generation"])
    assert replica_width(disaggregation, widths) == 12
    assert (
        engine_roles(disaggregation, 24, widths) == ["prefill", "decode", "decode"] * 2
    )
    with pytest.raises(ValueError, match="not a multiple of one replica's 12 GPUs"):
        engine_roles(disaggregation, 16, widths)


@pytest.mark.parametrize(
    ("prefill", "decode", "expected_widths", "roles_for_gpus"),
    [
        # 1 prefill at TP 4 next to 2 decode at TP 2: an 8-GPU replica.
        ({"tensor_model_parallel_size": 4}, {}, {"prefill": 4, "decode": 2}, 16),
        # PP counts toward a role's width like TP does.
        ({}, {"pipeline_model_parallel_size": 2}, {"prefill": 2, "decode": 4}, 10),
        # Both sides may move away from the shared layout.
        (
            {"tensor_model_parallel_size": 1},
            {"tensor_model_parallel_size": 4, "pipeline_model_parallel_size": 2},
            {"prefill": 1, "decode": 8},
            17,
        ),
    ],
    ids=["prefill_tp4", "decode_pp2", "both"],
)
def test_prefill_and_decode_may_differ_in_layout(
    prefill, decode, expected_widths, roles_for_gpus
) -> None:
    """Each role's width is the shared layout with its own overrides applied."""
    generation = _megatron_config(
        dynamo_cfg={"prefill_engine": prefill, "decode_engine": decode}
    )
    cfg = megatron_dynamo_cfg(generation)
    assert cfg.role_overrides("prefill") == prefill
    assert cfg.role_overrides("decode") == decode
    policy = _policy(generation)
    widths = engine_widths(policy)
    assert widths == expected_widths
    disaggregation = disaggregation_cfg(generation)
    replica = expected_widths["prefill"] + 2 * expected_widths["decode"]
    assert replica_width(disaggregation, widths) == replica
    assert engine_roles(disaggregation, roles_for_gpus, widths) == [
        "prefill",
        "decode",
        "decode",
    ] * (roles_for_gpus // replica)


@pytest.mark.parametrize(
    ("layout", "decode", "match"),
    [
        pytest.param(
            {"tensor_model_parallel_size": 4, "expert_model_parallel_size": 2},
            {},
            None,
            id="tp4_holds_ep2",
        ),
        pytest.param(
            {"tensor_model_parallel_size": 1, "expert_model_parallel_size": 4},
            {},
            "expert_model_parallel_size=4",
            id="ep_wider_than_the_engine",
        ),
        # Checked per role: a decode override that shrinks TP below the expert
        # grid fails naming the role, while the prefill side stays valid.
        pytest.param(
            {"tensor_model_parallel_size": 4, "expert_model_parallel_size": 2},
            {"tensor_model_parallel_size": 1},
            r"decode engines .*dynamo_cfg\.decode_engine",
            id="decode_override_too_narrow",
        ),
        # Expert TP counts toward the expert grid too, and PP multiplies both grids.
        pytest.param(
            {
                "tensor_model_parallel_size": 2,
                "expert_model_parallel_size": 2,
                "expert_tensor_parallel_size": 2,
                "pipeline_model_parallel_size": 2,
            },
            {},
            "not divisible",
            id="etp_and_pp",
        ),
    ],
)
def test_engine_width_must_hold_the_expert_parallel_grid(layout, decode, match) -> None:
    """MCore needs the engine's ranks divisible by ETP x EP x PP; refuse otherwise."""
    generation = _megatron_config(dynamo_cfg={"decode_engine": decode})
    generation["mcore_generation_config"].update(layout)
    policy = _policy(generation)

    if decode:
        assert engine_width(policy, "prefill") == 4
    if match is None:
        assert engine_widths(policy) == {"prefill": 4, "decode": 4}
    else:
        with pytest.raises(ValueError, match=match):
            engine_widths(policy)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(
            _set("disaggregation", "enabled", value=False),
            "requires policy.generation.disaggregation.enabled=true",
            id="needs_disaggregation",
        ),
        pytest.param(
            _set("disaggregation", "num_decode_engines", value=0),
            "at least one prefill and one decode",
            id="zero_decode",
        ),
        pytest.param(
            _set("disaggregation", "num_frontend_workers", value=2),
            "num_frontend_workers must be 1",
            id="one_frontend",
        ),
        pytest.param(
            _set("colocated", value={"enabled": True}),
            "colocated.enabled must be false",
            id="colocated",
        ),
        pytest.param(
            _set("refit_transport", value="nccl_reshard"),
            "refit_transport must be null",
            id="refit_transport",
        ),
        pytest.param(
            _set("quant_cfg", value={"algorithm": "nvfp4"}),
            "quant_cfg is not supported",
            id="quantization",
        ),
        pytest.param(
            _set("mcore_generation_config", "enable_prefix_caching", value=False),
            "enable_prefix_caching",
            id="prefix_caching",
        ),
        pytest.param(
            _set(
                "mcore_generation_config", "kv_cache_management_mode", value="recompute"
            ),
            "persist",
            id="kv_cache_mode",
        ),
        pytest.param(
            _set("mcore_generation_config", "expose_http_server", value=True),
            "expose_http_server",
            id="http_server",
        ),
    ],
)
def test_validation_rejects_unsupported_shapes(mutate, match) -> None:
    config = _megatron_config()
    mutate(config)
    with pytest.raises(ValueError, match=match):
        validate_megatron_dynamo_generation(config)


def test_validation_accepts_the_reference_shape_and_null_colocated() -> None:
    validate_megatron_dynamo_generation(_megatron_config())
    validate_megatron_dynamo_generation(_megatron_config(colocated=None))


def _master_config(generation: dict) -> SimpleNamespace:
    """The fields MegatronGeneration.validate_settings reads, at accepted values."""
    return SimpleNamespace(
        policy=_policy(generation),
        env={"should_use_nemo_gym": False},
        async_rl=SimpleNamespace(
            recompute_kv_cache_after_weight_updates=False,
            generation_fleet_health=SimpleNamespace(enabled=False),
        ),
    )


def test_megatron_generation_names_an_unservable_shape_first() -> None:
    generation = _megatron_config()
    generation["disaggregation"]["enabled"] = False
    with pytest.raises(ValueError, match="disaggregation.enabled=true"):
        resolve_generation_class(generation).validate_settings(
            _master_config(generation)
        )


@pytest.mark.parametrize("dynamo", [True, False], ids=["dynamo_on", "dynamo_off"])
def test_a_user_supplied_engine_role_is_refused_by_both_classes(dynamo) -> None:
    """The role is written per engine; a stray one in the user's config is refused.

    The check lives in MegatronGeneration.validate_settings, which both classes
    run, so the role is caught with Dynamo on and off; the derived per-engine
    configs that carry a role never pass through validate_settings.
    """
    generation = {**_megatron_config(), "dynamo": dynamo, "_dynamo_role": "prefill"}
    with pytest.raises(ValueError, match="set per engine by MegatronDynamoGeneration"):
        resolve_generation_class(generation).validate_settings(
            _master_config(generation)
        )


def test_megatron_generation_refuses_router_replay_behind_dynamo() -> None:
    """The KV handoff carries no routing-replay buffers, so the switch refuses router replay."""
    generation = _megatron_config()
    master_config = _master_config(generation)
    master_config.policy["router_replay"] = {"enabled": True}
    with pytest.raises(
        ValueError,
        match="router_replay.enabled must be false when policy.generation.dynamo=true",
    ):
        resolve_generation_class(generation).validate_settings(master_config)
