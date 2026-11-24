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

from nemo_rl.models.generation import (
    configure_generation_config,
    resolve_generation_class,
)
from nemo_rl.models.generation.disagg_layout import plan_engines, resolve_disagg_layout
from nemo_rl.models.generation.dynamo.config import dynamo_enabled
from nemo_rl.models.generation.megatron import MegatronGeneration
from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg
from nemo_rl.models.generation.megatron.dynamo_config import (
    engine_width,
    engine_widths,
    reject_megatron_dynamo_outside_single_controller,
    role_overrides,
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


def _megatron_config(prefill_engine=None, decode_engine=None, **overrides) -> dict:
    """A ``backend: megatron`` generation config serving P/D engines behind Dynamo.

    ``prefill_engine`` / ``decode_engine`` go into ``mcore_generation_config``.
    """
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
    for key, value in (
        ("prefill_engine", prefill_engine),
        ("decode_engine", decode_engine),
    ):
        if value is not None:
            config["mcore_generation_config"][key] = value
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


def test_role_overrides_are_empty_by_default() -> None:
    config = _megatron_config()
    assert role_overrides(config, "prefill") == {}
    assert role_overrides(config, "decode") == {}


def test_role_layouts_never_reach_megatron_cfg() -> None:
    policy = _policy(
        _megatron_config(
            prefill_engine={"tensor_model_parallel_size": 4},
            decode_engine={"tensor_model_parallel_size": 1},
        )
    )
    merged = merged_inference_megatron_cfg(policy)  # type: ignore[arg-type]
    assert "prefill_engine" not in merged and "decode_engine" not in merged
    assert merged["tensor_model_parallel_size"] == 2


@pytest.mark.parametrize("backend", ["sglang", "trtllm", "dynamo"])
def test_dynamo_switch_is_refused_for_backends_without_a_dynamo_path(backend) -> None:
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    with pytest.raises(ValueError, match=f"does not apply to backend='{backend}'"):
        configure_generation_config({"backend": backend, "dynamo": True}, tokenizer)


def test_dynamo_switch_resolves_to_megatron_generation() -> None:
    assert resolve_generation_class(_megatron_config()) is MegatronGeneration


def test_dynamo_switch_passes_generation_config_validation_for_megatron() -> None:
    """The shared ``dynamo`` switch also serves vLLM; its validation must not
    refuse the Megatron backend."""
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)
    config = configure_generation_config(
        _megatron_config(stop_token_ids=None), tokenizer
    )
    assert config["stop_token_ids"] == [1]


def test_engines_take_their_width_from_the_policy_layout() -> None:
    policy = _policy()
    assert engine_widths(policy) == {"prefill": 2, "decode": 2}
    # The same mcore_generation_config overrides the engine workers are built with.
    policy["generation"]["mcore_generation_config"]["tensor_model_parallel_size"] = 4
    widths = engine_widths(policy)
    assert widths == {"prefill": 4, "decode": 4}
    layout = resolve_disagg_layout(policy["generation"])
    roles, engine_gpus, replicas = plan_engines(
        24,
        layout=layout,
        aggregated_width=4,
        prefill_width=4,
        decode_width=4,
        backend_cfg_key="mcore_generation_config",
    )
    assert roles == ["prefill", "decode", "decode"] * 2
    assert engine_gpus == [4] * 6 and replicas == 2
    with pytest.raises(ValueError, match="replica width 12 GPUs"):
        plan_engines(
            16,
            layout=layout,
            aggregated_width=4,
            prefill_width=4,
            decode_width=4,
            backend_cfg_key="mcore_generation_config",
        )


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
    generation = _megatron_config(prefill_engine=prefill, decode_engine=decode)
    assert role_overrides(generation, "prefill") == prefill
    assert role_overrides(generation, "decode") == decode
    policy = _policy(generation)
    widths = engine_widths(policy)
    assert widths == expected_widths
    replica = expected_widths["prefill"] + 2 * expected_widths["decode"]
    roles, _, _ = plan_engines(
        roles_for_gpus,
        layout=resolve_disagg_layout(generation),
        aggregated_width=1,
        prefill_width=widths["prefill"],
        decode_width=widths["decode"],
        backend_cfg_key="mcore_generation_config",
    )
    assert roles == ["prefill", "decode", "decode"] * (roles_for_gpus // replica)


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
            r"decode engines .*mcore_generation_config\.decode_engine",
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
    generation = _megatron_config(decode_engine=decode)
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
        pytest.param(
            _set("_dynamo_role", value="prefill"),
            "set per engine by the Dynamo engine fleet",
            id="engine_role",
        ),
        pytest.param(
            _set("disaggregation", "num_prefill_engines", value=0),
            "at least one prefill and one decode",
            id="zero_prefill",
        ),
        pytest.param(
            _set("disaggregation", "frontend_tokenize", value=True),
            "frontend_tokenize false",
            id="frontend_tokenize",
        ),
        pytest.param(
            _set("real_quant", value=True),
            "real_quant is not supported",
            id="real_quant",
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
        async_rl=SimpleNamespace(
            recompute_kv_cache_after_weight_updates=False,
            generation_fleet_health=SimpleNamespace(enabled=False),
        ),
    )


def test_megatron_generation_refuses_the_switch_until_the_fleet_exists() -> None:
    with pytest.raises(NotImplementedError, match="not served by MegatronGeneration"):
        MegatronGeneration.validate_settings(_master_config(_megatron_config()))


def test_megatron_generation_names_an_unservable_shape_first() -> None:
    generation = _megatron_config()
    generation["disaggregation"]["enabled"] = False
    with pytest.raises(ValueError, match="disaggregation.enabled=true"):
        MegatronGeneration.validate_settings(_master_config(generation))


def test_megatron_generation_refuses_disaggregation_without_dynamo() -> None:
    """Only the Dynamo fleet reads the block; without it the run would serve
    plain aggregated engines."""
    generation = _megatron_config(dynamo=False)
    with pytest.raises(ValueError, match="requires policy.generation.dynamo=true"):
        MegatronGeneration.validate_settings(_master_config(generation))


@pytest.mark.parametrize("role_key", ["prefill_engine", "decode_engine"])
def test_megatron_generation_refuses_role_layouts_without_dynamo(role_key) -> None:
    """Only engines behind Dynamo read the per-role layouts; without it they
    would be dropped and the run would use the shared layout."""
    generation = _megatron_config(
        dynamo=False, **{role_key: {"tensor_model_parallel_size": 8}}
    )
    generation["disaggregation"]["enabled"] = False
    with pytest.raises(ValueError, match="ignored without it"):
        MegatronGeneration.validate_settings(_master_config(generation))


@pytest.mark.parametrize("role_key", ["prefill_engine", "decode_engine"])
def test_legacy_entrypoints_refuse_role_layouts_without_dynamo(role_key) -> None:
    """Legacy entrypoints never run ``validate_settings``; without this check the
    per-role layouts would be dropped and the run would use the shared layout."""
    generation = _megatron_config(
        dynamo=False, **{role_key: {"tensor_model_parallel_size": 8}}
    )
    with pytest.raises(ValueError, match="ignored without it"):
        reject_megatron_dynamo_outside_single_controller(generation)


def test_dynamo_cfg_is_the_shared_dynamo_schema() -> None:
    """Megatron reads the same ``dynamo_cfg`` block as vLLM, Gym fields included."""
    validate_megatron_dynamo_generation(
        _megatron_config(
            dynamo_cfg={
                "request_timeout_s": 30.0,
                "frontend_args": {
                    "session_affinity_ttl_secs": 600,
                    "render_workers": 2,
                },
            }
        )
    )
    with pytest.raises(ValueError, match="extra_forbidden|Extra inputs"):
        validate_megatron_dynamo_generation(_megatron_config(dynamo_cfg={"bogus": 1}))


def test_nvlink_domain_span_is_the_widest_engine() -> None:
    """The router pairs prefill and decode engines across the whole fleet, so
    only one engine needs NVLink, not a whole replica."""
    policy = _policy(
        _megatron_config(
            prefill_engine={"tensor_model_parallel_size": 4},
            decode_engine={"tensor_model_parallel_size": 1},
        )
    )
    assert MegatronGeneration.nvlink_domain_span(policy) == 4  # type: ignore[arg-type]
    # Decode wider than prefill: the span must follow the wider role.
    wide_decode = _policy(
        _megatron_config(
            prefill_engine={"tensor_model_parallel_size": 1},
            decode_engine={"tensor_model_parallel_size": 4},
        )
    )
    assert MegatronGeneration.nvlink_domain_span(wide_decode) == 4  # type: ignore[arg-type]
