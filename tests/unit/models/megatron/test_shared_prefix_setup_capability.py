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
"""Resolved-provider shared-prefix checks in ``nemo_rl.models.megatron.setup``."""

from __future__ import annotations

import importlib.util
from typing import Any
from unittest.mock import MagicMock

import pytest

# Probe the concrete module: test_modelopt_worker_utils.py leaves stub
# ``megatron.bridge`` packages in sys.modules that a package probe accepts.
pytest.importorskip("megatron.bridge.models.hybrid.hybrid_provider")
# The validator rejects shared mode before any provider check when the
# megatron.rl planner is missing, as in a Megatron-LM that predates it.
pytest.importorskip("megatron.rl.shared_prefix_execution")

from megatron.bridge.models.hybrid.hybrid_provider import (  # noqa: E402
    HybridModelProvider,
)

from nemo_rl.models.megatron import setup as megatron_setup  # noqa: E402

pytestmark = pytest.mark.mcore

TRAIN_CONFIG = {
    "megatron_cfg": {"enabled": True},
    "shared_prefix_training": {"mode": "train"},
}


@pytest.fixture(autouse=True)
def tp1_cp1_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        megatron_setup,
        "_get_mcore_shared_prefix_training_capability",
        lambda: frozenset(
            {
                megatron_setup.SUPPORTED_SHARED_PREFIX_TRAINING_CAPABILITY,
                megatron_setup.SUPPORTED_SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY,
                megatron_setup.SUPPORTED_SHARED_PREFIX_POSITIONLESS_ATTENTION_CAPABILITY,
            }
        ),
    )


def _provider(**overrides: Any) -> HybridModelProvider:
    provider = HybridModelProvider(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        attention_dropout=0.0,
        hidden_dropout=0.0,
    )
    for name, value in overrides.items():
        setattr(provider, name, value)
    return provider


def _moe_provider(**overrides: Any) -> HybridModelProvider:
    return _provider(
        **{
            "num_moe_experts": 8,
            "moe_router_load_balancing_type": "none",
            **overrides,
        }
    )


@pytest.mark.parametrize("window_size", [None, (-1, -1), [-1, -1]])
def test_full_attention_window_is_accepted_in_any_sequence_form(window_size):
    megatron_setup._validate_shared_prefix_model_capability(
        TRAIN_CONFIG, _provider(window_size=window_size)
    )


@pytest.mark.parametrize("window_size", [(128, 0), [128, 0]])
def test_sliding_window_is_rejected(window_size):
    with pytest.raises(NotImplementedError, match="sliding-window"):
        megatron_setup._validate_shared_prefix_model_capability(
            TRAIN_CONFIG, _provider(window_size=window_size)
        )


@pytest.mark.parametrize(
    ("load_balancing_type", "aux_loss_coeff"),
    [
        ("none", 1e-4),
        (["none", "none"], [0.0, 0.0]),
        (["none", "none"], [1e-4, 0.0]),
        ("aux_loss", 0.0),
        (["seq_aux_loss", "none"], [0.0, 1e-4]),
    ],
)
def test_aux_loss_coefficient_only_matters_for_aux_loss_types(
    load_balancing_type, aux_loss_coeff
):
    megatron_setup._validate_shared_prefix_model_capability(
        TRAIN_CONFIG,
        _moe_provider(
            moe_router_load_balancing_type=load_balancing_type,
            moe_aux_loss_coeff=aux_loss_coeff,
        ),
    )


@pytest.mark.parametrize(
    ("load_balancing_type", "aux_loss_coeff", "message"),
    [
        ("aux_loss", 1e-4, "moe_aux_loss_coeff=0"),
        (["none", "global_aux_loss"], [1e-4, 1e-3], "moe_aux_loss_coeff=0"),
        ("sinkhorn", 0.0, "'sinkhorn'"),
    ],
)
def test_active_router_balancing_is_rejected(
    load_balancing_type, aux_loss_coeff, message
):
    with pytest.raises(NotImplementedError, match=message):
        megatron_setup._validate_shared_prefix_model_capability(
            TRAIN_CONFIG,
            _moe_provider(
                moe_router_load_balancing_type=load_balancing_type,
                moe_aux_loss_coeff=aux_loss_coeff,
            ),
        )


def test_expert_rank_capacity_is_rejected():
    with pytest.raises(NotImplementedError, match="moe_expert_rank_capacity_factor"):
        megatron_setup._validate_shared_prefix_model_capability(
            TRAIN_CONFIG, _moe_provider(moe_expert_rank_capacity_factor=1.0)
        )


def test_missing_megatron_rl_planner_fails_at_setup(monkeypatch: pytest.MonkeyPatch):
    find_spec = importlib.util.find_spec

    def without_planner(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "megatron.rl.shared_prefix_execution":
            return None
        return find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", without_planner)
    with pytest.raises(
        NotImplementedError, match="megatron.rl.shared_prefix_execution"
    ):
        megatron_setup._validate_shared_prefix_model_capability(
            TRAIN_CONFIG, _provider()
        )


@pytest.mark.parametrize("field", ["moe_z_loss_coeff", "moe_input_jitter_eps"])
@pytest.mark.parametrize("value", [0.0, 1e-3])
def test_router_regularizers_must_resolve_to_null(field, value):
    with pytest.raises(NotImplementedError, match=f"model_overrides.{field}=null"):
        megatron_setup._validate_shared_prefix_model_capability(
            TRAIN_CONFIG, _moe_provider(**{field: value})
        )


def test_moe_config_does_not_copy_router_regularizers_in_shared_mode():
    model_cfg = MagicMock()
    model_cfg.moe_aux_loss_coeff = 1e-4
    model_cfg.moe_z_loss_coeff = None
    model_cfg.moe_input_jitter_eps = None
    config = {
        **TRAIN_CONFIG,
        "megatron_cfg": {
            "expert_tensor_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "moe_router_dtype": "float32",
            "moe_router_load_balancing_type": "none",
            "moe_router_bias_update_rate": 0.0,
            "moe_permute_fusion": True,
            "moe_enable_deepep": False,
            "moe_token_dispatcher_type": "alltoall",
            "moe_shared_expert_overlap": False,
            "moe_aux_loss_coeff": 0.0,
            "moe_z_loss_coeff": 0.0,
            "moe_input_jitter_eps": 0.0,
        },
    }

    megatron_setup._apply_moe_config(model_cfg, config)

    # The same YAML key means the same thing in every mode: model_overrides is
    # the one mechanism that sets these provider fields.
    assert model_cfg.moe_aux_loss_coeff == 1e-4
    assert model_cfg.moe_z_loss_coeff is None
    assert model_cfg.moe_input_jitter_eps is None
