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
"""Tests for the Megatron worker's setup check for shared-prefix packs."""

from types import SimpleNamespace
from typing import Any

import pytest

# Probe the concrete module: test_modelopt_worker_utils.py leaves stub
# ``megatron.bridge`` packages in sys.modules that a package probe accepts.
pytest.importorskip("megatron.bridge.models.hybrid.hybrid_provider")

from megatron.bridge.models.hybrid.hybrid_provider import (  # noqa: E402
    HybridModelProvider,
)

from nemo_rl.models.megatron.shared_prefix.setup import (  # noqa: E402
    validate_shared_prefix_model,
)

pytestmark = pytest.mark.mcore


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


def _policy_cfg(*, padding_multiple: int = 1, **megatron_cfg: Any) -> dict[str, Any]:
    return {
        "make_sequence_length_divisible_by": padding_multiple,
        "megatron_cfg": {"enabled": True, **megatron_cfg},
    }


def _validate(
    provider: Any,
    *,
    policy_cfg: dict[str, Any],
    mtp_enabled: bool = False,
    model_prepares_inputs: bool = False,
) -> None:
    validate_shared_prefix_model(
        provider,
        policy_cfg=policy_cfg,
        mtp_enabled=mtp_enabled,
        model_prepares_inputs=model_prepares_inputs,
    )


def test_hybrid_model_without_mtp_is_accepted() -> None:
    # Without MTP layers neither MTP rule applies.
    _validate(
        _provider(calculate_per_token_loss=False),
        policy_cfg=_policy_cfg(mtp_num_layers=0),
    )


def test_other_model_providers_are_rejected() -> None:
    with pytest.raises(NotImplementedError, match="HybridModelProvider"):
        _validate(SimpleNamespace(), policy_cfg=_policy_cfg())


def test_model_that_prepares_its_own_inputs_is_rejected() -> None:
    with pytest.raises(NotImplementedError, match="packs or slices its own inputs"):
        _validate(_provider(), policy_cfg=_policy_cfg(), model_prepares_inputs=True)


def test_megatron_config_check_rejects_cuda_graphs() -> None:
    with pytest.raises(NotImplementedError, match="CUDA graphs"):
        _validate(_provider(cuda_graph_impl="local"), policy_cfg=_policy_cfg())


@pytest.mark.parametrize(
    ("tp_size", "cp_size", "padding_multiple"),
    [(1, 1, 1), (1, 2, 4), (2, 1, 4), (2, 2, 16)],
)
def test_padding_multiple_that_fits_the_topology_is_accepted(
    tp_size: int, cp_size: int, padding_multiple: int
) -> None:
    _validate(
        _provider(
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            sequence_parallel=tp_size > 1,
        ),
        policy_cfg=_policy_cfg(padding_multiple=padding_multiple),
    )


@pytest.mark.parametrize(
    ("tp_size", "cp_size", "padding_multiple"),
    [(1, 2, 2), (2, 1, 2), (2, 2, 4)],
)
def test_padding_multiple_that_does_not_fit_the_topology_is_rejected(
    tp_size: int, cp_size: int, padding_multiple: int
) -> None:
    with pytest.raises(ValueError, match="policy.make_sequence_length_divisible_by"):
        _validate(
            _provider(
                tensor_model_parallel_size=tp_size,
                context_parallel_size=cp_size,
                sequence_parallel=tp_size > 1,
            ),
            policy_cfg=_policy_cfg(padding_multiple=padding_multiple),
        )


def test_mtp_model_planned_by_full_length_is_accepted() -> None:
    # No mtp_num_layers key: the driver fits training packs at full length.
    _validate(
        _provider(calculate_per_token_loss=True),
        policy_cfg=_policy_cfg(),
        mtp_enabled=True,
    )


def test_mtp_model_needs_the_per_token_loss() -> None:
    with pytest.raises(ValueError, match="calculate_per_token_loss"):
        _validate(
            _provider(calculate_per_token_loss=False),
            policy_cfg=_policy_cfg(),
            mtp_enabled=True,
        )


def test_mtp_model_planned_by_packed_length_is_rejected() -> None:
    # mtp_num_layers=0 makes the driver fit training packs at packed length.
    with pytest.raises(ValueError, match="mtp_num_layers"):
        _validate(
            _provider(calculate_per_token_loss=True),
            policy_cfg=_policy_cfg(mtp_num_layers=0),
            mtp_enabled=True,
        )
