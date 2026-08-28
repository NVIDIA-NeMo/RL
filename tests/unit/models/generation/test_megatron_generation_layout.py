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

"""CPU tests for colocated megatron generation layout selection.

`dedicated_inference_megatron_cfg` decides whether colocated generation runs
directly on the shared training model or whether the worker builds a second,
resharded inference model. Getting a "no" wrong is silent: generation still
produces tokens, so the GPU functional tests keep passing while the refit path
they were written to exercise never executes. That failure mode is exactly what
happened for GTP -- the layout comparison did not look at the weight-shard
counts, so a TP1 x GTP2 training model looked identical to a TP1 inference
model. These tests pin the decision without a GPU.
"""

from copy import deepcopy
from typing import Any, cast

import pytest

from nemo_rl.models.generation.megatron.config import (
    dedicated_inference_megatron_cfg,
    merged_inference_megatron_cfg,
)
from nemo_rl.models.policy import PolicyConfig


def _policy_config(
    *,
    megatron_overrides: dict[str, Any] | None = None,
    generation_overrides: dict[str, Any] | None = None,
) -> PolicyConfig:
    """A minimal policy config carrying only what the layout selector reads."""
    megatron_cfg: dict[str, Any] = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "context_parallel_size": 1,
        "tensor_parallel_num_weight_shards": None,
        "expert_tensor_parallel_num_weight_shards": None,
        "sequence_parallel": False,
        "transformer_impl": "transformer_engine",
        "activation_checkpointing": True,
    }
    megatron_cfg.update(megatron_overrides or {})
    return cast(
        PolicyConfig,
        {
            "megatron_cfg": megatron_cfg,
            "generation": {
                "backend": "megatron",
                "mcore_generation_config": deepcopy(generation_overrides or {}),
            },
        },
    )


def test_matched_layout_is_reshardless():
    """Identical train/inference layout and impl => generate on the shared model."""
    assert dedicated_inference_megatron_cfg(_policy_config()) is None


@pytest.mark.parametrize(
    "shards_key, tp_key",
    [
        ("tensor_parallel_num_weight_shards", "tensor_model_parallel_size"),
        (
            "expert_tensor_parallel_num_weight_shards",
            "expert_tensor_parallel_size",
        ),
    ],
)
def test_weight_shard_count_equal_to_tp_is_not_a_layout_change(shards_key, tp_key):
    """An explicit TP-equal shard count means "GTP off", same as null.

    `null` and an explicit value equal to the TP degree describe the same
    unsharded layout. If only one of them compared equal, users would get a
    pointless dedicated inference model (and a full refit every wake) purely
    from writing the default out longhand.
    """
    config = _policy_config(megatron_overrides={tp_key: 2, shards_key: 2})
    assert dedicated_inference_megatron_cfg(config) is None


@pytest.mark.parametrize(
    "shards_key, tp_key, remat_key",
    [
        (
            "tensor_parallel_num_weight_shards",
            "tensor_model_parallel_size",
            "tensor_parallel_num_weight_shards",
        ),
        (
            "expert_tensor_parallel_num_weight_shards",
            "expert_tensor_parallel_size",
            "expert_tensor_parallel_num_weight_shards",
        ),
    ],
)
def test_gtp_training_model_forces_a_dedicated_inference_model(
    shards_key, tp_key, remat_key
):
    """GTP on the training side must take the reshard path.

    Inference never uses GTP: refit reassembles the training model's dim-0
    weight shards into whole inference weights. So a GTP-sharded training model
    is by construction a different layout, and the resolved inference config
    must pin the shard count back down to its own TP degree.
    """
    config = _policy_config(megatron_overrides={shards_key: 2})
    inference_mcfg = dedicated_inference_megatron_cfg(config)
    assert inference_mcfg is not None
    assert inference_mcfg[remat_key] == inference_mcfg[tp_key]


def test_context_parallel_training_forces_a_dedicated_inference_model():
    """Inference pins CP=1, so any CP>1 training layout differs."""
    config = _policy_config(megatron_overrides={"context_parallel_size": 2})
    inference_mcfg = dedicated_inference_megatron_cfg(config)
    assert inference_mcfg is not None
    assert inference_mcfg["context_parallel_size"] == 1


def test_differing_transformer_impl_forces_a_dedicated_inference_model():
    """Same layout but a different impl still needs a second model."""
    config = _policy_config(
        generation_overrides={"transformer_impl": "inference_optimized"}
    )
    assert dedicated_inference_megatron_cfg(config) is not None


def test_merged_cfg_rejects_inference_optimized_without_sequence_parallel():
    """inference_optimized layers hard-require SP with TP>1.

    The colocated build bypasses validate_and_set_config, so this merge is the
    only place the user gets a named config key instead of a raw MCore assert.
    """
    config = _policy_config(
        megatron_overrides={"tensor_model_parallel_size": 2},
        generation_overrides={"transformer_impl": "inference_optimized"},
    )
    with pytest.raises(ValueError, match="sequence_parallel"):
        merged_inference_megatron_cfg(config)


def test_merged_cfg_disables_activation_checkpointing():
    """Inference never trains, so the training recompute setting must not leak."""
    merged = merged_inference_megatron_cfg(_policy_config())
    assert merged["activation_checkpointing"] is False
