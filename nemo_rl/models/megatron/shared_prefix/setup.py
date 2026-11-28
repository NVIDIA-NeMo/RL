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
"""Worker setup check for shared-prefix packs.

A tree is one prompt with its completions. A pack is one forward that holds
several trees and runs each prompt once.
"""

from typing import Any

from megatron.bridge.models.hybrid.hybrid_provider import HybridModelProvider
from megatron.core.datasets.shared_prefix.inputs import (
    resolve_shared_prefix_padding_multiple,
)
from megatron.core.models.hybrid.shared_prefix import validate_shared_prefix_config

from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.shared_prefix import bound_by_full_length


def validate_shared_prefix_model(
    model_cfg: Any,
    *,
    policy_cfg: PolicyConfig,
    mtp_enabled: bool,
    model_prepares_inputs: bool,
) -> None:
    """Reject a Megatron model that cannot run shared-prefix packs.

    Megatron owns the rules on the model config and on the padding multiple.
    NeMo-RL owns the choice of Megatron Bridge provider, the rule on who prepares
    the model inputs and the two rules on multi-token prediction (MTP).

    Args:
        model_cfg: The Megatron Bridge provider that the worker's model is built from.
        policy_cfg: The worker's policy config.
        mtp_enabled: Whether the built model has MTP layers.
        model_prepares_inputs: Whether the built model packs its own sequences or
            slices its own inputs for context parallelism.

    Raises:
        NotImplementedError: If the provider is not a ``HybridModelProvider``, or if
            the model prepares its own inputs.
        ValueError: If ``policy.make_sequence_length_divisible_by`` does not fit the
            model's TP and CP sizes, or if a setting conflicts with the MTP layers.
    """
    if not isinstance(model_cfg, HybridModelProvider):
        raise NotImplementedError(
            "policy.sequence_packing.share_prefixes needs a model from Megatron "
            f"Bridge's HybridModelProvider, got {type(model_cfg).__name__}. "
            "Set share_prefixes to false for this model."
        )
    if model_prepares_inputs:
        raise NotImplementedError(
            "policy.sequence_packing.share_prefixes needs a model that takes the "
            "packed, context-parallel-sharded inputs that the worker builds. This "
            "model packs or slices its own inputs. Set share_prefixes to false for "
            "this model."
        )
    validate_shared_prefix_config(model_cfg)
    try:
        resolve_shared_prefix_padding_multiple(
            tp_size=model_cfg.tensor_model_parallel_size,
            cp_size=model_cfg.context_parallel_size,
            padding_multiple=policy_cfg["make_sequence_length_divisible_by"],
        )
    except ValueError as error:
        raise ValueError(
            "policy.make_sequence_length_divisible_by cannot pad shared-prefix "
            f"packs: {error}"
        ) from error
    if not mtp_enabled:
        return
    if not model_cfg.calculate_per_token_loss:
        raise ValueError(
            "The model has MTP layers, and the MTP loss of a shared-prefix pack "
            "needs calculate_per_token_loss=true on the model config."
        )
    if not bound_by_full_length(policy_cfg, budget_key="train_mb_tokens"):
        raise ValueError(
            "The model has MTP layers, and MTP processes every row of a training "
            "pack at its full length. The driver fits training packs to "
            "policy.sequence_packing.train_mb_tokens by packed length, which counts "
            "a shared prompt once. Remove policy.megatron_cfg.mtp_num_layers=0, or "
            "load a model without MTP layers."
        )
