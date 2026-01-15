# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""DPO loss functions.

This module provides the loss function for Direct Preference Optimization,
which combines preference loss with optional SFT regularization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nemo_rl.algorithms.loss_functions import DPOLossFn

if TYPE_CHECKING:
    from nemo_rl.algorithms.dpo.config import DPOConfig
    from nemo_rl.algorithms.interfaces import LossFunction


class DPOLoss(DPOLossFn):
    """Direct Preference Optimization Loss.

    Wrapper around DPOLossFn for consistent naming and future extensibility.

    Example:
        >>> config = {"reference_policy_kl_penalty": 0.1, ...}
        >>> loss_fn = DPOLoss(config)
        >>> loss, metrics = loss_fn(logits, data, global_valid_seqs, global_valid_toks)
    """

    pass


def create_dpo_loss_function(config: "DPOConfig") -> "LossFunction":
    """Create a DPO loss function from configuration.

    Args:
        config: DPO configuration with loss parameters.

    Returns:
        DPOLossFn instance configured for DPO training.

    Example:
        >>> loss_fn = create_dpo_loss_function(dpo_config)
    """
    loss_config = {
        "reference_policy_kl_penalty": config.get("reference_policy_kl_penalty", 0.1),
        "preference_loss_weight": config.get("preference_loss_weight", 1.0),
        "sft_loss_weight": config.get("sft_loss_weight", 0.0),
        "preference_average_log_probs": config.get("preference_average_log_probs", False),
        "sft_average_log_probs": config.get("sft_average_log_probs", False),
    }
    return DPOLossFn(loss_config)
