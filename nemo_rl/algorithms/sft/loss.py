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
"""SFT loss functions.

This module provides the loss function for Supervised Fine-Tuning,
which is a wrapper around NLLLoss with SFT-specific defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nemo_rl.algorithms.loss_functions import NLLLoss

if TYPE_CHECKING:
    from nemo_rl.algorithms.interfaces import LossFunction


class SFTLoss(NLLLoss):
    """Supervised Fine-Tuning Loss.

    This is a wrapper around NLLLoss that provides SFT-specific naming
    and can be extended with SFT-specific functionality in the future.

    Example:
        >>> loss_fn = SFTLoss()
        >>> loss, metrics = loss_fn(logits, data, global_valid_seqs, global_valid_toks)
    """

    pass


def create_sft_loss_function() -> "LossFunction":
    """Create the default SFT loss function.

    Returns:
        NLLLoss instance configured for SFT training.

    Example:
        >>> loss_fn = create_sft_loss_function()
        >>> loss, metrics = loss_fn(logits, data, global_valid_seqs, global_valid_toks)
    """
    return NLLLoss()
