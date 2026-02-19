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
from abc import ABC, abstractmethod
from typing import Any, Optional, TypedDict

import torch

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.utils.timer import Timer


class ValueOutputSpec(TypedDict):
    """values: Tensor of value predictions [batch_size, sequence_length]."""

    values: torch.Tensor


class ValueInterface(ABC):
    """Abstract base class defining the interface for value functions."""

    @abstractmethod
    def get_values(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        timer: Optional[Timer] = None,
    ) -> BatchedDataDict[ValueOutputSpec]:
        """Get value predictions for observations.

        Args:
            data: BatchedDataDict containing input sequences (tokens)
            timer: Optional timer for profiling

        Returns:
            BatchedDataDict containing:
                - values: Tensor of value predictions [batch_size, sequence_length]
        """
        pass

    @abstractmethod
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        *,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> dict[str, Any]:
        """Train the value function on a global batch of data.

        Args:
            data: BatchedDataDict containing training data
            loss_fn: Loss function to use for training
            eval_mode: Whether to run in evaluation mode (no gradient updates)
            gbs: Global batch size override (if None, uses config default)
            mbs: Micro batch size override (if None, uses config default)
            timer: Optional timer for profiling

        Returns:
            Dictionary containing training metrics (loss, grad_norm, etc.)
        """
        pass

    @abstractmethod
    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        """Prepare the value model for training (e.g., load to GPU)."""
        pass

    @abstractmethod
    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        """Clean up after training."""
        pass

    @abstractmethod
    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        """Save model checkpoint."""
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown workers and clean up resources."""
        pass
