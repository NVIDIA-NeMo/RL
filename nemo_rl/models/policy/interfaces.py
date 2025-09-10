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

import ray
import torch

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec


class LogprobOutputSpec(TypedDict):
    """logprobs: Tensor of log probabilities."""

    logprobs: torch.Tensor


class ReferenceLogprobOutputSpec(TypedDict):
    """logprobs: Tensor of log probabilities."""

    reference_logprobs: torch.Tensor


class ScoreOutputSpec(TypedDict):
    """scores: Tensor of scores."""

    scores: torch.Tensor


class PolicyInterface(ABC):
    """Abstract base class defining the interface for RL policies."""

    @abstractmethod
    def get_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get logprobs of actions from observations.

        Args:
            data: BatchedDataDict containing rollouts (tokens)

        Returns:
            BatchedDataDict containing:
                - logprobs: Tensor of logprobs of actions
        """
        pass

    @abstractmethod
    def get_reference_policy_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get logprobs of actions from observations.

        Args:
            data: BatchedDataDict containing rollouts (tokens)

        Returns:
            BatchedDataDict containing:
                - logprobs: Tensor of logprobs of actions
        """
        pass

    @abstractmethod
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a global batch of data.

        Args:
            data: BatchedDataDict containing rollouts (tokens)
            loss_fn: Loss function to use for training
            eval_mode: Whether to run in evaluation mode (no gradient updates)
            gbs: Global batch size override (if None, uses config default)
            mbs: Micro batch size override (if None, uses config default)
        """
        pass

    @abstractmethod
    def score(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict[ScoreOutputSpec]:
        """Score a batch of data using the policy."""
        pass

    @abstractmethod
    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        pass


class ColocatablePolicyInterface(PolicyInterface):
    @abstractmethod
    def init_collective(
        self, ip: str, port: int, world_size: int
    ) -> list[ray.ObjectRef]:
        pass

    @abstractmethod
    def offload_before_refit(self) -> None:
        pass

    @abstractmethod
    def offload_after_refit(self) -> None:
        pass

    @abstractmethod
    def prepare_refit_info(self) -> Optional[dict[str, Any]]:
        pass

    @abstractmethod
    def prepare_weights_for_ipc(self, *args: Any, **kwargs: Any) -> list[list[str]]:
        pass

    @abstractmethod
    def get_weights_ipc_handles(self, keys: list[str]) -> dict[str, Any]:
        pass

    @abstractmethod
    def broadcast_weights_for_collective(self) -> list[ray.ObjectRef]:
        pass
