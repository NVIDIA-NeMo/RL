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
from enum import Enum
from typing import Any, Optional, TypedDict

import torch

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.utils.timer import Timer


class OffloadMode(Enum):
    """Controls how aggressively to offload during finish_training().

    EVAL_ONLY: keep model on GPU in eval mode, optionally offload optimizer
        based on worker config (offload_optimizer_for_logprob). Used by
        algorithm code before logprob / KV-scale inference.
    OPTIMIZER_ONLY: offload optimizer, keep model on GPU in its current mode.
        Used by colocated weight synchronizers before weight transfer (model
        must stay on GPU for CUDA IPC / HTTP streaming).
    FULL: offload everything appropriate for the deployment topology. In
        colocated mode this offloads model + optimizer; in non-colocated mode
        this sets eval mode, keeps model on GPU, and optionally offloads
        optimizer.
    """

    EVAL_ONLY = "eval_only"
    OPTIMIZER_ONLY = "optimizer_only"
    FULL = "full"


class LogprobOutputSpec(TypedDict):
    """logprobs: Tensor of log probabilities."""

    logprobs: torch.Tensor


class ReferenceLogprobOutputSpec(TypedDict):
    """logprobs: Tensor of log probabilities."""

    reference_logprobs: torch.Tensor


class ScoreOutputSpec(TypedDict):
    """scores: Tensor of scores."""

    scores: torch.Tensor


class TopkLogitsOutputSpec(TypedDict):
    """Per-position top-k logits and corresponding global token indices."""

    topk_logits: torch.Tensor
    topk_indices: torch.Tensor


class PolicyTrainerInterface(ABC):
    """Abstract base class defining the interface for RL policy training."""

    @abstractmethod
    def get_logprobs(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        timer: Optional[Timer] = None,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get logprobs of actions from observations.

        Args:
            data: BatchedDataDict containing rollouts (tokens)

        Returns:
            BatchedDataDict containing:
                - ``logprobs``: Tensor of logprobs of actions
        """
        pass

    @abstractmethod
    def get_reference_policy_logprobs(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get logprobs of actions from observations.

        Args:
            data: BatchedDataDict containing rollouts (tokens)

        Returns:
            BatchedDataDict containing:
                - ``logprobs``: Tensor of logprobs of actions
        """
        pass

    @abstractmethod
    def get_topk_logits(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
        timer: Optional[Timer] = None,
    ) -> BatchedDataDict[TopkLogitsOutputSpec]:
        """Get per-position top-k logits and global indices for a batch of inputs.

        Notes:
            - Aligns to next-token positions → returns S-1 positions.
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
    def calibrate_qkv_fp8_scales(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """Calibrate FP8 scales for Q/K/V activations used by KV cache.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths.
            micro_batch_size: Optional override for micro batch size during calibration.
            percentile: Percentile for per-tensor amax estimation.
            margin: Safety margin multiplier applied to amax.
            include_q: Whether to also compute scale for Q in addition to K/V.

        Returns:
            Dict with overall configuration and per-layer scales.
        """
        pass

    @abstractmethod
    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        """Transition to training phase. Load model and optimizer to GPU, set train mode."""
        pass

    @abstractmethod
    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        """Transition out of training phase. Free GPU resources.

        Accepts an optional ``offload_mode`` kwarg (OffloadMode enum):
          - EVAL_ONLY: model on GPU in eval mode (for logprob inference).
          - OPTIMIZER_ONLY: offload optimizer only (for weight staging).
          - FULL (default): full offload based on deployment topology.
        """
        pass

    @abstractmethod
    def save_checkpoint(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        pass


# Backward compatibility
PolicyInterface = PolicyTrainerInterface
