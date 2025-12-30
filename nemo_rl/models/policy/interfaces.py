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


class TopkLogitsOutputSpec(TypedDict):
    """Per-position top-k logits and corresponding global token indices."""

    topk_logits: torch.Tensor
    topk_indices: torch.Tensor


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
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
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
    def get_topk_logits(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
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
        self, ip: str, port: int, world_size: int, *, train_world_size: int
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
    def stream_weights_via_ipc_zmq(
        self, *args: Any, **kwargs: Any
    ) -> list[ray.ObjectRef]:
        pass

    def stream_weights_via_http(
        self, sglang_url_to_gpu_uuids: dict[str, list[str]]
    ) -> list[ray.ObjectRef]:
        """Stream model weights to SGLang servers via HTTP API.

        Args:
            sglang_url_to_gpu_uuids: Dict mapping SGLang server URL to list of GPU UUIDs it uses
        """
        raise NotImplementedError(
            "stream_weights_via_http is not implemented for this policy worker"
        )

    @abstractmethod
    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> list[ray.ObjectRef]:
        pass

    @abstractmethod
    def prepare_for_lp_inference(self) -> None:
        pass
