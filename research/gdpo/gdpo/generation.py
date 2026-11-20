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
"""Automodel generation adapter for masked diffusion language models."""

from typing import TYPE_CHECKING, Any

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationConfig,
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

if TYPE_CHECKING:
    from nemo_rl.models.policy.lm_policy import Policy


class _NoOpWeightSynchronizer(WeightSynchronizer):
    """Weight-sync adapter for generation that shares policy tensors."""

    @property
    def is_stale(self) -> bool:
        """Reports current weights because generation reads policy tensors."""
        return False

    def init_communicator(self) -> None:
        """Skips communicator setup because there is no weight transfer."""
        pass

    def sync_weights(
        self,
        *,
        timer: Timer | None = None,
        kv_scales: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Return immediately because there is no second weight copy."""
        return {}

    def shutdown(self) -> None:
        """Skips teardown because no communication resources are owned."""
        pass


class AutomodelGeneration(GenerationInterface):
    """Generation interface that denoises on the training policy's own weights.

    Masked diffusion models decode by iteratively unmasking a fixed-width canvas
    rather than by appending tokens. SGLang serves some of them (LLaDA2.0 and
    SDAR, via ``--dllm-algorithm``), but not LLaDA-8B, and vLLM and TRT-LLM
    serve none of them. Rollouts here instead run in the training
    workers, which makes this backend colocated by construction: there are no
    separate inference weights, so there is nothing to refit and no collective
    to initialize.
    """

    def __init__(self, config: PolicyConfig, policy: "Policy"):
        """Initializes the backend around an existing training policy.

        Args:
            config: The full policy configuration. Its ``generation`` block
                supplies sampling and denoising parameters.
            policy: The training ``Policy`` whose workers will denoise.
        """
        generation_config = config["generation"]
        assert generation_config is not None, (
            "policy.generation must be configured to use the automodel backend."
        )
        self._policy_config = config
        self.cfg: GenerationConfig = generation_config
        self._policy = policy
        self.weight_synchronizer = _NoOpWeightSynchronizer()

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        """Returns no work: generation shares the training weights, so no refit."""
        return []

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Denoises a batch of prompts across the policy's data-parallel ranks.

        Args:
            data: Right-padded prompts with ``input_ids`` and ``input_lengths``.
            greedy: Whether to decode greedily, overriding the sampling config.

        Returns:
            A BatchedDataDict conforming to ``GenerationOutputSpec``.
        """
        # Sequence packing and dynamic batching are rejected for dLLM policies
        # (see validate_gdpo_config), so a plain data-parallel split is enough.
        sharded_data = data.shard_by_batch_size(
            self._policy.data_parallel_size, batch_size=None
        )
        seed = torch.randint(0, 2**31 - 1, ()).item()
        futures = self._policy.worker_group.run_all_workers_sharded_data(
            "generate",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["context_parallel", "tensor_parallel"],
            output_is_replicated=["context_parallel", "tensor_parallel"],
            common_kwargs={"greedy": greedy, "seed": seed},
        )
        return BatchedDataDict.from_batches(
            self._policy.worker_group.get_all_worker_results(futures)
        )

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Returns success: the training weights are already resident."""
        return True

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Returns success: there is no inference engine to tear down."""
        return True

    def blocks_training(self) -> bool:
        """Reports that generation must stand down before a training step.

        Denoising runs in the training workers themselves, so it always holds
        the GPUs training needs. This backend is colocated by construction --
        validate_gdpo_config rejects anything else -- so unlike
        MegatronGeneration there is no non-colocated case to distinguish.
        """
        return True

    def wake_carries_weight_updates(self) -> bool:
        """Reports that waking alone serves the latest weights.

        There is no separate copy to refresh: rollouts read the training
        tensors outright, which is the second case the interface describes.
        """
        return True

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Accepts the cross-backend refit-prep contract; dLLM needs none of it."""
        pass

    def shutdown(self) -> bool:
        """Returns success without tearing anything down.

        This backend never owns the policy it denoises with, so shutting it
        down here would kill workers the trainer still holds. The caller
        shuts the training policy down separately, the same way
        ``MegatronGeneration`` behaves when colocated.
        """
        return True
