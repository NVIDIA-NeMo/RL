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
"""Generation-focused policy implementation.

This module provides the GenerationPolicy class that handles all generation-related
responsibilities:
- Text generation (generate())
- Scoring (score())
- Weight updates from training policy (update_weights())

Example:
    >>> from nemo_rl.models.policy import GenerationPolicy
    >>> generation_policy = GenerationPolicy(cluster, config, tokenizer)
    >>> outputs = generation_policy.generate(prompts)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    import ray

    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.named_sharding import NamedSharding
    from nemo_rl.distributed.worker_groups import RayWorkerGroup
    from nemo_rl.models.generation.interfaces import (
        GenerationDatumSpec,
        GenerationOutputSpec,
    )
    from nemo_rl.models.policy import PolicyConfig
    from nemo_rl.models.policy.interfaces import ScoreOutputSpec


class GenerationPolicyProtocol(Protocol):
    """Protocol defining the generation policy interface.

    This protocol specifies the methods required for any generation policy
    implementation. Use this for type hints when accepting any generation
    policy implementation.
    """

    def generate(
        self,
        data: "BatchedDataDict[GenerationDatumSpec]",
        greedy: bool = False,
    ) -> "BatchedDataDict[GenerationOutputSpec]":
        """Generate responses from prompts."""
        ...

    def update_weights(self, state_dict: dict[str, Any]) -> None:
        """Update model weights from a state dict."""
        ...


class GenerationPolicy:
    """Policy class focused on generation responsibilities.

    GenerationPolicy handles all generation-related operations:
    - Text generation from prompts
    - Response scoring
    - Weight updates from training

    This class is designed to be used in conjunction with TrainingPolicy
    for complete policy functionality. For backward compatibility, the
    original Policy class combines both interfaces.

    Attributes:
        worker_group: Ray worker group for distributed execution.
        sharding_annotations: Named sharding for distributed data.
        cfg: Policy configuration.

    Example:
        >>> generation_policy = GenerationPolicy(worker_group, sharding, config)
        >>> # Generate responses
        >>> outputs = generation_policy.generate(prompts, greedy=False)
        >>> # Score responses
        >>> scores = generation_policy.score(data)
    """

    def __init__(
        self,
        worker_group: "RayWorkerGroup",
        sharding_annotations: "NamedSharding",
        cfg: "PolicyConfig",
    ):
        """Initialize GenerationPolicy.

        Args:
            worker_group: Ray worker group for distributed execution.
            sharding_annotations: Named sharding for distributed data.
            cfg: Policy configuration dictionary.
        """
        self.worker_group = worker_group
        self.sharding_annotations = sharding_annotations
        self.cfg = cfg

    def generate(
        self,
        data: "BatchedDataDict[GenerationDatumSpec]",
        greedy: bool = False,
    ) -> "BatchedDataDict[GenerationOutputSpec]":
        """Generate responses from input prompts.

        Args:
            data: BatchedDataDict containing:
                - input_ids: Tokenized prompt sequences
                - input_lengths: Length of each prompt
            greedy: If True, uses greedy decoding. Otherwise samples.

        Returns:
            BatchedDataDict containing:
                - output_ids: Generated token sequences
                - generation_lengths: Length of each generation
                - unpadded_sequence_lengths: Total sequence lengths
                - logprobs: Log probabilities of generated tokens

        Raises:
            AssertionError: If data is not a BatchedDataDict or missing required fields.
            ValueError: If generation config is not set or output is missing required keys.
        """
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        # Verify input data format
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "Missing required input fields"
        )

        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data = data.shard_by_batch_size(dp_size, batch_size=None)

        futures = self.worker_group.run_all_workers_sharded_data(
            "generate",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["tensor_parallel", "pipeline_parallel"],
            output_is_replicated=["tensor_parallel", "pipeline_parallel"],
            common_kwargs={"greedy": greedy},
        )

        assert self.cfg["generation"] is not None, "Generation config is not set"
        result: BatchedDataDict = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures),
            pad_value_dict={"output_ids": self.cfg["generation"]["_pad_token_id"]},
        )

        # Verify output has all required fields
        required_keys = [
            "output_ids",
            "generation_lengths",
            "unpadded_sequence_lengths",
            "logprobs",
        ]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for GenerationOutputSpec: {missing_keys}"
            )

        return result

    def score(
        self, data: "BatchedDataDict[GenerationDatumSpec]"
    ) -> "BatchedDataDict[ScoreOutputSpec]":
        """Score a batch of sequences using the policy.

        Args:
            data: BatchedDataDict containing:
                - input_ids: Tokenized sequences
                - input_lengths: Length of each sequence

        Returns:
            BatchedDataDict containing:
                - scores: Model scores for each sequence

        Raises:
            AssertionError: If data is not a BatchedDataDict or missing required fields.
            ValueError: If output is missing required keys.
        """
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        # Verify input data format
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "Missing required input fields"
        )

        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data = data.shard_by_batch_size(dp_size, batch_size=None)

        futures = self.worker_group.run_all_workers_sharded_data(
            "score",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            output_is_replicated=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            common_kwargs={},
        )

        result: BatchedDataDict = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures),
        )

        required_keys = ["scores"]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for ScoreOutputSpec: {missing_keys}"
            )

        return result

    def prepare_for_generation(self) -> bool:
        """Prepare the policy for generation mode.

        Returns:
            True when preparation is complete.
        """
        return True

    def finish_generation(self) -> bool:
        """Clean up after generation.

        Returns:
            True when cleanup is complete.
        """
        return True

    def invalidate_kv_cache(self) -> bool:
        """Invalidate the KV cache.

        Returns:
            True when cache is invalidated.
        """
        return True

    def update_weights(self, state_dict: dict[str, Any]) -> None:
        """Update model weights from a state dictionary.

        This is typically called after training to sync weights
        with the training policy.

        Args:
            state_dict: Dictionary containing model state.
        """
        # Implementation depends on the specific weight syncing mechanism
        # For now, this is a placeholder for the interface
        pass
