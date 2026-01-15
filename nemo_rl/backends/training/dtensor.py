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
"""DTensor-based training backend implementation.

This module provides the DTensorBackend class that implements the TrainingBackend
protocol using PyTorch's DTensor for distributed training.

Example:
    >>> from nemo_rl.backends.training import DTensorBackend, TrainingBackendConfig
    >>>
    >>> backend = DTensorBackend()
    >>> config = TrainingBackendConfig(
    ...     backend_type='dtensor',
    ...     model_name='meta-llama/Llama-2-7b-hf',
    ...     precision='bfloat16',
    ... )
    >>> backend.setup(config)
    >>> metrics = backend.train_step(batch, loss_fn)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from nemo_rl.backends.training.base import (
    TrainingBackend,
    TrainingBackendConfig,
    register_training_backend,
)

if TYPE_CHECKING:
    from nemo_rl.algorithms.interfaces import LossFunction
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.named_sharding import NamedSharding
    from nemo_rl.distributed.worker_groups import RayWorkerGroup


@register_training_backend("dtensor")
class DTensorBackend:
    """Training backend using PyTorch DTensor for distributed training.

    This backend leverages PyTorch's DTensor (Distributed Tensor) abstraction
    along with FSDP2 (Fully Sharded Data Parallel v2) for efficient distributed
    training of large language models.

    Features:
        - Tensor parallelism via DTensor
        - Data parallelism via FSDP2
        - Context parallelism support
        - Sequence packing for efficiency
        - CPU offloading for memory optimization

    Attributes:
        config: The backend configuration.
        worker_group: Ray worker group for distributed execution.
        sharding_annotations: Named sharding for distributed data.

    Example:
        >>> backend = DTensorBackend()
        >>> backend.setup(TrainingBackendConfig(
        ...     model_name='meta-llama/Llama-2-7b-hf',
        ...     precision='bfloat16',
        ... ))
        >>> metrics = backend.train_step(batch, loss_fn)
    """

    def __init__(
        self,
        worker_group: Optional["RayWorkerGroup"] = None,
        sharding_annotations: Optional["NamedSharding"] = None,
    ):
        """Initialize DTensorBackend.

        Args:
            worker_group: Optional pre-existing Ray worker group.
            sharding_annotations: Optional pre-existing sharding annotations.
        """
        self._config: Optional[TrainingBackendConfig] = None
        self._worker_group = worker_group
        self._sharding_annotations = sharding_annotations
        self._is_initialized = False
        self._policy_config: Optional[dict[str, Any]] = None

    def setup(self, config: TrainingBackendConfig) -> None:
        """Initialize the DTensor backend with configuration.

        Args:
            config: Training backend configuration.

        Raises:
            ValueError: If configuration is invalid.
            RuntimeError: If DTensor initialization fails.
        """
        if config.backend_type != "dtensor":
            raise ValueError(
                f"DTensorBackend received config with backend_type='{config.backend_type}'. "
                "Expected 'dtensor'."
            )

        self._config = config

        # Build policy config from backend config
        self._policy_config = self._build_policy_config(config)

        self._is_initialized = True

    def _build_policy_config(self, config: TrainingBackendConfig) -> dict[str, Any]:
        """Build policy configuration from backend config.

        Args:
            config: Training backend configuration.

        Returns:
            Policy configuration dictionary.
        """
        # Extract dtensor-specific kwargs
        dtensor_kwargs = config.backend_kwargs.get("dtensor_cfg", {})

        return {
            "model_name": config.model_name,
            "precision": config.precision,
            "train_global_batch_size": config.train_global_batch_size,
            "train_micro_batch_size": config.train_micro_batch_size,
            "max_grad_norm": config.max_grad_norm,
            "dtensor_cfg": {
                "enabled": True,
                "tensor_parallel_size": dtensor_kwargs.get("tensor_parallel_size", 1),
                "context_parallel_size": dtensor_kwargs.get("context_parallel_size", 1),
                "sequence_parallel": dtensor_kwargs.get("sequence_parallel", False),
                "cpu_offload": dtensor_kwargs.get("cpu_offload", False),
                "activation_checkpointing": dtensor_kwargs.get(
                    "activation_checkpointing", False
                ),
                "_v2": dtensor_kwargs.get("_v2", True),
                **{k: v for k, v in dtensor_kwargs.items() if k not in [
                    "tensor_parallel_size", "context_parallel_size",
                    "sequence_parallel", "cpu_offload", "activation_checkpointing", "_v2"
                ]},
            },
            "megatron_cfg": {"enabled": False},
            "dynamic_batching": config.backend_kwargs.get(
                "dynamic_batching", {"enabled": False}
            ),
            "sequence_packing": config.backend_kwargs.get(
                "sequence_packing", {"enabled": False}
            ),
            "logprob_batch_size": config.backend_kwargs.get(
                "logprob_batch_size", config.train_micro_batch_size
            ),
            "offload_optimizer_for_logprob": config.backend_kwargs.get(
                "offload_optimizer_for_logprob", False
            ),
            "generation": config.backend_kwargs.get("generation", None),
            "optimizer": config.backend_kwargs.get("optimizer", {
                "name": "torch.optim.AdamW",
                "kwargs": {"lr": 1e-5, "betas": (0.9, 0.999), "eps": 1e-8},
            }),
            "scheduler": config.backend_kwargs.get("scheduler", None),
        }

    def train_step(
        self,
        batch: "BatchedDataDict[Any]",
        loss_fn: "LossFunction",
        eval_mode: bool = False,
        global_batch_size: Optional[int] = None,
        micro_batch_size: Optional[int] = None,
    ) -> dict[str, Any]:
        """Execute a training step using DTensor workers.

        Args:
            batch: Batched input data.
            loss_fn: Loss function to compute gradients.
            eval_mode: If True, run in evaluation mode (no weight updates).
            global_batch_size: Override global batch size.
            micro_batch_size: Override micro batch size.

        Returns:
            Dictionary containing training metrics.

        Raises:
            RuntimeError: If backend is not initialized.
        """
        self._check_initialized()

        gbs = global_batch_size or self._config.train_global_batch_size
        mbs = micro_batch_size or self._config.train_micro_batch_size

        # Shard data across workers
        dp_size = self._sharding_annotations.get_axis_size("data_parallel")
        sharded_data, _ = self._shard_data(batch, gbs)

        # Execute training on workers
        futures = self._worker_group.run_all_workers_sharded_data(
            "train",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            output_is_replicated=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            common_kwargs={
                "loss_fn": loss_fn,
                "eval_mode": eval_mode,
                "gbs": gbs,
                "mbs": mbs,
            },
        )

        results = self._worker_group.get_all_worker_results(futures)
        return self._aggregate_train_results(results)

    def _shard_data(
        self,
        data: "BatchedDataDict[Any]",
        batch_size: Optional[int],
        for_logprobs: bool = False,
    ) -> tuple["BatchedDataDict[Any]", Optional[list[int]]]:
        """Shard data across data parallel workers.

        Args:
            data: Input data to shard.
            batch_size: Target batch size.
            for_logprobs: Whether this is for logprob computation.

        Returns:
            Tuple of (sharded_data, unsorted_indices).
        """
        dp_size = self._sharding_annotations.get_axis_size("data_parallel")

        dynamic_batching = self._policy_config.get("dynamic_batching", {})
        sequence_packing = self._policy_config.get("sequence_packing", {})

        if dynamic_batching.get("enabled", False):
            tokens_key = "logprob_mb_tokens" if for_logprobs else "train_mb_tokens"
            dynamic_batching_args = {
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_round": dynamic_batching.get("sequence_length_round", 1),
                "max_tokens_per_microbatch": dynamic_batching.get(tokens_key, 0),
            }
            return data.shard_by_batch_size(
                dp_size, batch_size=batch_size, dynamic_batching_args=dynamic_batching_args
            )
        elif sequence_packing.get("enabled", False):
            tokens_key = "logprob_mb_tokens" if for_logprobs else "train_mb_tokens"
            sequence_packing_args = {
                "algorithm": sequence_packing.get("algorithm", "first_fit_decreasing"),
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_pad_multiple": 1,
                "max_tokens_per_microbatch": sequence_packing.get(tokens_key, 0),
            }
            return data.shard_by_batch_size(
                dp_size, batch_size=batch_size, sequence_packing_args=sequence_packing_args
            )
        else:
            return data.shard_by_batch_size(dp_size, batch_size=batch_size), None

    def _aggregate_train_results(self, results: list[dict]) -> dict[str, Any]:
        """Aggregate training results from workers.

        Args:
            results: List of result dictionaries from workers.

        Returns:
            Aggregated metrics dictionary.
        """
        from collections import defaultdict

        aggregated = {
            "loss": results[0]["global_loss"],
            "grad_norm": results[0]["grad_norm"],
        }

        if "moe_metrics" in results[0]:
            aggregated["moe_metrics"] = results[0]["moe_metrics"]

        # Aggregate microbatch metrics
        all_mb_metrics = defaultdict(list)
        for r in results:
            for k, v in r.get("all_mb_metrics", {}).items():
                all_mb_metrics[k].extend(v)
        aggregated["all_mb_metrics"] = dict(all_mb_metrics)

        return aggregated

    def get_logprobs(
        self,
        batch: "BatchedDataDict[Any]",
        micro_batch_size: Optional[int] = None,
    ) -> "BatchedDataDict[Any]":
        """Compute log probabilities for the input batch.

        Args:
            batch: Batched input data.
            micro_batch_size: Override micro batch size.

        Returns:
            BatchedDataDict containing 'logprobs' tensor.

        Raises:
            RuntimeError: If backend is not initialized.
        """
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        self._check_initialized()

        sharded_data, unsorted_indices = self._shard_data(batch, None, for_logprobs=True)

        futures = self._worker_group.run_all_workers_sharded_data(
            "get_logprobs",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            output_is_replicated=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            common_kwargs={"micro_batch_size": micro_batch_size},
        )

        logprobs: BatchedDataDict = BatchedDataDict.from_batches(
            self._worker_group.get_all_worker_results(futures)
        )

        if unsorted_indices:
            logprobs.reorder_data(unsorted_indices)

        return logprobs

    def save_checkpoint(
        self,
        path: str | Path,
        optimizer_path: Optional[str | Path] = None,
        tokenizer_path: Optional[str | Path] = None,
    ) -> None:
        """Save model checkpoint to disk.

        Args:
            path: Path to save model weights.
            optimizer_path: Optional path to save optimizer state.
            tokenizer_path: Optional path to save tokenizer.

        Raises:
            RuntimeError: If backend is not initialized.
        """
        import ray

        self._check_initialized()

        kwargs = {
            "weights_path": str(path),
            "optimizer_path": str(optimizer_path) if optimizer_path else None,
            "tokenizer_path": str(tokenizer_path) if tokenizer_path else None,
        }

        ray.get(self._worker_group.run_all_workers_single_data("save_checkpoint", **kwargs))

    def load_checkpoint(
        self,
        path: str | Path,
        optimizer_path: Optional[str | Path] = None,
    ) -> None:
        """Load model checkpoint from disk.

        Args:
            path: Path to load model weights from.
            optimizer_path: Optional path to load optimizer state from.

        Raises:
            RuntimeError: If backend is not initialized.
        """
        import ray

        self._check_initialized()

        kwargs = {
            "weights_path": str(path),
            "optimizer_path": str(optimizer_path) if optimizer_path else None,
        }

        ray.get(self._worker_group.run_all_workers_single_data("load_checkpoint", **kwargs))

    def prepare_for_training(self) -> None:
        """Prepare the backend for training mode."""
        import ray

        self._check_initialized()
        ray.get(self._worker_group.run_all_workers_single_data("prepare_for_training"))

    def prepare_for_inference(self) -> None:
        """Prepare the backend for inference mode (log probability computation)."""
        import ray

        self._check_initialized()
        ray.get(self._worker_group.run_all_workers_single_data("prepare_for_lp_inference"))

    def shutdown(self) -> None:
        """Clean up resources and shut down the backend."""
        import ray

        if self._worker_group is not None:
            try:
                ray.get(self._worker_group.run_all_workers_single_data("shutdown"))
            except Exception:
                pass  # Ignore errors during shutdown
        self._is_initialized = False

    def _check_initialized(self) -> None:
        """Check if the backend is initialized.

        Raises:
            RuntimeError: If backend is not initialized.
        """
        if not self._is_initialized:
            raise RuntimeError(
                "DTensorBackend is not initialized. Call setup() first."
            )

    @property
    def is_initialized(self) -> bool:
        """Check if the backend has been initialized."""
        return self._is_initialized

    @property
    def backend_type(self) -> str:
        """Return the backend type identifier."""
        return "dtensor"

    @property
    def config(self) -> Optional[TrainingBackendConfig]:
        """Return the current configuration."""
        return self._config

    @property
    def policy_config(self) -> Optional[dict[str, Any]]:
        """Return the policy configuration dictionary."""
        return self._policy_config

    def get_reference_policy_logprobs(
        self,
        batch: "BatchedDataDict[Any]",
        micro_batch_size: Optional[int] = None,
    ) -> "BatchedDataDict[Any]":
        """Get log probabilities from the reference policy.

        Args:
            batch: Batched input data.
            micro_batch_size: Override micro batch size.

        Returns:
            BatchedDataDict containing 'reference_logprobs' tensor.
        """
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        self._check_initialized()

        sharded_data, unsorted_indices = self._shard_data(batch, None, for_logprobs=True)

        futures = self._worker_group.run_all_workers_sharded_data(
            "get_reference_policy_logprobs",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            output_is_replicated=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            common_kwargs={"micro_batch_size": micro_batch_size},
        )

        logprobs: BatchedDataDict = BatchedDataDict.from_batches(
            self._worker_group.get_all_worker_results(futures)
        )

        if unsorted_indices:
            logprobs.reorder_data(unsorted_indices)

        return logprobs
