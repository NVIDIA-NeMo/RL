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
import os
import warnings
from contextlib import nullcontext
from typing import Any, Optional, Union

import numpy as np
import ray
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import (
    BatchedDataDict,
    DynamicBatchingArgs,
    SequencePackingArgs,
    SlicedDataDict,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.models.value.config import ValueConfig
from nemo_rl.models.value.interfaces import ValueInterface, ValueOutputSpec
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.timer import Timer

PathLike = Union[str, "os.PathLike[Any]"]


class Value(ValueInterface):
    """Value function model for PPO using distributed training with Ray workers."""

    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: ValueConfig,
        tokenizer: PreTrainedTokenizerBase,
        name_prefix: str = "lm_value",
        workers_per_node: Optional[Union[int, list[int]]] = None,
        init_optimizer: bool = True,
        weights_path: Optional[PathLike] = None,
        optimizer_path: Optional[PathLike] = None,
    ):
        """Initialize the Value model.

        Args:
            cluster: Ray virtual cluster for distributed training
            config: Configuration for the value model
            tokenizer: Tokenizer for the model
            name_prefix: Prefix for worker names
            workers_per_node: Number of workers per node
            init_optimizer: Whether to initialize the optimizer
            weights_path: Path to load model weights from
            optimizer_path: Path to load optimizer state from
        """
        if weights_path:
            weights_path = os.path.abspath(weights_path)
        if optimizer_path:
            optimizer_path = os.path.abspath(optimizer_path)

        worker_builder_cls: str
        tp_size = 1
        pp_size = 1
        cp_size = 1

        # Value models use the same backend configuration as policy models
        megatron_enable = bool(config.get("megatron_cfg", {}).get("enabled", False))
        dtensor_enable = bool(config.get("dtensor_cfg", {}).get("enabled", False))

        if megatron_enable and dtensor_enable:
            raise ValueError(
                "Configure either Megatron (value.megatron_cfg.enabled=true) or "
                "DTensor (value.dtensor_cfg.enabled=true), not both."
            )

        if megatron_enable:
            raise NotImplementedError(
                "Megatron backend is not yet implemented for Value models. "
                "Please use DTensor backend (value.dtensor_cfg.enabled=true)."
            )
        else:
            if not dtensor_enable:
                raise ValueError(
                    "Please set value.dtensor_cfg.enabled=true to use DTensor training backend."
                )

            # Check if _v2 is enabled (defaults to False for backward compatibility)
            use_v2 = config.get("dtensor_cfg", {}).get("_v2", False)
            if use_v2:
                worker_builder_cls = "nemo_rl.models.value.workers.dtensor_value_worker_v2.DTensorValueWorkerV2"

                if "TORCH_CUDA_ARCH_LIST" not in os.environ:
                    warnings.warn(
                        "TORCH_CUDA_ARCH_LIST is not set. This is needed if using DeepEP in DTensorValueWorker V2. "
                        "This variable is set in our container, but if you are running a custom container or baremetal, "
                        "you may need to set this variable manually. Example: export TORCH_CUDA_ARCH_LIST='9.0 10.0'"
                    )
            else:
                raise NotImplementedError(
                    "DTensor V1 backend is not implemented for Value models. "
                    "Please set value.dtensor_cfg._v2=true to use DTensor V2."
                )

            tp_size = config["dtensor_cfg"]["tensor_parallel_size"]
            cp_size = config["dtensor_cfg"]["context_parallel_size"]

            env_vars = config["dtensor_cfg"].get("env_vars", {})

        # Validate world_size compatibility with parallelism configuration
        model_parallel_size = pp_size * cp_size * tp_size
        actual_world_size = cluster.world_size()

        if actual_world_size < model_parallel_size:
            raise ValueError(
                f"World size ({actual_world_size}) is insufficient for the parallelism configuration. "
                f"Required minimum world size: PP({pp_size}) * CP({cp_size}) * TP({tp_size}) = {model_parallel_size}. "
                f"This would result in DP = {actual_world_size}/{model_parallel_size} = {actual_world_size / model_parallel_size:.3f}, but DP must be ≥ 1. "
                f"Please either increase the number of GPUs/nodes or reduce the parallelism parameters."
            )

        if actual_world_size % model_parallel_size != 0:
            dp_size_float = actual_world_size / model_parallel_size
            raise ValueError(
                f"World size ({actual_world_size}) must be divisible by PP * CP * TP ({model_parallel_size}). "
                f"The data parallel size (DP = world_size / (PP * CP * TP)) must be a positive integer. "
                f"Current DP would be {actual_world_size}/{model_parallel_size} = {dp_size_float:.6f}, which is not an integer. "
                f"Please adjust your cluster size or parallelism parameters."
            )

        self.sharding_annotations = NamedSharding(
            layout=np.arange(cluster.world_size()).reshape(
                pp_size,  # PP
                -1,  # DP
                cp_size,  # CP
                tp_size,  # TP
            ),
            names=[
                "pipeline_parallel",
                "data_parallel",
                "context_parallel",
                "tensor_parallel",
            ],
        )

        from ray.util.queue import Queue as RayQueue

        pre_init_queue = RayQueue()
        worker_builder = RayWorkerBuilder(
            worker_builder_cls,
            config,
            tokenizer=tokenizer,
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            worker_sharding_annotations=self.sharding_annotations,
            pre_init_communication_queue=pre_init_queue,
        )

        if cluster._sorted_bundle_indices is not None:
            # The cluster has initialized a unified placement group across nodes
            group_size = cluster.num_gpus_per_node
            tied_groups = [
                (i // group_size, [bundle_idx])
                for i, bundle_idx in enumerate(cluster._sorted_bundle_indices)
            ]

            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                bundle_indices_list=tied_groups,
                sharding_annotations=self.sharding_annotations,
                env_vars=env_vars or {},
            )
        else:
            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                workers_per_node=workers_per_node,
                sharding_annotations=self.sharding_annotations,
                env_vars=env_vars or {},
            )

        # Configure dynamic batching
        if config["dynamic_batching"]["enabled"]:
            assert pp_size == 1, (
                "Dynamic batching is only supported for single pipeline parallel stage"
            )
            self.use_dynamic_batches = True
            self.dynamic_batching_args: DynamicBatchingArgs = {
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_round": config["dynamic_batching"][
                    "sequence_length_round"
                ],
                "max_tokens_per_microbatch": 0,  # Override in each call
            }
            assert not config["sequence_packing"]["enabled"], (
                "Dynamic Batching is exclusive of Sequence Packing. Please disable Sequence Packing to use Dynamic Batching"
            )
        else:
            self.use_dynamic_batches = False

        # Configure sequence packing
        if config["sequence_packing"]["enabled"]:
            self.use_sequence_packing = True
            sequence_length_pad_multiple = (
                cp_size * 2 * tp_size if cp_size > 1 else tp_size
            )
            self.sequence_packing_args: SequencePackingArgs = {
                "algorithm": config["sequence_packing"]["algorithm"],
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_pad_multiple": sequence_length_pad_multiple,
            }
            assert not config["dynamic_batching"]["enabled"], (
                "Sequence Packing is exclusive of Dynamic Batching. Please disable Dynamic Batching"
            )
        else:
            self.use_sequence_packing = False

        self.cfg = config

    def get_values(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        timer: Optional[Timer] = None,
    ) -> BatchedDataDict[ValueOutputSpec]:
        """Get value predictions for a batch of data.

        Args:
            data: BatchedDataDict containing input sequences
            timer: Optional timer for profiling

        Returns:
            BatchedDataDict containing value predictions [batch_size, sequence_length]
        """
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict]
        unsorted_data_indices: list[int]

        with timer.time("get_values/shard_data") if timer else nullcontext():
            if self.use_dynamic_batches:
                self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg[
                    "dynamic_batching"
                ]["logprob_mb_tokens"]
                sharded_data, unsorted_data_indices = data.shard_by_batch_size(  # type: ignore
                    dp_size,
                    batch_size=None,
                    dynamic_batching_args=self.dynamic_batching_args,
                )
            elif self.use_sequence_packing:
                self.sequence_packing_args["max_tokens_per_microbatch"] = self.cfg[
                    "sequence_packing"
                ]["logprob_mb_tokens"]
                sharded_data, unsorted_data_indices = data.shard_by_batch_size(
                    dp_size,
                    batch_size=None,
                    sequence_packing_args=self.sequence_packing_args,
                )
            else:
                sharded_data = data.shard_by_batch_size(  # type: ignore
                    dp_size,
                    batch_size=None,
                )

        with (
            timer.time("get_values/submit_value_futures")
            if timer
            else nullcontext()
        ):
            futures = self.worker_group.run_all_workers_sharded_data(
                "get_values",
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
            )
        values: BatchedDataDict[ValueOutputSpec] = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )

        # Reorder data if dynamic batching or sequence packing was used
        if self.use_dynamic_batches or self.use_sequence_packing:
            values.reorder_data(unsorted_data_indices)

        return values

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
        """Train the value function on a batch of data with a given loss function.

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
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]

        # Shard and replicate the batch
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        with timer.time("value_training/sharding_data") if timer else nullcontext():
            if self.use_dynamic_batches:
                self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg[
                    "dynamic_batching"
                ]["train_mb_tokens"]
                sharded_data, _ = data.shard_by_batch_size(
                    dp_size,
                    batch_size=batch_size,
                    dynamic_batching_args=self.dynamic_batching_args,
                )
            elif self.use_sequence_packing:
                self.sequence_packing_args["max_tokens_per_microbatch"] = self.cfg[
                    "sequence_packing"
                ]["train_mb_tokens"]
                sharded_data, _ = data.shard_by_batch_size(
                    dp_size,
                    batch_size=batch_size,
                    sequence_packing_args=self.sequence_packing_args,
                )
            else:
                sharded_data = data.shard_by_batch_size(
                    dp_size,
                    batch_size=batch_size,
                )

        # Train each shard in parallel
        with (
            timer.time("value_training/submit_training_futures")
            if timer
            else nullcontext()
        ):
            futures = self.worker_group.run_all_workers_sharded_data(
                "train",
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
                common_kwargs={
                    "loss_fn": loss_fn,
                    "eval_mode": eval_mode,
                    "gbs": batch_size,
                    "mbs": micro_batch_size,
                },
            )
        results = self.worker_group.get_all_worker_results(futures)

        # Aggregate the results
        aggregated_results = {
            "loss": results[0]["global_loss"],
            "grad_norm": results[0]["grad_norm"],
        }

        # Aggregate metrics across all workers
        from collections import defaultdict

        all_mb_metrics = defaultdict(list)
        for r in results:
            for k, v in r["all_mb_metrics"].items():
                all_mb_metrics[k].extend(v)
        aggregated_results["all_mb_metrics"] = dict(all_mb_metrics)

        return aggregated_results

    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        """Prepare the value model for training (load to GPU)."""
        futures = self.worker_group.run_all_workers_single_data("prepare_for_training")
        ray.get(futures)

    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        """Clean up after training."""
        pass

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        checkpointing_cfg: Optional[CheckpointingConfig] = None,
    ) -> None:
        """Save a checkpoint of the value model."""
        use_v2 = self.cfg.get("dtensor_cfg", {}).get("_v2", False)

        if use_v2:
            futures = self.worker_group.run_all_workers_single_data(
                "save_checkpoint",
                weights_path=weights_path,
                optimizer_path=optimizer_path,
                tokenizer_path=tokenizer_path,
                checkpointing_cfg=checkpointing_cfg,
            )
        else:
            if (
                checkpointing_cfg is not None
                and checkpointing_cfg.get("model_save_format", None) is not None
            ):
                raise ValueError(
                    "model_save_format must be None or omitted if using DTensorValueWorker (_v2=False)."
                )
            futures = self.worker_group.run_all_workers_single_data(
                "save_checkpoint",
                weights_path=weights_path,
                optimizer_path=optimizer_path,
                tokenizer_path=tokenizer_path,
            )
        ray.get(futures)

    def shutdown(self) -> bool:
        """Shut down all value workers and clean up resources."""
        try:
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during value model shutdown: {e}")
            return False

    def __del__(self) -> None:
        """Shuts down the worker groups when the object is deleted or garbage collected."""
        if hasattr(self, "worker_group"):
            self.worker_group.shutdown(cleanup_method="shutdown")
