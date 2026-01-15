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
"""Unified Policy class combining training and generation capabilities.

This module provides the Policy class that serves as a facade combining
TrainingPolicy and GenerationPolicy functionality. For new code, consider
using TrainingPolicy and GenerationPolicy directly for cleaner separation
of concerns.

Example:
    >>> from nemo_rl.models.policy import Policy
    >>> policy = Policy(cluster, config, tokenizer)
    >>> # Use for training
    >>> results = policy.train(data, loss_fn)
    >>> # Use for generation
    >>> outputs = policy.generate(prompts)

See Also:
    - TrainingPolicy: For training-only use cases
    - GenerationPolicy: For generation-only use cases
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import ray
from ray.util.queue import Queue as RayQueue
from transformers import AutoProcessor, PreTrainedTokenizerBase

from nemo_rl.distributed.batched_data_dict import (
    BatchedDataDict,
    DynamicBatchingArgs,
    SequencePackingArgs,
    SlicedDataDict,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.generation_policy import GenerationPolicy
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
    ScoreOutputSpec,
    TopkLogitsOutputSpec,
)
from nemo_rl.models.policy.training_policy import TrainingPolicy
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.flops_tracker import (
    FLOPTracker,
    get_default_hf_config,
)

if TYPE_CHECKING:
    from nemo_rl.algorithms.interfaces import LossFunction

PathLike = Union[str, "os.PathLike[Any]"]


class Policy(ColocatablePolicyInterface, GenerationInterface):
    """Unified Policy combining training and generation capabilities.

    This class serves as a facade that combines TrainingPolicy and GenerationPolicy
    functionality for backward compatibility. It maintains the original interface
    while delegating to the focused implementations.

    For new code, consider using TrainingPolicy and GenerationPolicy directly:
        - TrainingPolicy: train(), get_logprobs(), save_checkpoint()
        - GenerationPolicy: generate(), score()

    Attributes:
        worker_group: Ray worker group for distributed execution.
        sharding_annotations: Named sharding for distributed data.
        cfg: Policy configuration.
        training_policy: Delegate for training operations.
        generation_policy: Delegate for generation operations.

    Example:
        >>> policy = Policy(cluster, config, tokenizer)
        >>> # Training
        >>> results = policy.train(data, loss_fn)
        >>> # Generation
        >>> outputs = policy.generate(prompts)
        >>> # Save checkpoint
        >>> policy.save_checkpoint(weights_path, optimizer_path)
    """

    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: PolicyConfig,
        tokenizer: PreTrainedTokenizerBase,
        name_prefix: str = "lm_policy",
        workers_per_node: Optional[Union[int, list[int]]] = None,
        init_optimizer: bool = True,
        weights_path: Optional[PathLike] = None,
        optimizer_path: Optional[PathLike] = None,
        init_reference_model: bool = True,
        processor: Optional[AutoProcessor] = None,
    ):
        """Initialize the Policy.

        Args:
            cluster: Ray virtual cluster for distributed execution.
            config: Policy configuration dictionary.
            tokenizer: Tokenizer for text processing.
            name_prefix: Prefix for worker names.
            workers_per_node: Number of workers per node.
            init_optimizer: Whether to initialize the optimizer.
            weights_path: Path to load model weights from.
            optimizer_path: Path to load optimizer state from.
            init_reference_model: Whether to initialize a reference model.
            processor: Optional processor for multimodal models.
        """
        if weights_path:
            weights_path = os.path.abspath(weights_path)
        if optimizer_path:
            optimizer_path = os.path.abspath(optimizer_path)

        worker_builder_cls: str
        tp_size = 1
        pp_size = 1
        cp_size = 1

        megatron_enable = bool(config.get("megatron_cfg", {}).get("enabled", False))
        dtensor_enable = bool(config.get("dtensor_cfg", {}).get("enabled", False))
        if megatron_enable and dtensor_enable:
            raise ValueError(
                "Configure either Megatron (policy.megatron_cfg.enabled=true) or "
                "DTensor (policy.dtensor_cfg.enabled=true), not both."
            )
        if megatron_enable:
            worker_builder_cls = "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
            tp_size = config["megatron_cfg"]["tensor_model_parallel_size"]
            pp_size = config["megatron_cfg"]["pipeline_model_parallel_size"]
            cp_size = config["megatron_cfg"]["context_parallel_size"]

            env_vars = config["megatron_cfg"].get("env_vars", {})

            if "TORCH_CUDA_ARCH_LIST" not in os.environ:
                raise RuntimeError(
                    "TORCH_CUDA_ARCH_LIST is not set. This is required in Megatron backend."
                )

        else:
            if not dtensor_enable:
                raise ValueError(
                    "Please either set policy.megatron_cfg.enabled=true to use Megatron training backend "
                    "or set policy.dtensor_cfg.enabled=true to use DTensor training backend."
                )

            use_v2 = config.get("dtensor_cfg", {}).get("_v2", False)
            if use_v2:
                worker_builder_cls = "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"

                if "TORCH_CUDA_ARCH_LIST" not in os.environ:
                    warnings.warn(
                        "TORCH_CUDA_ARCH_LIST is not set. This is needed if using DeepEP."
                    )
            else:
                assert (
                    config["dtensor_cfg"].get("lora_cfg", {}).get("enabled", False)
                    is False
                ), "LoRA is not supported for DTensorPolicyWorker V1"
                worker_builder_cls = "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker"

            tp_size = config["dtensor_cfg"]["tensor_parallel_size"]
            cp_size = config["dtensor_cfg"]["context_parallel_size"]

            env_vars = config["dtensor_cfg"].get("env_vars", {})

        # Validate world_size compatibility
        model_parallel_size = pp_size * cp_size * tp_size
        actual_world_size = cluster.world_size()

        if actual_world_size < model_parallel_size:
            raise ValueError(
                f"World size ({actual_world_size}) is insufficient for the parallelism configuration. "
                f"Required minimum world size: PP({pp_size}) * CP({cp_size}) * TP({tp_size}) = {model_parallel_size}."
            )

        if actual_world_size % model_parallel_size != 0:
            dp_size_float = actual_world_size / model_parallel_size
            raise ValueError(
                f"World size ({actual_world_size}) must be divisible by PP * CP * TP ({model_parallel_size}). "
                f"Current DP would be {dp_size_float:.6f}, which is not an integer."
            )

        self.sharding_annotations = NamedSharding(
            layout=np.arange(cluster.world_size()).reshape(
                pp_size,
                -1,  # DP
                cp_size,
                tp_size,
            ),
            names=[
                "pipeline_parallel",
                "data_parallel",
                "context_parallel",
                "tensor_parallel",
            ],
        )

        pre_init_queue = RayQueue()
        worker_builder = RayWorkerBuilder(
            worker_builder_cls,
            config,
            tokenizer=tokenizer,
            processor=processor,
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_reference_model=init_reference_model,
            worker_sharding_annotations=self.sharding_annotations,
            pre_init_communication_queue=pre_init_queue,
        )

        if cluster._sorted_bundle_indices is not None:
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

        self.cfg = config

        # Initialize FLOPs tracker
        try:
            flops_tracker = FLOPTracker.from_config(
                config["model_name"], get_default_hf_config(config["model_name"])
            )
        except ValueError as e:
            flops_tracker = None
            print(f"FLOPS tracker not supported for model {config['model_name']}: {e}")

        # Create delegate policies
        self._training_policy = TrainingPolicy(
            worker_group=self.worker_group,
            sharding_annotations=self.sharding_annotations,
            cfg=config,
            flops_tracker=flops_tracker,
        )

        self._generation_policy = GenerationPolicy(
            worker_group=self.worker_group,
            sharding_annotations=self.sharding_annotations,
            cfg=config,
        )

        # Keep legacy attributes for backward compatibility
        self.flops_tracker = flops_tracker
        self.use_dynamic_batches = self._training_policy.use_dynamic_batches
        self.use_sequence_packing = self._training_policy.use_sequence_packing
        if self.use_dynamic_batches:
            self.dynamic_batching_args = self._training_policy.dynamic_batching_args
        if self.use_sequence_packing:
            self.sequence_packing_args = self._training_policy.sequence_packing_args

    # =========================================================================
    # Training Methods (delegated to TrainingPolicy)
    # =========================================================================

    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: "LossFunction",
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        return self._training_policy.train(data, loss_fn, eval_mode, gbs, mbs)

    def get_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a data dict."""
        return self._training_policy.get_logprobs(data)

    def get_reference_policy_logprobs(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs of the reference policy for a data dict."""
        return self._training_policy.get_reference_policy_logprobs(
            data, micro_batch_size
        )

    def get_topk_logits(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[TopkLogitsOutputSpec]:
        """Dispatch get_topk_logits to workers."""
        return self._training_policy.get_topk_logits(data, k, micro_batch_size)

    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        """Prepare for training mode."""
        self._training_policy.prepare_for_training()

    def prepare_for_lp_inference(self, *args: Any, **kwargs: Any) -> None:
        """Prepare for log probability inference."""
        self._training_policy.prepare_for_lp_inference()

    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        """Clean up after training."""
        self._training_policy.finish_training()

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        checkpointing_cfg: Optional[CheckpointingConfig] = None,
    ) -> None:
        """Save a checkpoint of the model."""
        self._training_policy.save_checkpoint(
            weights_path, optimizer_path, tokenizer_path, checkpointing_cfg
        )

    def calibrate_qkv_fp8_scales(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """Trigger KV-cache FP8 scale calibration."""
        return self._training_policy.calibrate_qkv_fp8_scales(
            data, micro_batch_size, percentile, margin, include_q
        )

    # =========================================================================
    # Generation Methods (delegated to GenerationPolicy)
    # =========================================================================

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using the policy."""
        return self._generation_policy.generate(data, greedy)

    def score(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict[ScoreOutputSpec]:
        """Score a batch of data using the policy."""
        return self._generation_policy.score(data)

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Prepare for generation mode."""
        return self._generation_policy.prepare_for_generation()

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Clean up after generation."""
        return self._generation_policy.finish_generation()

    def invalidate_kv_cache(self, *args: Any, **kwargs: Any) -> bool:
        """Invalidate the KV cache."""
        return self._generation_policy.invalidate_kv_cache()

    # =========================================================================
    # Infrastructure Methods (colocation, weight sync, profiling)
    # =========================================================================

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        """Initialize the collective communication."""
        futures = self.worker_group.run_all_workers_single_data(
            "init_collective",
            ip=ip,
            port=port,
            world_size=world_size,
            train_world_size=train_world_size,
        )
        return futures

    def prepare_refit_info(self) -> Optional[dict[str, Any]]:
        """Prepare the info for refit."""
        futures = self.worker_group.run_all_workers_single_data("prepare_refit_info")
        results = ray.get(futures)
        return results[0]

    def get_free_memory_bytes(self) -> int:
        """Get the available free memory."""
        futures = self.worker_group.run_all_workers_single_data("get_free_memory_bytes")
        free_memory_bytes = min(ray.get(future) for future in futures)
        return free_memory_bytes

    def stream_weights_via_ipc_zmq(
        self, buffer_size_bytes: int, kv_scales: Optional[dict[str, float]] = None
    ) -> list[ray.ObjectRef]:
        """Send the weights for IPC handles via ZMQ socket."""
        futures = self.worker_group.run_all_workers_single_data(
            "stream_weights_via_ipc_zmq",
            buffer_size_bytes=buffer_size_bytes,
            kv_scales=kv_scales,
        )
        return futures

    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> list[ray.ObjectRef]:
        """Broadcast the weights for collective communication."""
        futures = self.worker_group.run_all_workers_single_data(
            "broadcast_weights_for_collective",
            kv_scales=kv_scales,
        )
        return futures

    def offload_before_refit(self) -> None:
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data("offload_before_refit")
        ray.get(futures)

    def offload_after_refit(self) -> None:
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data("offload_after_refit")
        ray.get(futures)

    def shutdown(self) -> bool:
        """Shut down all workers and clean up resources."""
        try:
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during policy shutdown: {e}")
            return False

    def __del__(self) -> None:
        """Shuts down worker groups when object is deleted."""
        if hasattr(self, "worker_group"):
            self.worker_group.shutdown(cleanup_method="shutdown")

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        self._training_policy.start_gpu_profiling()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        self._training_policy.stop_gpu_profiling()

    def print_node_ip_and_gpu_id(self) -> list[tuple[str, int]]:
        """Print the node IP and GPU ID of the current worker."""
        results = ray.get(
            self.worker_group.run_all_workers_single_data(
                "report_node_ip_and_gpu_id",
            )
        )
        all_node_ips = sorted(set([result[0] for result in results]))
        all_gpu_ids = sorted(set([result[1] for result in results]))

        worker_id_list = [
            [list() for _ in range(len(all_gpu_ids))] for _ in range(len(all_node_ips))
        ]
        for worker_id, (ip, gpu_id) in enumerate(results):
            node_idx = all_node_ips.index(ip)
            gpu_idx = all_gpu_ids.index(gpu_id)
            worker_id_list[node_idx][gpu_idx].append("worker-" + str(worker_id))

        from prettytable import PrettyTable

        table = PrettyTable()
        table.title = "Policy worker mapping to Nodes and GPUs"
        table.field_names = ["Node_IP"] + [
            "GPU_ID=" + str(gpu_id) for gpu_id in all_gpu_ids
        ]
        for i, node_idx in enumerate(all_node_ips):
            row = [node_idx]
            for j in range(len(all_gpu_ids)):
                row.append(tuple(worker_id_list[i][j]))
            table.add_row(row)

        print(table)
