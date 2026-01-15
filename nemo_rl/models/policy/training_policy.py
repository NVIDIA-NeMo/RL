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
"""Training-focused policy implementation."""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional, Protocol

if TYPE_CHECKING:
    from nemo_rl.algorithms.interfaces import LossFunction
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.distributed.named_sharding import NamedSharding
    from nemo_rl.distributed.worker_groups import RayWorkerGroup
    from nemo_rl.models.generation.interfaces import GenerationDatumSpec
    from nemo_rl.models.policy import PolicyConfig
    from nemo_rl.models.policy.interfaces import (
        LogprobOutputSpec,
        ReferenceLogprobOutputSpec,
        TopkLogitsOutputSpec,
    )
    from nemo_rl.utils.checkpoint import CheckpointingConfig
    from nemo_rl.utils.flops_tracker import FLOPTracker


class TrainingPolicyProtocol(Protocol):
    """Protocol defining the training policy interface."""

    def train(
        self,
        data: "BatchedDataDict[Any]",
        loss_fn: "LossFunction",
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]: ...

    def get_logprobs(
        self, data: "BatchedDataDict[GenerationDatumSpec]"
    ) -> "BatchedDataDict[LogprobOutputSpec]": ...

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        checkpointing_cfg: Optional["CheckpointingConfig"] = None,
    ) -> None: ...


class TrainingPolicy:
    """Policy class focused on training: train(), get_logprobs(), save_checkpoint()."""

    def __init__(
        self,
        worker_group: "RayWorkerGroup",
        sharding_annotations: "NamedSharding",
        cfg: "PolicyConfig",
        flops_tracker: Optional["FLOPTracker"] = None,
    ):
        self.worker_group = worker_group
        self.sharding_annotations = sharding_annotations
        self.cfg = cfg
        self.flops_tracker = flops_tracker
        self._init_batching_config()

    def _init_batching_config(self) -> None:
        """Initialize dynamic batching and sequence packing configurations."""
        cfg = self.cfg
        self.use_dynamic_batches = cfg["dynamic_batching"]["enabled"]
        if self.use_dynamic_batches:
            self.dynamic_batching_args = {
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_round": cfg["dynamic_batching"]["sequence_length_round"],
                "max_tokens_per_microbatch": 0,
            }

        self.use_sequence_packing = cfg.get("sequence_packing", {}).get("enabled", False)
        if self.use_sequence_packing:
            cp_size = self._get_parallel_size("context_parallel")
            tp_size = self._get_parallel_size("tensor_parallel")
            pad_mult = cp_size * 2 * tp_size if cp_size > 1 else tp_size
            self.sequence_packing_args = {
                "algorithm": cfg["sequence_packing"]["algorithm"],
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_pad_multiple": pad_mult,
            }

    def _get_parallel_size(self, parallel_type: str) -> int:
        """Get parallel size from config."""
        if self.cfg.get("megatron_cfg", {}).get("enabled", False):
            key = "tensor_model_parallel_size" if parallel_type == "tensor_parallel" else "context_parallel_size"
            return self.cfg["megatron_cfg"].get(key, 1)
        elif self.cfg.get("dtensor_cfg", {}).get("enabled", False):
            key = "tensor_parallel_size" if parallel_type == "tensor_parallel" else "context_parallel_size"
            return self.cfg["dtensor_cfg"].get(key, 1)
        return 1

    def _shard_data(self, data, batch_size, for_logprobs=False):
        """Shard data across data parallel workers."""
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        tokens_key = "logprob_mb_tokens" if for_logprobs else "train_mb_tokens"
        unsorted_indices = None

        if self.use_dynamic_batches:
            self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg["dynamic_batching"][tokens_key]
            sharded, unsorted_indices = data.shard_by_batch_size(
                dp_size, batch_size=batch_size, dynamic_batching_args=self.dynamic_batching_args
            )
        elif self.use_sequence_packing:
            self.sequence_packing_args["max_tokens_per_microbatch"] = self.cfg["sequence_packing"][tokens_key]
            sharded, unsorted_indices = data.shard_by_batch_size(
                dp_size, batch_size=batch_size, sequence_packing_args=self.sequence_packing_args
            )
        else:
            sharded = data.shard_by_batch_size(dp_size, batch_size=batch_size)
        return sharded, unsorted_indices

    def train(
        self,
        data: "BatchedDataDict[Any]",
        loss_fn: "LossFunction",
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]
        sharded_data, _ = self._shard_data(data, batch_size, for_logprobs=False)

        if self.flops_tracker:
            self.flops_tracker.reset()
            for shard in sharded_data:
                self.flops_tracker.track_batch(shard["input_lengths"].tolist())

        futures = self.worker_group.run_all_workers_sharded_data(
            "train", data=sharded_data, in_sharded_axes=["data_parallel"],
            replicate_on_axes=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            output_is_replicated=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            common_kwargs={"loss_fn": loss_fn, "eval_mode": eval_mode, "gbs": batch_size, "mbs": micro_batch_size},
        )
        results = self.worker_group.get_all_worker_results(futures)
        return self._aggregate_train_results(results)

    def _aggregate_train_results(self, results: list[dict]) -> dict[str, Any]:
        """Aggregate training results from workers."""
        aggregated = {"loss": results[0]["global_loss"], "grad_norm": results[0]["grad_norm"]}
        if "moe_metrics" in results[0]:
            aggregated["moe_metrics"] = results[0]["moe_metrics"]

        if self.flops_tracker:
            aggregated["total_flops"] = self.flops_tracker.total_flops
            aggregated["num_ranks"] = self.worker_group.cluster.world_size()
            try:
                from nemo_rl.utils.flops_tracker import get_theoretical_tflops
                gpus_per_worker = self.worker_group.cluster.world_size() / len(results)
                aggregated["theoretical_tflops"] = gpus_per_worker * sum(
                    get_theoretical_tflops(r["gpu_name"], r["model_dtype"]) for r in results
                )
            except Exception as e:
                warnings.warn(f"Error getting theoretical flops: {e}")

        all_mb_metrics = defaultdict(list)
        for r in results:
            for k, v in r["all_mb_metrics"].items():
                all_mb_metrics[k].extend(v)
        aggregated["all_mb_metrics"] = dict(all_mb_metrics)
        return aggregated

    def get_logprobs(self, data: "BatchedDataDict[GenerationDatumSpec]") -> "BatchedDataDict[LogprobOutputSpec]":
        """Get the log probabilities of the model for given data."""
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        sharded_data, unsorted_indices = self._shard_data(data, None, for_logprobs=True)
        futures = self.worker_group.run_all_workers_sharded_data(
            "get_logprobs", data=sharded_data, in_sharded_axes=["data_parallel"],
            replicate_on_axes=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            output_is_replicated=["context_parallel", "tensor_parallel", "pipeline_parallel"],
        )
        logprobs: BatchedDataDict = BatchedDataDict.from_batches(self.worker_group.get_all_worker_results(futures))
        if unsorted_indices:
            logprobs.reorder_data(unsorted_indices)
        return logprobs

    def get_reference_policy_logprobs(
        self, data: "BatchedDataDict[GenerationDatumSpec]", micro_batch_size: Optional[int] = None
    ) -> "BatchedDataDict[ReferenceLogprobOutputSpec]":
        """Get the log probabilities of the reference policy for given data."""
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        sharded_data, unsorted_indices = self._shard_data(data, None, for_logprobs=True)
        futures = self.worker_group.run_all_workers_sharded_data(
            "get_reference_policy_logprobs", data=sharded_data, in_sharded_axes=["data_parallel"],
            replicate_on_axes=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            output_is_replicated=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            common_kwargs={"micro_batch_size": micro_batch_size},
        )
        logprobs: BatchedDataDict = BatchedDataDict.from_batches(self.worker_group.get_all_worker_results(futures))
        if unsorted_indices:
            logprobs.reorder_data(unsorted_indices)
        return logprobs

    def get_topk_logits(
        self, data: "BatchedDataDict[GenerationDatumSpec]", k: int, micro_batch_size: Optional[int] = None
    ) -> "BatchedDataDict[TopkLogitsOutputSpec]":
        """Get per-position top-k logits and global indices for a batch of inputs."""
        import torch
        from nemo_rl.distributed.batched_data_dict import BatchedDataDict

        sharded_data, unsorted_indices = self._shard_data(data, None, for_logprobs=True)
        futures = self.worker_group.run_all_workers_sharded_data(
            "get_topk_logits", data=sharded_data, in_sharded_axes=["data_parallel"],
            replicate_on_axes=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            output_is_replicated=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            common_kwargs={"k": k, "micro_batch_size": micro_batch_size},
        )
        worker_batches = self.worker_group.get_all_worker_results(futures)
        stacked: BatchedDataDict = BatchedDataDict()
        stacked["topk_logits"] = torch.cat([wb["topk_logits"] for wb in worker_batches], dim=0)
        stacked["topk_indices"] = torch.cat([wb["topk_indices"] for wb in worker_batches], dim=0)
        if unsorted_indices:
            stacked.reorder_data(unsorted_indices)
        return stacked

    def prepare_for_training(self) -> None:
        """Prepare the policy for training mode."""
        import ray
        ray.get(self.worker_group.run_all_workers_single_data("prepare_for_training"))

    def prepare_for_lp_inference(self) -> None:
        """Prepare the policy for log probability inference."""
        import ray
        ray.get(self.worker_group.run_all_workers_single_data("prepare_for_lp_inference"))

    def finish_training(self) -> None:
        """Clean up after training."""
        pass

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        checkpointing_cfg: Optional["CheckpointingConfig"] = None,
    ) -> None:
        """Save a checkpoint of the model."""
        import ray
        use_v2 = self.cfg.get("dtensor_cfg", {}).get("_v2", False)
        kwargs = {"weights_path": weights_path, "optimizer_path": optimizer_path, "tokenizer_path": tokenizer_path}
        if use_v2:
            kwargs["checkpointing_cfg"] = checkpointing_cfg
        elif checkpointing_cfg and checkpointing_cfg.get("model_save_format"):
            raise ValueError("model_save_format must be None if using DTensorPolicyWorker (_v2=False).")
        ray.get(self.worker_group.run_all_workers_single_data("save_checkpoint", **kwargs))

    def calibrate_qkv_fp8_scales(
        self, data: "BatchedDataDict[GenerationDatumSpec]", micro_batch_size: Optional[int] = None,
        percentile: float = 99.9, margin: float = 1.05, include_q: bool = False,
    ) -> dict[str, Any]:
        """Trigger KV-cache FP8 scale calibration across workers."""
        sharded_data, _ = self._shard_data(data, None, for_logprobs=True)
        futures = self.worker_group.run_all_workers_sharded_data(
            "calibrate_qkv_fp8_scales", data=sharded_data, in_sharded_axes=["data_parallel"],
            replicate_on_axes=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            output_is_replicated=["context_parallel", "tensor_parallel", "pipeline_parallel"],
            common_kwargs={"micro_batch_size": micro_batch_size, "percentile": percentile, "margin": margin, "include_q": include_q},
        )
        return self.worker_group.get_all_worker_results(futures)[0]

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        import ray
        ray.get(self.worker_group.run_all_workers_single_data("start_gpu_profiling"))

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        import ray
        ray.get(self.worker_group.run_all_workers_single_data("stop_gpu_profiling"))
