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
from collections import defaultdict
from typing import Any, Optional, Union

import numpy as np
import ray
import torch
from ray.util.queue import Queue as RayQueue
from transformers import AutoProcessor, PreTrainedTokenizerBase

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
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
    ReferenceLogprobOutputSpec,
    ScoreOutputSpec,
    TopkLogitsOutputSpec,
)
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.flops_tracker import (
    FLOPTracker,
    get_default_hf_config,
    get_theoretical_tflops,
)

PathLike = Union[str, "os.PathLike[Any]"]


class Policy(ColocatablePolicyInterface, GenerationInterface):
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
                    "TORCH_CUDA_ARCH_LIST is not set. This is required in Megatron backend. This variable is set in our container, but "
                    "if you are running a custom container or baremetal, you may need to set this variable manually. Example: export TORCH_CUDA_ARCH_LIST='9.0 10.0'"
                )

        else:
            if not dtensor_enable:
                raise ValueError(
                    "Please either set policy.megatron_cfg.enabled=true to use Megatron training backend "
                    "or set policy.dtensor_cfg.enabled=true to use DTensor training backend."
                )

            # Check if _v2 is enabled in dtensor_cfg (defaults to False for backward compatibility)
            use_v2 = config.get("dtensor_cfg", {}).get("_v2", False)
            if use_v2:
                worker_builder_cls = "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"

                if "TORCH_CUDA_ARCH_LIST" not in os.environ:
                    warnings.warn(
                        "TORCH_CUDA_ARCH_LIST is not set. This is needed if using DeepEP in DTensorPolicyWorker V2. This variable is set in our container, but "
                        "if you are running a custom container or baremetal, you may need to set this variable manually. Example: export TORCH_CUDA_ARCH_LIST='9.0 10.0'"
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
            # The cluster has initialized a unified placemenet group across nodes
            # In this case, we need to create workers based on sorted bundle indices
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
                "max_tokens_per_microbatch": 0,  # Override this in each different call (presumably different sizes)
            }
            assert not config["sequence_packing"]["enabled"], (
                "Dynamic Batching is exclusive of Sequence Packing. Please disable Sequence Packing to use Dynamic Batching"
            )
        else:
            self.use_dynamic_batches = False

        # initialize FLOPs tracker
        try:
            self.flops_tracker = FLOPTracker.from_config(
                config["model_name"], get_default_hf_config(config["model_name"])
            )
        except ValueError as e:
            self.flops_tracker = None
            print(f"FLOPS tracker not supported for model {config['model_name']}: {e}")

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
        # this function should co-work with vllm, so we should wait for all futures to complete outside
        return futures

    def get_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a data dict.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict]
        unsorted_data_indices: list[int]

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
            # we just shard into DP shards here as Sequence packing allows for CP.
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

        futures = self.worker_group.run_all_workers_sharded_data(
            "get_logprobs",
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
        logprobs: BatchedDataDict[LogprobOutputSpec] = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )

        # dynamic batching sorts the inputs by sequence length to improve load balancing,
        # so change it back here
        if self.use_dynamic_batches or self.use_sequence_packing:
            logprobs.reorder_data(unsorted_data_indices)

        return logprobs

    def get_reference_policy_logprobs(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs of the reference policy for a data dict.

        Returns: Identical to get_logprobs.
        """
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict]
        unsorted_data_indices: list[int]
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

        futures = self.worker_group.run_all_workers_sharded_data(
            "get_reference_policy_logprobs",
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
            common_kwargs={"micro_batch_size": micro_batch_size},
        )
        logprobs: BatchedDataDict[ReferenceLogprobOutputSpec] = (
            BatchedDataDict.from_batches(
                self.worker_group.get_all_worker_results(futures)
            )
        )

        # dynamic batching sorts the inputs by sequence length to improve load balancing,
        # so change it back here
        if self.use_dynamic_batches or self.use_sequence_packing:
            logprobs.reorder_data(unsorted_data_indices)

        return logprobs

    def get_topk_logits(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[TopkLogitsOutputSpec]:
        """Dispatch get_topk_logits to workers (no CP/packed support initially)."""
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict]
        unsorted_data_indices: list[int]
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
            # we just shard into DP shards here as Sequence packing allows for CP.
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

        futures = self.worker_group.run_all_workers_sharded_data(
            "get_topk_logits",
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
            common_kwargs={"k": k, "micro_batch_size": micro_batch_size},
        )

        # Avoid BatchedDataDict.from_batches here because it flattens rows for tensors with ndim>2 ([B,S,k] -> [B,S*k]).
        worker_batches = self.worker_group.get_all_worker_results(futures)
        all_topk_logits = [wb["topk_logits"] for wb in worker_batches]
        all_topk_indices = [wb["topk_indices"] for wb in worker_batches]

        stacked: BatchedDataDict[TopkLogitsOutputSpec] = BatchedDataDict()
        stacked["topk_logits"] = torch.cat(all_topk_logits, dim=0)
        stacked["topk_indices"] = torch.cat(all_topk_indices, dim=0)

        if self.use_dynamic_batches or self.use_sequence_packing:
            stacked.reorder_data(unsorted_data_indices)

        return stacked

    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]
        # Shard and replicate the batch
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
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

        if self.flops_tracker is not None:
            self.flops_tracker.reset()
            for shard in sharded_data:
                input_lengths = shard["input_lengths"]
                self.flops_tracker.track_batch(input_lengths.tolist())

        # Train each shard in parallel
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
        if "moe_metrics" in results[0]:
            aggregated_results["moe_metrics"] = results[0]["moe_metrics"]

        if self.flops_tracker is not None:
            aggregated_results["total_flops"] = self.flops_tracker.total_flops
            aggregated_results["num_ranks"] = self.worker_group.cluster.world_size()
            gpus_per_worker = self.worker_group.cluster.world_size() / len(results)

            try:
                aggregated_results["theoretical_tflops"] = gpus_per_worker * sum(
                    get_theoretical_tflops(r["gpu_name"], r["model_dtype"])
                    for r in results
                )
            except Exception as e:
                warnings.warn(f"Error getting theoretical flops: {e}")

        # Aggregate metrics across all workers
        all_mb_metrics = defaultdict(list)
        for r in results:
            for k, v in r["all_mb_metrics"].items():
                all_mb_metrics[k].extend(v)
        aggregated_results["all_mb_metrics"] = dict(all_mb_metrics)

        return aggregated_results

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using the policy."""
        # Verify input data is right-padded
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
        result: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures),
            pad_value_dict={"output_ids": self.cfg["generation"]["_pad_token_id"]},
        )

        # Verify the output has all required fields
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
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict[ScoreOutputSpec]:
        """Score a batch of data using the policy."""
        # Verify input data is right-padded
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

        result: BatchedDataDict[ScoreOutputSpec] = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures),
        )
        required_keys = [
            "scores",
        ]
        missing_keys = [key for key in required_keys if key not in result]
        if missing_keys:
            raise ValueError(
                f"Missing required keys for ScoreOutputSpec: {missing_keys}"
            )

        return result

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        # We don't need to do anything here
        return True

    def prepare_for_training(self, *args: Any, **kwargs: Any) -> None:
        # onload everything to the GPU
        futures = self.worker_group.run_all_workers_single_data("prepare_for_training")
        ray.get(futures)

    def prepare_for_lp_inference(self, *args: Any, **kwargs: Any) -> None:
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_for_lp_inference"
        )
        ray.get(futures)

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        # We don't need to do anything here
        return True

    def invalidate_kv_cache(self, *args: Any, **kwargs: Any) -> bool:
        # We don't need to do anything here
        return True

    def prepare_refit_info(self) -> Optional[dict[str, Any]]:
        """Prepare the info for refit.

        Returns:
            dict: A dictionary containing the info for refit.
        """
        futures = self.worker_group.run_all_workers_single_data("prepare_refit_info")
        results = ray.get(futures)
        # Only get the first worker's info since all workers will have the same result
        return results[0]

    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        # Placeholder implementation
        pass

    def calibrate_qkv_fp8_scales(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """Trigger KV-cache FP8 scale calibration across Megatron workers and return results.

        Note: The backend `MegatronPolicyWorker.calibrate_qkv_fp8_scales` already implements
        distributed reduction, returning results merged across ranks. Therefore, we shard the
        input by DP and call in parallel, then take the result from the first worker.
        """
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        if self.use_dynamic_batches:
            self.dynamic_batching_args["max_tokens_per_microbatch"] = self.cfg[
                "dynamic_batching"
            ]["logprob_mb_tokens"]
            sharded_data, _ = data.shard_by_batch_size(  # type: ignore
                dp_size,
                batch_size=None,
                dynamic_batching_args=self.dynamic_batching_args,
            )
        elif self.use_sequence_packing:
            self.sequence_packing_args["max_tokens_per_microbatch"] = self.cfg[
                "sequence_packing"
            ]["logprob_mb_tokens"]
            sharded_data, _ = data.shard_by_batch_size(
                dp_size,
                batch_size=None,
                sequence_packing_args=self.sequence_packing_args,
            )
        else:
            sharded_data = data.shard_by_batch_size(  # type: ignore
                dp_size,
                batch_size=None,
            )

        futures = self.worker_group.run_all_workers_sharded_data(
            "calibrate_qkv_fp8_scales",
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
                "micro_batch_size": micro_batch_size,
                "percentile": percentile,
                "margin": margin,
                "include_q": include_q,
            },
        )
        results = self.worker_group.get_all_worker_results(futures)
        return results[0]

    def get_free_memory_bytes(self) -> int:
        """Get the available free memory."""
        futures = self.worker_group.run_all_workers_single_data("get_free_memory_bytes")
        # minimum free memory from all workers for safety
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

    def stream_weights_via_http(
        self, sglang_url_to_gpu_uuids: dict[str, list[str]]
    ) -> list[ray.ObjectRef]:
        """Send the weights to SGLang servers via HTTP API.

        Args:
            sglang_url_to_gpu_uuids: Dict mapping SGLang server URL to list of GPU UUIDs it uses
        """
        futures = self.worker_group.run_all_workers_single_data(
            "stream_weights_via_http",
            sglang_url_to_gpu_uuids=sglang_url_to_gpu_uuids,
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
        # this function should co-work with vllm, so we should wait for all futures to complete outside
        return futures

    def offload_before_refit(self) -> None:
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data("offload_before_refit")
        ray.get(futures)

    def offload_after_refit(self) -> None:
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data("offload_after_refit")
        ray.get(futures)

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        checkpointing_cfg: Optional[CheckpointingConfig] = None,
    ) -> None:
        """Save a checkpoint of the model."""
        # Only pass checkpointing_cfg for DTensor v2
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
                    "model_save_format must be None or omitted if using DTensorPolicyWorker (_v2=False)."
                )
            futures = self.worker_group.run_all_workers_single_data(
                "save_checkpoint",
                weights_path=weights_path,
                optimizer_path=optimizer_path,
                tokenizer_path=tokenizer_path,
            )
        ray.get(futures)

    def shutdown(self) -> bool:
        """Shut down all HF workers and clean up resources."""
        try:
            # Use the worker group's shutdown method with the worker's cleanup method
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during policy shutdown: {e}")
            return False

    def __del__(self) -> None:
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call worker_group.shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls worker_group.shutdown().
        """
        if hasattr(self, "worker_group"):
            self.worker_group.shutdown(cleanup_method="shutdown")

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        futures = self.worker_group.run_all_workers_single_data("start_gpu_profiling")
        ray.get(futures)

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        futures = self.worker_group.run_all_workers_single_data("stop_gpu_profiling")
        ray.get(futures)

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
