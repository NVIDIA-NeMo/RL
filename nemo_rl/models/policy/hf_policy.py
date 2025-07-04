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
from typing import List, Optional, Union

import ray
from transformers import AutoTokenizer

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict, DynamicBatchingCfg
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.interfaces import PolicyInterface
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.dtensor_policy_worker import DTensorPolicyWorker
from nemo_rl.models.policy.fsdp1_policy_worker import FSDP1PolicyWorker


class HfPolicy(PolicyInterface, GenerationInterface):
    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: PolicyConfig,
        tokenizer: AutoTokenizer,
        name_prefix: str = "hf_policy",
        workers_per_node: Optional[Union[int, List[int]]] = None,
        init_optimizer: bool = True,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_reference_model: bool = True,
    ):
        if weights_path:
            weights_path = os.path.abspath(weights_path)
        if optimizer_path:
            optimizer_path = os.path.abspath(optimizer_path)

        node_bundle_indices = None
        self.tensor_parallel_size = 1

        if config["dtensor_cfg"]["enabled"]:
            worker_builder_cls = DTensorPolicyWorker
            self.tensor_parallel_size = config["dtensor_cfg"]["tensor_parallel_size"]
            node_bundle_indices = self._get_tied_worker_bundle_indices(cluster)
        else:
            worker_builder_cls = FSDP1PolicyWorker

        worker_builder = RayWorkerBuilder(
            worker_builder_cls,
            config,
            tokenizer=tokenizer,
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_reference_model=init_reference_model,
        )

        self.worker_group = RayWorkerGroup(
            cluster,
            worker_builder,
            name_prefix=name_prefix,
            bundle_indices_list=node_bundle_indices,
        )

        if config["dynamic_batching"]["enabled"]:
            assert config["dtensor_cfg"]["enabled"], (
                "Dynamic batch is only supported for DTensor policy."
            )
            self.use_dynamic_batches = True
            self.dynamic_batching_cfg: DynamicBatchingCfg = {
                "input_key": "input_ids",
                "input_lengths_key": "input_lengths",
                "sequence_length_round": config["dynamic_batching"][
                    "sequence_length_round"
                ],
            }
        else:
            self.use_dynamic_batches = False

        self.dp_size = self.worker_group.world_size // self.tensor_parallel_size
        self.cfg = config

    def _get_tied_worker_bundle_indices(self, cluster):
        """Calculate bundle indices for tensor parallel workers."""
        # Get the placement groups (nodes) from the cluster
        placement_groups = cluster.get_placement_groups()

        tied_worker_groups = []

        # For each node (placement group), create tied worker groups of size tensor_parallel_size
        for node_idx, pg in enumerate(placement_groups):
            # How many bundles (GPUs) are on this node
            bundles_on_node = pg.bundle_count
            tied_worker_groups_on_node = bundles_on_node // self.tensor_parallel_size

            if tied_worker_groups_on_node > 0:
                for group_idx in range(tied_worker_groups_on_node):
                    # Local bundle indices for this tied worker group (consecutive GPUs on this node)
                    start_idx = group_idx * self.tensor_parallel_size
                    end_idx = start_idx + self.tensor_parallel_size
                    local_bundle_indices = list(range(start_idx, end_idx))
                    tied_worker_groups.append((node_idx, local_bundle_indices))

        if not tied_worker_groups:
            raise ValueError(
                f"Cannot create any tensor parallel tied worker groups with size {self.tensor_parallel_size}. "
                f"Make sure each node has at least {self.tensor_parallel_size} GPUs."
            )

        return tied_worker_groups

    def get_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict:
        """Get the logprobs of the model for a data dict.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        if self.use_dynamic_batches:
            self.dynamic_batching_cfg["max_tokens_per_microbatch"] = self.cfg[
                "dynamic_batching"
            ]["logprob_mb_tokens"]
            sharded_data, unsorted_data_indices = data.shard_by_batch_size(
                self.dp_size,
                batch_size=None,
                dynamic_batching_cfg=self.dynamic_batching_cfg,
            )
        else:
            sharded_data = data.shard_by_batch_size(
                self.dp_size,
                batch_size=None,
            )

        futures = self.worker_group.run_all_workers_multiple_data(
            "get_logprobs", sharded_data, only_on="all_tied_workers"
        )
        logprobs = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )

        # dynamic batching sorts the inputs by sequence length to improve load balancing,
        # so change it back here
        if self.use_dynamic_batches:
            logprobs.reorder_data(unsorted_data_indices)

        return logprobs

    def get_reference_policy_logprobs(
        self, data: BatchedDataDict[GenerationDatumSpec], micro_batch_size: int = None
    ) -> BatchedDataDict:
        """Get the logprobs of the reference policy for a data dict.

        Returns: Identical to get_logprobs.
        """
        if self.use_dynamic_batches:
            self.dynamic_batching_cfg["max_tokens_per_microbatch"] = self.cfg[
                "dynamic_batching"
            ]["logprob_mb_tokens"]
            sharded_data, unsorted_data_indices = data.shard_by_batch_size(
                self.dp_size,
                batch_size=None,
                dynamic_batching_cfg=self.dynamic_batching_cfg,
            )
        else:
            sharded_data = data.shard_by_batch_size(
                self.dp_size,
                batch_size=None,
            )

        futures = self.worker_group.run_all_workers_multiple_data(
            "get_reference_policy_logprobs",
            sharded_data,
            common_kwargs={"micro_batch_size": micro_batch_size},
            only_on="all_tied_workers",
        )

        logprobs = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )

        # dynamic batching sorts the inputs by sequence length to improve load balancing,
        # so change it back here
        if self.use_dynamic_batches:
            logprobs.reorder_data(unsorted_data_indices)

        return logprobs

    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ):
        """Train the policy on a batch of data with a given loss function."""
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]
        # Shard and replicate the batch
        if self.use_dynamic_batches:
            self.dynamic_batching_cfg["max_tokens_per_microbatch"] = self.cfg[
                "dynamic_batching"
            ]["train_mb_tokens"]
            sharded_data, _ = data.shard_by_batch_size(
                self.dp_size,
                batch_size=batch_size,
                dynamic_batching_cfg=self.dynamic_batching_cfg,
            )
        else:
            sharded_data = data.shard_by_batch_size(
                self.dp_size,
                batch_size=batch_size,
            )

        # Train each shard in parallel
        futures = self.worker_group.run_all_workers_multiple_data(
            "train",
            sharded_data,
            common_kwargs={
                "loss_fn": loss_fn,
                "eval_mode": eval_mode,
                "gbs": batch_size,
                "mbs": micro_batch_size,
            },
            only_on="all_tied_workers",
        )
        results = self.worker_group.get_all_worker_results(futures)
        return results[0]

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

        sharded_data = data.shard_by_batch_size(self.dp_size, batch_size=None)
        futures = self.worker_group.run_all_workers_multiple_data(
            "generate",
            sharded_data,
            common_kwargs={"greedy": greedy},
            only_on="all_tied_workers",
        )
        result = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures),
            pad_value_dict={"output_ids": self.cfg["generation"]["pad_token_id"]},
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

    def prepare_for_generation(self, *args, **kwargs):
        # We don't need to do anything here
        pass

    def prepare_for_training(self, *args, **kwargs):
        # onload everything to the GPU
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_for_training", only_on="all_tied_workers"
        )
        ray.get(futures)

    def prepare_for_lp_inference(self, *args, **kwargs):
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_for_lp_inference", only_on="all_tied_workers"
        )
        ray.get(futures)

    def finish_generation(self, *args, **kwargs):
        # We don't need to do anything here
        pass

    def finish_training(self, *args, **kwargs):
        # Placeholder implementation
        pass

    def prepare_weights_for_ipc(self):
        """Prepare the weights for IPC.

        Returns:
            dict: A dictionary containing the state_dict_info of the model.
        """
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_weights_for_ipc", only_on="all_tied_workers"
        )
        # only get the first worker's result is enough since all workers will have the same result
        return ray.get(futures)[0]

    def get_weights_ipc_handles(self, key):
        """Fetch weight IPC handles from all workers.

        Returns:
            dict: A dictionary mapping device UUIDs to parameter IPC handles.
        """
        # Collect IPC handles from all workers
        worker_handles = ray.get(
            [
                worker.get_weights_ipc_handles.remote(key)
                for worker in self.worker_group.workers
            ]
        )

        # Combine all worker handles into a single dictionary
        all_handles = {}
        for handle in worker_handles:
            all_handles.update(handle)

        return all_handles

    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data(
            "offload_before_refit", only_on="all_tied_workers"
        )
        ray.get(futures)

    def offload_after_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        futures = self.worker_group.run_all_workers_single_data(
            "offload_after_refit", only_on="all_tied_workers"
        )
        ray.get(futures)

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ):
        """Save a checkpoint of the model."""
        futures = self.worker_group.run_all_workers_single_data(
            "save_checkpoint",
            weights_path,
            optimizer_path,
            tokenizer_path,
            only_on="all_tied_workers",
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

    def __del__(self):
        """Shuts down the worker groups when the object is deleted or is garbage collected.

        This is an extra safety net in case the user forgets to call worker_group.shutdown() and the pointer to
        the object is lost due to leaving a function scope. It's always recommended that the
        user calls worker_group.shutdown().
        """
        self.worker_group.shutdown()

    def get_entropy(
        self, data: BatchedDataDict[GenerationDatumSpec]
    ) -> BatchedDataDict:
        """Get the logprobs of the model for a data dict.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        if self.use_dynamic_batches:
            self.dynamic_batching_cfg["max_tokens_per_microbatch"] = self.cfg[
                "dynamic_batching"
            ]["logprob_mb_tokens"]
            sharded_data, unsorted_data_indices = data.shard_by_batch_size(
                self.dp_size,
                batch_size=None,
                dynamic_batching_cfg=self.dynamic_batching_cfg,
            )
        else:
            sharded_data = data.shard_by_batch_size(
                self.dp_size,
                batch_size=None,
            )

        futures = self.worker_group.run_all_workers_multiple_data(
            "get_entropy", sharded_data, only_on="all_tied_workers"
        )
        entropy = BatchedDataDict.from_batches(
            self.worker_group.get_all_worker_results(futures)
        )

        # dynamic batching sorts the inputs by sequence length to improve load balancing,
        # so change it back here
        if self.use_dynamic_batches:
            entropy.reorder_data(unsorted_data_indices)

        return entropy
