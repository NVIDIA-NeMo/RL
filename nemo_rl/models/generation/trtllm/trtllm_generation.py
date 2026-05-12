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

"""``GenerationInterface`` implementation backed by TRT-LLM.

Non-colocated: separate train / inference GPU sets, NCCL broadcast for
weight sync. Colocated: shares GPUs with the policy and uses sleep/wakeup
to time-multiplex GPU memory between training and inference phases.
"""

import asyncio
import os
from typing import Any, AsyncGenerator, Optional, Union

import numpy as np
import ray

from nemo_rl.distributed.batched_data_dict import BatchedDataDict, SlicedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.generation.trtllm.config import TrtllmConfig


class TrtllmGeneration(GenerationInterface):
    """TRT-LLM generation backend (colocated requires async_engine=true)."""

    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: TrtllmConfig,
        name_prefix: str = "trtllm_policy",
        workers_per_node: Optional[Union[int, list[int]]] = None,
    ):
        self.cfg = config
        self.tp_size = self.cfg["trtllm_cfg"]["tensor_parallel_size"]
        self.model_parallel_size = self.tp_size

        assert cluster.world_size() % self.model_parallel_size == 0, (
            f"Cluster world_size ({cluster.world_size()}) must be divisible by "
            f"TP size ({self.model_parallel_size})."
        )
        self.dp_size = cluster.world_size() // self.model_parallel_size

        # MoE: TRT-LLM partitions TP on MoE layers into moe_tp × moe_ep, so
        # the product must equal the main tensor_parallel_size. Validate here
        # to fail fast — the LLM constructor would otherwise raise a less
        # actionable error deep inside the engine.
        moe_tp = self.cfg["trtllm_cfg"].get("moe_tensor_parallel_size")
        moe_ep = self.cfg["trtllm_cfg"].get("moe_expert_parallel_size")
        if moe_tp is not None or moe_ep is not None:
            moe_tp_v = moe_tp if moe_tp is not None else 1
            moe_ep_v = moe_ep if moe_ep is not None else 1
            assert moe_tp_v * moe_ep_v == self.tp_size, (
                f"moe_tensor_parallel_size ({moe_tp_v}) * moe_expert_parallel_size "
                f"({moe_ep_v}) must equal tensor_parallel_size ({self.tp_size})."
            )

        missing_keys = [k for k in TrtllmConfig.__required_keys__ if k not in self.cfg]
        if "model_name" not in self.cfg:
            missing_keys.append("model_name")
        assert not missing_keys, (
            f"TrtllmConfig missing keys: {missing_keys}"
        )

        self.sharding_annotations = NamedSharding(
            layout=np.arange(cluster.world_size()).reshape(
                self.dp_size, self.tp_size,
            ),
            names=["data_parallel", "tensor_parallel"],
        )

        self.colocated_enabled = bool(
            self.cfg.get("colocated", {}).get("enabled", False)
        )
        self.async_engine = self.cfg["trtllm_cfg"].get("async_engine", False)
        if self.colocated_enabled and not self.async_engine:
            raise NotImplementedError(
                "TRT-LLM colocated mode requires trtllm_cfg.async_engine=true "
                "(sleep/wakeup is only wired through the async worker)."
            )

        # Colocated reuses the policy's placement group → no PACK rearrangement.
        # Non-colocated uses PACK so inference workers cluster together rather
        # than interleaving with training bundles (SPREAD would emit something
        # like [[0,3,6],[1,4,7],[2,5]] across nodes, breaking TP placement).
        strategy = None if self.colocated_enabled else "PACK"
        # Cross-node TP needs a single unified placement group so workers can
        # land on bundles spanning node boundaries (e.g. TP=8 on 2x4 GPUs).
        needs_cross_node_parallelism = (
            self.model_parallel_size > cluster.num_gpus_per_node
        )
        cluster._init_placement_groups(
            strategy=strategy,
            use_unified_pg=needs_cross_node_parallelism,
        )

        if self.async_engine:
            worker_cls = "nemo_rl.models.generation.trtllm.trtllm_worker_async.TrtllmAsyncGenerationWorker"
        else:
            worker_cls = "nemo_rl.models.generation.trtllm.trtllm_worker.TrtllmGenerationWorker"
        worker_builder = RayWorkerBuilder(worker_cls, config)

        # NCCL_CUMEM_ENABLE=1 is needed for the non-colocated NCCL collective
        # broadcast; colocated shares the policy's NCCL group so don't touch it.
        env_vars: dict[str, str] = {}
        if not self.colocated_enabled:
            env_vars["NCCL_CUMEM_ENABLE"] = "1"

        if self.model_parallel_size > 1:
            node_bundle_indices = self._get_tied_worker_bundle_indices(cluster)
            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                bundle_indices_list=node_bundle_indices,
                sharding_annotations=self.sharding_annotations,
                env_vars=env_vars,
            )
        else:
            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                workers_per_node=workers_per_node,
                sharding_annotations=self.sharding_annotations,
                env_vars=env_vars,
            )

        # post-init on workers (starts HTTP server when expose_http_server=true,
        # finishes async engine setup for the async worker variant).
        post_init_method = "post_init_async" if self.async_engine else "post_init"
        futures = self.worker_group.run_all_workers_single_data(
            post_init_method,
            run_rank_0_only_axes=["tensor_parallel"],
        )
        ray.get(futures)

        # Round-robin DP shard used by generate_async for per-sample dispatch.
        self.current_generate_dp_shard_idx = 0

        self.dp_openai_server_base_urls = self._report_dp_openai_server_base_urls()

        self.device_uuids = self._report_device_id()

        assert self.dp_size == self.worker_group.dp_size, (
            f"DP size mismatch: expected {self.dp_size}, got {self.worker_group.dp_size}"
        )

    # ------------------------------------------------------------------ #
    #  Placement helpers (simplified from VllmGeneration)
    # ------------------------------------------------------------------ #

    def _get_tied_worker_bundle_indices(
        self, cluster: RayVirtualCluster,
    ) -> list[tuple[int, list[int]]]:
        placement_groups = cluster.get_placement_groups()
        if not placement_groups:
            raise ValueError("No placement groups in cluster")

        tied_groups: list[tuple[int, list[int]]] = []
        for pg_idx, pg in enumerate(placement_groups):
            if pg.bundle_count == 0:
                continue
            n_groups = pg.bundle_count // self.model_parallel_size
            for g in range(n_groups):
                start = g * self.model_parallel_size
                tied_groups.append((pg_idx, list(range(start, start + self.model_parallel_size))))

        if not tied_groups:
            raise ValueError("Cannot allocate worker groups with available resources")
        return tied_groups

    def _report_device_id(self) -> list[list[str]]:
        futures = self.worker_group.run_all_workers_single_data(
            "report_device_id_async" if self.async_engine else "report_device_id",
            run_rank_0_only_axes=["tensor_parallel"],
        )
        return ray.get(futures)

    def _report_dp_openai_server_base_urls(self) -> list[Optional[str]]:
        """Collect HTTP server base URLs from each DP-rank-0 worker."""
        if not self.cfg["trtllm_cfg"].get("expose_http_server"):
            return [None] * self.dp_size
        futures = self.worker_group.run_all_workers_single_data(
            "report_dp_openai_server_base_url",
            run_rank_0_only_axes=["tensor_parallel"],
        )
        return ray.get(futures)

    # ------------------------------------------------------------------ #
    #  GenerationInterface
    # ------------------------------------------------------------------ #

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int,
    ) -> list[ray.ObjectRef]:
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group not initialised")

        total_workers = len(self.worker_group.workers)
        workers_per_group = total_workers // self.dp_size
        rank_prefix_list = list(range(0, total_workers, workers_per_group))

        return self.worker_group.run_all_workers_multiple_data(
            "init_collective_async" if self.async_engine else "init_collective",
            rank_prefix=rank_prefix_list,
            run_rank_0_only_axes=["tensor_parallel"],
            common_kwargs={
                "ip": ip,
                "port": port,
                "world_size": world_size,
                "train_world_size": train_world_size,
            },
        )

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        assert isinstance(data, BatchedDataDict)
        assert "input_ids" in data and "input_lengths" in data

        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict] = data.shard_by_batch_size(
            dp_size, allow_uneven_shards=True,
        )
        future_bundle = self.worker_group.run_all_workers_sharded_data(
            "generate_async" if self.async_engine else "generate",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=None,
            output_is_replicated=None,
            common_kwargs={"greedy": greedy},
        )
        results = self.worker_group.get_all_worker_results(future_bundle)

        combined: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            results, pad_value_dict={"output_ids": self.cfg["_pad_token_id"]},
        )

        required = ["output_ids", "generation_lengths", "unpadded_sequence_lengths", "logprobs"]
        missing = [k for k in required if k not in combined]
        if missing:
            raise ValueError(f"Missing generation output keys: {missing}")
        return combined

    async def generate_async(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Yield a single-sample generation result.

        Called by run_async_multi_turn_rollout, which dispatches one sample at
        a time per coroutine. The async worker's max_concurrency lets multiple
        in-flight Ray calls share the same AsyncLLM, which batches them
        internally via asyncio.gather.
        """
        if not self.async_engine:
            raise RuntimeError(
                "generate_async requires trtllm_cfg.async_engine=true."
            )

        if "input_ids" not in data or "input_lengths" not in data:
            raise AssertionError(
                "input_ids and input_lengths are required in data for generate_async"
            )
        if len(data["input_ids"]) == 0:
            return
        assert data.size == 1, (
            f"generate_async expects single-sample data, got batch_size={data.size}."
        )

        leader_worker_idx = self.worker_group.get_dp_leader_worker_idx(
            self.current_generate_dp_shard_idx
        )
        worker_result_ref = self.worker_group.run_single_worker_single_data(
            method_name="generate_async",
            worker_idx=leader_worker_idx,
            data=data,
            greedy=greedy,
        )
        self.current_generate_dp_shard_idx = (
            self.current_generate_dp_shard_idx + 1
        ) % self.worker_group.dp_size

        timeout_seconds = float(
            os.environ.get("NRL_TRTLLM_ASYNC_TIMEOUT_SECONDS", "900")
        )
        try:
            result = await asyncio.wait_for(
                worker_result_ref, timeout=timeout_seconds
            )
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"TRT-LLM async generation timed out after {timeout_seconds}s. "
                f"Tune with NRL_TRTLLM_ASYNC_TIMEOUT_SECONDS."
            )

        result["gen_leader_worker_idx"] = [int(leader_worker_idx)]
        # Worker.generate_async returns a single-sample BatchedDataDict; idx in
        # the input batch is always 0 (caller already split per-sample).
        yield (0, result)

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Wake inference workers up. No-op for non-colocated."""
        if not self.colocated_enabled:
            return True
        try:
            futures = self.worker_group.run_all_workers_single_data(
                "wake_up_async",
                run_rank_0_only_axes=["tensor_parallel"],
                **kwargs,
            )
            results = ray.get(futures)
            return all(r for r in results if r is not None)
        except Exception as e:
            print(f"Error in prepare_for_generation: {e}")
            return False

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Sleep workers (colocated) or reset prefix cache (non-colocated)."""
        try:
            if self.colocated_enabled:
                method_name = "sleep_async"
            else:
                method_name = (
                    "reset_prefix_cache_async" if self.async_engine
                    else "reset_prefix_cache"
                )
            futures = self.worker_group.run_all_workers_single_data(
                method_name,
                run_rank_0_only_axes=["tensor_parallel"],
            )
            results = ray.get(futures)
            return all(r for r in results if r is not None)
        except Exception as e:
            print(f"Error in finish_generation: {e}")
            return False

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        futures = self.worker_group.run_all_workers_single_data(
            "prepare_refit_info_async" if self.async_engine else "prepare_refit_info",
            state_dict_info=state_dict_info,
            run_rank_0_only_axes=["tensor_parallel"],
        )
        ray.get(futures)

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group not initialised")
        return self.worker_group.run_all_workers_single_data(
            "update_weights_from_collective_async" if self.async_engine
            else "update_weights_from_collective",
            run_rank_0_only_axes=["tensor_parallel"],
        )

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        """Receive weights via CUDA-IPC + ZMQ (colocated mode)."""
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group not initialised")
        return self.worker_group.run_all_workers_single_data(
            "update_weights_via_ipc_zmq_async" if self.async_engine
            else "update_weights_via_ipc_zmq",
            run_rank_0_only_axes=["tensor_parallel"],
        )

    def invalidate_kv_cache(self) -> bool:
        try:
            futures = self.worker_group.run_all_workers_single_data(
                "reset_prefix_cache_async" if self.async_engine else "reset_prefix_cache",
                run_rank_0_only_axes=["tensor_parallel"],
            )
            results = ray.get(futures)
            return all(r for r in results if r is not None)
        except Exception as e:
            print(f"Error invalidating TRT-LLM caches: {e}")
            return False

    def shutdown(self) -> bool:
        try:
            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during TRT-LLM shutdown: {e}")
            return False

    def __del__(self) -> None:
        self.shutdown()
