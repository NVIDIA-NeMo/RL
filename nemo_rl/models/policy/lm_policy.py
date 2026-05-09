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
from contextlib import nullcontext
from typing import Any, Optional, Union

import numpy as np
import ray
import torch
from ray.util.queue import Queue as RayQueue
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from transformers import AutoProcessor, PreTrainedTokenizerBase

from nemo_rl.algorithms.loss.interfaces import LossFunction
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
from nemo_rl.utils.timer import Timer

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
        worker_extension_cls_fqn: Optional[str] = None,
    ):
        if weights_path:
            weights_path = os.path.abspath(weights_path)
        if optimizer_path:
            optimizer_path = os.path.abspath(optimizer_path)

        worker_builder_cls_fqn: str
        tp_size = 1
        pp_size = 1
        cp_size = 1
        use_v2 = False

        megatron_enable = bool(config.get("megatron_cfg", {}).get("enabled", False))
        dtensor_enable = bool(config.get("dtensor_cfg", {}).get("enabled", False))
        draft_enabled = bool(config.get("draft", {}).get("enabled", False))
        if megatron_enable and dtensor_enable:
            raise ValueError(
                "Configure either Megatron (policy.megatron_cfg.enabled=true) or "
                "DTensor (policy.dtensor_cfg.enabled=true), not both."
            )
        if draft_enabled and not megatron_enable:
            raise ValueError(
                "policy.draft.enabled=true is only supported with the Megatron backend. "
                "Set policy.megatron_cfg.enabled=true or disable policy.draft."
            )
        if draft_enabled and bool(
            config.get("sequence_packing", {}).get("enabled", False)
        ):
            raise ValueError(
                "policy.draft.enabled=true does not support sequence packing yet. "
                "Disable policy.sequence_packing.enabled or policy.draft."
            )
        if megatron_enable:
            worker_builder_cls_fqn = "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
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
                worker_builder_cls_fqn = "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"

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
                worker_builder_cls_fqn = "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker"

            tp_size = config["dtensor_cfg"]["tensor_parallel_size"]
            cp_size = config["dtensor_cfg"]["context_parallel_size"]

            env_vars = config["dtensor_cfg"].get("env_vars", {})

        # If a worker extension class is provided, use it instead of the default worker builder class
        if worker_extension_cls_fqn is not None:
            print(
                f"Using worker extension class: {worker_extension_cls_fqn}, please make sure it is a subclass of {worker_builder_cls_fqn}."
            )
            worker_builder_cls_fqn = worker_extension_cls_fqn

        # Validate world_size compatibility with parallelism configuration
        model_parallel_size = pp_size * cp_size * tp_size
        actual_world_size = cluster.world_size()

        if (
            not bool(os.environ.get("NRL_IGNORE_TP_ACCURACY_CHECK"))
            and "logprob_batch_size" in config
            and tp_size >= 4
        ):
            sep_line = "\n" + ("-" * 80)
            assert config["train_micro_batch_size"] == config["logprob_batch_size"], (
                f"{sep_line}\n"
                "There is a known batch-variant accuracy issue with TP>=4 for both DTensor and Megatron backend.\n"
                "See https://docs.nvidia.com/nemo/rl/latest/guides/dtensor-tp-accuracy.html#root-cause for more details.\n"
                "\n"
                "Please choose either of the following solutions to avoid this issue:\n"
                "1. Set tp_size to 1 or 2. (tensor_parallel_size for DTensor, or tensor_model_parallel_size for Megatron)\n"
                "2. Set policy.train_micro_batch_size and policy.logprob_batch_size to be the same value.\n"
                "3. Set loss_fn.force_on_policy_ratio=true to force ratio=1.0, this requires train_global_batch_size == num_prompts_per_step * num_generations_per_prompt.\n"
                "4. Set NRL_IGNORE_TP_ACCURACY_CHECK=1 to bypass this check. (not recommended)"
                f"{sep_line}\n"
            )

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

        worker_kwargs = dict(
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
            init_reference_model=init_reference_model,
            worker_sharding_annotations=self.sharding_annotations,
            pre_init_communication_queue=pre_init_queue,
        )

        if use_v2:
            # DTensor v2 workers reconstruct tokenizer/processor locally to avoid
            # pickling across incompatible transformers versions (v4 head → v5 worker).
            config["tokenizer"]["use_processor"] = processor is not None
        else:
            worker_kwargs["tokenizer"] = tokenizer
            worker_kwargs["processor"] = processor

        worker_builder = RayWorkerBuilder(
            worker_builder_cls_fqn,
            config,
            **worker_kwargs,
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
            sequence_length_pad_multiple = config["make_sequence_length_divisible_by"]
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

        # RL-412 follow-up: cross-cluster weight-sync NCCL group runs in a
        # sibling Ray actor (RefitWorker) so a gen peer dying mid-broadcast
        # only poisons that actor's CUDA context, not the train workers'
        # Megatron NCCL groups. Lazily spawned on first ``init_collective``.
        self._refit_worker: Optional[Any] = None
        self._refit_worker_zmq_address: Optional[str] = None
        self._refit_worker_node_id: Optional[str] = None
        self._cross_cluster_world_size: Optional[int] = None
        # Gate the RefitWorker path on backend (megatron only by default —
        # the DTensor refit pipeline uses a different broadcast hook). The
        # ``NRL_USE_REFIT_WORKER`` env var lets us toggle off for debugging.
        self._use_refit_worker: bool = (
            megatron_enable
            and os.getenv("NRL_USE_REFIT_WORKER", "1") == "1"
        )

    def run_all_workers_single_data(self, method_name: str, *args, **kwargs) -> Any:
        """Run a method on all workers in parallel with the same data.

        Mainly used for worker extension classes.

        Args:
            method_name: The name of the method to run.
            *args: The positional arguments to pass to the method.
            **kwargs: The keyword arguments to pass to the method.

        Returns:
            The results of the method run on all workers.
        """
        futures = self.worker_group.run_all_workers_single_data(
            method_name, *args, **kwargs
        )
        results = ray.get(futures)
        return results

    def run_all_workers_multiple_data(self, method_name: str, *args, **kwargs) -> Any:
        """Run a method on all workers in parallel with different data.

        Mainly used for worker extension classes.

        Args:
            method_name: The name of the method to run.
            *args: The positional arguments to pass to the method.
            **kwargs: The keyword arguments to pass to the method.

        Returns:
            The results of the method run on all workers.
        """
        futures = self.worker_group.run_all_workers_multiple_data(
            method_name, *args, **kwargs
        )
        results = ray.get(futures)
        return results

    def _ensure_refit_worker(self, target_node_id: str, gpu_index: int) -> Any:
        """Ensure a RefitWorker actor is running on ``target_node_id``.

        - If no actor exists, spawn one with hard NodeAffinitySchedulingStrategy.
        - If one exists but on a different node, ``ray.kill`` it and respawn.
        - If one exists on the right node and is alive, reuse it.
        """
        from nemo_rl.models.policy.workers.refit_worker import RefitWorker

        if self._refit_worker is not None:
            # If pinned node hasn't changed, just verify liveness with a
            # short ping. If liveness check raises, fall through to respawn.
            if self._refit_worker_node_id == target_node_id:
                try:
                    ray.get(self._refit_worker.is_alive.remote(), timeout=10)
                    return self._refit_worker
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[lm_policy._ensure_refit_worker] existing actor "
                        f"liveness ping failed ({type(e).__name__}: {e}); "
                        f"respawning",
                        flush=True,
                    )
            # Wrong node or unhealthy — kill and rebuild.
            try:
                ray.kill(self._refit_worker)
            except Exception:  # noqa: BLE001
                pass
            self._refit_worker = None
            self._refit_worker_zmq_address = None
            self._refit_worker_node_id = None

        print(
            f"[lm_policy._ensure_refit_worker] spawning RefitWorker on "
            f"node_id={target_node_id} gpu_index={gpu_index}",
            flush=True,
        )
        # Pin the actor onto the same physical GPU as train rank 0 by
        # setting ``CUDA_VISIBLE_DEVICES`` in its runtime env BEFORE the
        # process imports torch. ``num_gpus=0`` (set in the @ray.remote
        # decorator) prevents Ray from carving out yet another GPU; the
        # env var below is what tells the CUDA runtime which physical
        # device to bind to.
        self._refit_worker = RefitWorker.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=target_node_id, soft=False
            ),
            runtime_env={
                "env_vars": {
                    # Pin to train rank 0's physical GPU.
                    "CUDA_VISIBLE_DEVICES": str(gpu_index),
                    # Tell Ray NOT to overwrite CUDA_VISIBLE_DEVICES
                    # to "" on this num_gpus=0 actor — without this,
                    # Ray's GPU manager wins and torch sees no GPUs.
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                }
            },
        ).remote(gpu_index=gpu_index)
        # Verify placement landed where we asked. Raises on placement failure.
        actual_node = ray.get(
            self._refit_worker.get_node_id.remote(), timeout=60
        )
        if actual_node != target_node_id:
            raise RuntimeError(
                f"RefitWorker placement landed on {actual_node!r}, expected "
                f"{target_node_id!r}; hard NodeAffinity should have prevented this"
            )
        self._refit_worker_node_id = actual_node
        # Bring up the ZMQ REP socket and remember its address — the
        # train-side rank 0 worker will REQ-connect to it during
        # ``broadcast_weights_for_collective``.
        self._refit_worker_zmq_address = ray.get(
            self._refit_worker.start_zmq_server.remote(), timeout=60
        )
        print(
            f"[lm_policy._ensure_refit_worker] RefitWorker ready, "
            f"zmq={self._refit_worker_zmq_address}",
            flush=True,
        )
        return self._refit_worker

    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        """Initialize the cross-cluster weight-sync collective.

        Under the RefitWorker architecture (default for the Megatron
        backend), the cross-cluster group is owned by a sibling Ray
        actor on the same node + same physical GPU as train rank 0.
        Train workers do not participate. The new world layout is
        ``1 + gen_world_size`` (RefitWorker + N gen ranks).

        ``world_size`` here is the legacy total (train + gen) preserved
        for the gen-side caller's signature; we recompute the actual
        cross-cluster world internally.
        """
        if not self._use_refit_worker:
            # Legacy path: every train rank participates in the
            # cross-cluster group. Kept for the DTensor backend.
            futures = self.worker_group.run_all_workers_single_data(
                "init_collective",
                ip=ip,
                port=port,
                world_size=world_size,
                train_world_size=train_world_size,
            )
            return futures

        # New cross-cluster world is just RefitWorker (rank 0) + gen ranks.
        gen_world_size = world_size - train_world_size
        cross_cluster_world = 1 + gen_world_size
        self._cross_cluster_world_size = cross_cluster_world

        # Locate train rank 0 (Ray node id + physical GPU index) so the
        # RefitWorker actor lands on the same node and shares its GPU.
        rank0 = self.worker_group.workers[0]
        target_node_id = ray.get(rank0.get_node_id.remote(), timeout=30)
        # ``report_node_ip_and_gpu_id`` returns (ip, gpu_id). The gpu_id
        # is what Ray's ``get_gpu_ids()`` returned (may be int or str
        # depending on Ray version) — coerce to int for the env var.
        rank0_gpu_info = ray.get(
            rank0.report_node_ip_and_gpu_id.remote(), timeout=30
        )
        gpu_index = int(rank0_gpu_info[1])

        self._ensure_refit_worker(target_node_id=target_node_id, gpu_index=gpu_index)

        # Train workers do NOT participate in the cross-cluster group
        # under the RefitWorker architecture — skip dispatching the
        # base ``init_collective`` to them. Their CUDA contexts stay
        # fully insulated from cross-cluster failures.

        # Return a single future representing the RefitWorker's NCCL
        # rendezvous with the gen ranks. ``ensure_collective_synced`` on
        # the caller side will ``ray.wait`` on this together with the
        # gen-side futures and surface failure.
        future = self._refit_worker.init_collective.remote(
            ip, port, cross_cluster_world, 0
        )
        return [future]

    def abort_collective(self) -> list[ray.ObjectRef]:
        """Abort the cross-cluster weight-sync NCCL group.

        Under the RefitWorker architecture, "abort" means ``ray.kill`` the
        sibling actor — its CUDA context dies with it (along with any
        queued NCCL/CUDA error from a dead peer), and the next
        ``init_collective`` spawns a fresh actor on a clean context. We
        return ``[]`` because there are no per-worker futures to await:
        the kill is best-effort-immediate.

        Legacy path (DTensor backend): dispatch ``abort_collective`` on
        every train worker.
        """
        if not self._use_refit_worker:
            return self.worker_group.run_all_workers_single_data("abort_collective")

        if self._refit_worker is not None:
            print(
                "[lm_policy.abort_collective] ray.kill RefitWorker (poisoned "
                "context dies with the process)",
                flush=True,
            )
            stale_handle = self._refit_worker
            try:
                ray.kill(stale_handle)
            except Exception as e:  # noqa: BLE001
                print(
                    f"[lm_policy.abort_collective] ray.kill raised "
                    f"{type(e).__name__}: {e}",
                    flush=True,
                )
            # ``ray.kill`` is asynchronous: it requests termination and
            # returns immediately, but the actor process may still be
            # winding down (and crucially, still HOLDING its ZMQ
            # ipc:///tmp/refit-worker-<pid>.sock and its bound TCPStore
            # port) for a few seconds. If the next ``init_collective``
            # races ahead and asks Ray to spawn a fresh RefitWorker
            # before the old one's resources are released, the new
            # actor's ``start_zmq_server`` can fail to bind (different
            # PID → different socket path so usually fine) but more
            # importantly, the NCCL TCPStore's master_address:port comes
            # from the train cluster (same machine across respawns) and
            # the dead actor may still be sitting on a half-open NCCL
            # socket. Probing the actor with a bounded ``is_alive`` ping
            # confirms the kill propagated; we swallow the expected
            # RayActorError here.
            try:
                ray.get(stale_handle.is_alive.remote(), timeout=10)
                print(
                    "[lm_policy.abort_collective] WARNING: stale RefitWorker "
                    "still responding to is_alive after ray.kill — "
                    "Ray scheduler hasn't propagated the kill yet",
                    flush=True,
                )
            except Exception:  # noqa: BLE001 — RayActorError on success
                pass
            self._refit_worker = None
            self._refit_worker_zmq_address = None
            self._refit_worker_node_id = None
            self._cross_cluster_world_size = None

        # Train workers don't hold a model_update_group under RefitWorker
        # mode — return [] so the caller's ``ray.wait`` is a no-op.
        return []

    def get_logprobs(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        timer: Optional[Timer] = None,
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

        with timer.time("get_logprobs/shard_data") if timer else nullcontext():
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

        with (
            timer.time("get_logprobs/submit_logprob_futures")
            if timer
            else nullcontext()
        ):
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
        timer: Optional[Timer] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs of the reference policy for a data dict.

        Returns: Identical to get_logprobs.
        """
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict]
        unsorted_data_indices: list[int]
        with (
            timer.time("get_reference_policy_logprobs/shard_data")
            if timer
            else nullcontext()
        ):
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
            timer.time(
                "get_reference_policy_logprobs/submit_reference_policy_logprob_futures"
            )
            if timer
            else nullcontext()
        ):
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
        timer: Optional[Timer] = None,
    ) -> BatchedDataDict[TopkLogitsOutputSpec]:
        """Dispatch get_topk_logits to workers (no CP/packed support initially)."""
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        sharded_data: list[SlicedDataDict]
        unsorted_data_indices: list[int]
        with timer.time("get_topk_logits/shard_data") if timer else nullcontext():
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

        with (
            timer.time("get_topk_logits/submit_topk_logits_futures")
            if timer
            else nullcontext()
        ):
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
        timer: Optional[Timer] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        batch_size = gbs or self.cfg["train_global_batch_size"]
        micro_batch_size = mbs or self.cfg["train_micro_batch_size"]
        # Shard and replicate the batch
        dp_size = self.sharding_annotations.get_axis_size("data_parallel")
        with timer.time("policy_training/sharding_data") if timer else nullcontext():
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
        with (
            timer.time("policy_training/submit_training_futures")
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
        """Broadcast the weights for collective communication.

        Under the RefitWorker architecture: dispatch the per-rank-0
        ``stream_weights_to_refit_worker`` method on every worker (only
        rank 0 actually pushes — see worker-side fast path), and in
        parallel dispatch the RefitWorker's ``broadcast_weights_until_complete``
        which receives via ZMQ and forwards to gen ranks via NCCL. The
        caller (``grpo.refit_policy_generation``) ``ray.get``s these
        together with the gen-side futures.
        """
        if not self._use_refit_worker:
            futures = self.worker_group.run_all_workers_single_data(
                "broadcast_weights_for_collective",
                kv_scales=kv_scales,
            )
            return futures

        if self._refit_worker is None or self._refit_worker_zmq_address is None:
            raise RuntimeError(
                "broadcast_weights_for_collective called before "
                "init_collective spawned the RefitWorker"
            )

        # Pre-flight: make sure the RefitWorker is actually alive BEFORE we
        # dispatch the train-side senders. Otherwise the senders connect to
        # a stale ZMQ socket that nobody is reading from and wedge for the
        # full ZMQ recv timeout (10 min). Cheap ping (sub-ms when healthy);
        # raises if the actor is dead so the caller can ``abort_collective``
        # + ``ensure_collective_synced`` again with a fresh actor.
        try:
            ray.get(self._refit_worker.is_alive.remote(), timeout=10)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "RefitWorker liveness ping failed before "
                f"broadcast_weights_for_collective ({type(e).__name__}: {e}) — "
                "actor likely died between init_collective and broadcast; "
                "caller must abort_collective + re-run ensure_collective_synced"
            ) from e

        # Sender: only rank 0 actually streams; other ranks no-op via the
        # worker-side fast path (see ``MegatronPolicyWorker.broadcast_weights_for_collective``).
        # We still dispatch on all workers so any subclass bookkeeping
        # runs symmetrically and so ``run_all_workers_single_data``'s
        # contract holds.
        sender_futures = self.worker_group.run_all_workers_single_data(
            "broadcast_weights_for_collective",
            kv_scales=kv_scales,
            refit_worker_zmq_address=self._refit_worker_zmq_address,
        )
        # Receiver / forwarder: kick off the RefitWorker's loop. It
        # returns when the sender posts COMPLETE.
        receiver_future = self._refit_worker.broadcast_weights_until_complete.remote(
            kv_scales=kv_scales
        )
        return list(sender_futures) + [receiver_future]

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
        # Best-effort kill the sibling RefitWorker so its placement group
        # is freed before we tear down the train worker group.
        if self._refit_worker is not None:
            try:
                ray.kill(self._refit_worker)
            except Exception as e:  # noqa: BLE001
                print(f"Error killing RefitWorker during shutdown: {e}")
            self._refit_worker = None
            self._refit_worker_zmq_address = None
            self._refit_worker_node_id = None
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
