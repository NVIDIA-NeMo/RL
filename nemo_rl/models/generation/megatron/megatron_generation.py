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

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, AsyncGenerator, Optional, cast

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoProcessor
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.held_port import RemoteHeldPortReservation
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
    RefitPayloadMode,
    reject_unenforceable_refit_deadline,
)
from nemo_rl.models.generation.megatron.config import (
    MCoreGenerationConfig,
    dedicated_inference_megatron_cfg,
    merged_inference_megatron_cfg,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

if TYPE_CHECKING:
    from nemo_rl.algorithms.single_controller_utils.config import MasterConfig
    from nemo_rl.distributed.worker_groups import RayWorkerGroup
    from nemo_rl.models.policy.lm_policy import Policy
    from nemo_rl.weight_sync.membership import RefitMembership


class MegatronGeneration(GenerationInterface):
    """Generation interface backed by Megatron (colocated or non-colocated)."""

    @staticmethod
    def effective_megatron_cfg(config: PolicyConfig) -> dict[str, Any]:
        """The megatron_cfg the generation workers actually run with.

        Colocated generation shares the training model, so the training
        values apply; non-colocated builds a dedicated policy with
        mcore_generation_config merged on top. Always returns a fresh dict.
        """
        if config["generation"]["colocated"]["enabled"]:
            return dict(config["megatron_cfg"])
        return merged_inference_megatron_cfg(config)

    @classmethod
    def nvlink_domain_span(cls, config: PolicyConfig) -> int:
        """Largest GPU group requiring full NVLink connectivity.

        Colocated reshard hosts a second, inference-layout model on the same ranks.
        """
        layouts = [cls.effective_megatron_cfg(config)]
        if config["generation"]["colocated"]["enabled"]:
            inference_mcfg = dedicated_inference_megatron_cfg(config)
            if inference_mcfg is not None:
                layouts.append(inference_mcfg)
        return max(
            max(
                mcfg["tensor_model_parallel_size"] * mcfg["context_parallel_size"],
                mcfg.get("expert_tensor_parallel_size", 1)
                * mcfg.get("expert_model_parallel_size", 1),
            )
            for mcfg in layouts
        )

    @classmethod
    def init_cluster_placement_groups(
        cls,
        cluster: RayVirtualCluster,
        config: PolicyConfig,
    ) -> None:
        """Pre-initialize the inference cluster's placement groups.

        Args:
            cluster: The inference `RayVirtualCluster`.
            config: The full `PolicyConfig` (megatron parallelism + colocation).
        """
        colocated = config["generation"]["colocated"]["enabled"]
        cluster._init_placement_groups(
            strategy=None if colocated else "PACK",
            use_unified_pg=cls.nvlink_domain_span(config) > cluster.num_gpus_per_node,
        )

    @classmethod
    def reserve_http_server_address(
        cls,
        cluster: RayVirtualCluster,
        config: PolicyConfig,
    ) -> tuple[str, int, ray.actor.ActorHandle]:
        """Reserve the OpenAI server address before any generation worker exists.

        This is megatron's substitute for vLLM's `defer_model_load` overlap.
        See https://github.com/NVIDIA-NeMo/RL/issues/3752

        Args:
            cluster: The cluster the generation workers will run on.
            config: The full `PolicyConfig`.

        Returns:
            Tuple of (server base URL, reserved port, port-holder actor handle).
            The caller must keep the handle referenced until rank 0 has adopted
            the socket (worker init complete), then `ray.kill` it.
        """
        # Colocated generation shares the training policy's cluster and uses the
        # default placement-group init, triggered lazily by the read below.
        if not config["generation"]["colocated"]["enabled"]:
            cls.init_cluster_placement_groups(cluster, config)

        # Distributed rank 0 lands on the first bundle handed to the worker
        # group: sorted-first for a unified placement group, else bundle 0 of
        # the first group (mirrors Policy's worker-group construction).
        placement_groups = cluster.get_placement_groups()
        rank0_bundle_index = (
            cluster._sorted_bundle_indices[0]
            if cluster._sorted_bundle_indices is not None
            else 0
        )
        # Zero-gap reservation: a holder actor on the rank-0 node binds and
        # HOLDS the socket (num_cpus=0, so it schedules even on a full bundle);
        # rank 0 later adopts the live fd via receive_held_socket, so the port
        # can never be stolen in between and any free port is safe.
        holder = RemoteHeldPortReservation.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_groups[0],
                placement_group_bundle_index=rank0_bundle_index,
            ),
        ).remote()
        node_ip, port = ray.get(holder.address.remote())
        return f"http://{node_ip}:{port}/v1", port, holder

    @classmethod
    def validate_settings(cls, master_config: "MasterConfig") -> None:
        """Reject config the Megatron generation backend cannot honor."""
        policy_config: PolicyConfig = master_config.policy
        recompute_kv_cache_after_weight_updates: bool = (
            master_config.async_rl.recompute_kv_cache_after_weight_updates
        )
        if not (
            "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]
        ):
            raise ValueError(
                "policy.generation.backend='megatron' requires the Megatron trainer "
                "(policy.megatron_cfg.enabled=true): refit transfers weights via Megatron reshard "
                "collective from the Megatron trainer."
            )

        mcore_cfg = cast(MCoreGenerationConfig, policy_config["generation"])[
            "mcore_generation_config"
        ]
        # Recompute-after-refit is implemented engine-side (kv_cache_management_mode="recompute");
        # the loop-level flag must agree with that mode, and setup errors on a mismatch.
        kv_cache_mode = mcore_cfg["kv_cache_management_mode"]
        if recompute_kv_cache_after_weight_updates != (kv_cache_mode == "recompute"):
            raise ValueError(
                "async_rl.recompute_kv_cache_after_weight_updates="
                f"{recompute_kv_cache_after_weight_updates} conflicts with "
                "policy.generation.mcore_generation_config."
                f"kv_cache_management_mode={kv_cache_mode!r}: with "
                "policy.generation.backend='megatron' the two must agree. Either "
                "set the flag to true with kv_cache_management_mode='recompute', "
                "or leave the flag false with 'persist'/'offload'."
            )

        if master_config.async_rl.generation_fleet_health.enabled:
            raise NotImplementedError(
                "async_rl.generation_fleet_health.enabled=true is not supported "
                f"for the {cls.__name__} generation backend"
            )

    @classmethod
    def verify_served_address(
        cls, served_urls: list[Optional[str]], reserved_url: str
    ) -> None:
        """Fail loud if the engine serves anywhere but the pre-published address."""
        if served_urls != [reserved_url]:
            raise RuntimeError(
                "Megatron server came up at a different address than the one "
                f"pre-published to NeMo Gym: reserved {reserved_url}, serving {served_urls}."
            )

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: PreTrainedTokenizerBase,
        cluster: Optional[RayVirtualCluster] = None,
        policy: Optional["Policy"] = None,
        name_prefix: str = "megatron_generation",
        processor: Optional[AutoProcessor] = None,
        skip_weight_load: bool = False,
        reserved_http_server_port: Optional[int] = None,
    ):
        """Initialize a MegatronGeneration instance.

        Exactly one of `cluster` or `policy` must be provided.

        Args:
            config: PolicyConfig for the Megatron model.
            tokenizer: The tokenizer for the model.
            cluster: Cluster for a dedicated, non-colocated inference Policy.
            policy: Existing training Policy reused for colocated generation.
            name_prefix: Prefix for naming the worker group (non-colocated only).
            processor: Optional processor for VLMs (non-colocated only).
            skip_weight_load: Do not load weights from the checkpoint; refit will do it.
                Inference-engine initialization is deferred until that first refit so CUDA
                graphs capture the final persistent weight buffers rather than placeholder
                checkpoint tensors.
            reserved_http_server_port: Driver-reserved OpenAI server port for non-colocated.
        """
        # Import here to avoid circular imports
        from nemo_rl.models.policy.lm_policy import Policy

        assert (cluster is None) != (policy is None), (
            "Provide exactly one of `cluster` or `policy`."
        )
        assert not (skip_weight_load and policy is not None), (
            "skip_weight_load only applies to the dedicated inference policy."
        )
        assert not (reserved_http_server_port is not None and policy is not None), (
            "reserved_http_server_port only applies to the dedicated inference "
            "policy; when colocated, pass it to the training policy instead."
        )

        # `self.cfg` exposes the `generation` that matches the `GenerationInterface` contract.
        # `self._policy_config` keeps a reference to the full PolicyConfig. Dedicated
        # inference receives a copy because worker setup may modify it.
        self._policy_config = config
        self.cfg: MCoreGenerationConfig = config["generation"]
        refit_transport = self.cfg.get("refit_transport")
        if refit_transport not in (None, "mcore", "nccl_reshard"):
            raise ValueError(
                "policy.generation.refit_transport must be null, 'mcore', or "
                f"'nccl_reshard' for Megatron generation, got {refit_transport!r}."
            )
        if refit_transport == "mcore":
            refit_backend = self.cfg["mcore_generation_config"].get(
                "refit_backend", "gloo"
            )
            if refit_backend not in ("gloo", "nccl", "nccl_m2n", "nvshmem"):
                raise ValueError(
                    "policy.generation.mcore_generation_config.refit_backend "
                    "must be 'gloo', 'nccl', 'nccl_m2n', or 'nvshmem' when "
                    f"refit_transport='mcore', got {refit_backend!r}."
                )
            if policy is not None and refit_backend == "nccl_m2n":
                raise ValueError(
                    "policy.generation.mcore_generation_config.refit_backend="
                    "'nccl_m2n' is only supported with non-colocated generation."
                )
        elif self.cfg["mcore_generation_config"].get("refit_backend") in (
            "nccl_m2n",
            "nvshmem",
        ):
            # Only the native MCore copy service reads refit_backend. Warn rather
            # than raise: every shipped exemplar sets refit_backend unconditionally,
            # so another transport legitimately inherits a non-null value. A user
            # who explicitly asked for a non-default transport, though, would
            # otherwise silently get neither.
            # .get(): unlike the 'mcore' branch above, the other paths never
            # requires this key, so reading it must not turn an omitted key into a
            # KeyError.
            warnings.warn(
                "policy.generation.mcore_generation_config.refit_backend="
                f"{self.cfg['mcore_generation_config']['refit_backend']!r} is "
                f"ignored when refit_transport={refit_transport!r}; it is only "
                "read by the native MCore refit (refit_transport='mcore').",
                stacklevel=2,
            )
        # Populated after the first prepare_for_generation (which starts the HTTP server).
        self.dp_openai_server_base_urls: list[Optional[str]] = []
        # Installed by setup via create_weight_synchronizer.
        self.weight_synchronizer: Optional["WeightSynchronizer"] = None
        # The nccl_reshard synchronizer records its current rank layout before
        # dispatching any communicator or refit calls. None is the legacy/full-group
        # path used by refit implementations that do not manage membership.
        self._refit_membership: Optional["RefitMembership"] = None

        if policy is not None:
            # Reuse the existing training policy.
            self._policy = policy
            self._owns_policy = False
            if self.cfg["mcore_generation_config"]["expose_http_server"]:
                self._policy.offload_before_refit()
                self.prepare_for_generation()
            return

        # Stand up a dedicated inference-only policy.
        self._owns_policy = True
        self._policy_config = {
            **deepcopy(config),
            "megatron_cfg": self.effective_megatron_cfg(config),
        }
        # Reserve GPUs before Policy workers grab them, to prevent disjoint NVLS domains.
        self.init_cluster_placement_groups(cluster, self._policy_config)
        self._policy = Policy(
            cluster=cluster,
            config=self._policy_config,
            tokenizer=tokenizer,
            name_prefix=name_prefix,
            processor=processor,
            init_optimizer=False,
            init_reference_model=False,
            skip_weight_load=skip_weight_load,
            refit_role="destination",
            reserved_http_server_port=reserved_http_server_port,
        )

        # Skip-load models do not have their final refit weight buffers yet.
        # Defer engine initialization so CUDA graphs capture the persistent buffers.
        # The engine + HTTP server then first come up at the initial refit.
        if not skip_weight_load:
            self.prepare_for_generation()

    @property
    def uses_native_refit(self) -> bool:
        """Whether non-colocated refit uses Megatron Core's native mechanism."""
        return self.cfg.get("refit_transport") == "mcore"

    def get_refit_payload_mode(self) -> RefitPayloadMode:
        """Use the Megatron-to-Megatron logical-weight exception for M-to-N."""
        return "logical_weights"

    @property
    def worker_group(self) -> "RayWorkerGroup":
        """The underlying policy's worker group (fleet-health probes read dp_size)."""
        return self._policy.worker_group

    def init_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
        refit_backend: Optional[str] = None,
    ) -> list[ray.ObjectRef]:
        """Join the configured refit collective after the training ranks.

        Args:
            ip: IP address for the process group rendezvous.
            port: Port for the process group rendezvous.
            world_size: Total world size (train + inference workers).
            train_world_size: Number of training workers (used to offset ranks).
            refit_backend: Optional override for the native MCore copy-service
                backend ("gloo", "nccl", or "nccl_m2n"; "nvshmem" is
                currently broken and warns at setup, see
                https://github.com/NVIDIA-NeMo/RL/issues/3646). Ignored by
                the packed collective and nccl_reshard transports.

        Returns:
            List of Ray ObjectRefs for the collective init futures.
        """
        if self.uses_native_refit:
            backend = (
                refit_backend or self.cfg["mcore_generation_config"]["refit_backend"]
            )
            return self._policy.init_collective_mcore_generation(
                ip,
                port,
                world_size,
                rank_offset=train_world_size,
                refit_execution_batch_bytes=self.cfg["mcore_generation_config"][
                    "refit_execution_batch_bytes"
                ],
                refit_backend=backend,
            )
        return self._policy.init_collective(
            ip,
            port,
            world_size,
            train_world_size=train_world_size,
            rank_offset=train_world_size,
        )

    def update_weights_from_collective(
        self, refit_timeout_s: Optional[float] = None
    ) -> list[ray.ObjectRef]:
        """Receive weights through the configured Megatron refit mechanism."""
        if self.uses_native_refit:
            reject_unenforceable_refit_deadline("Megatron", refit_timeout_s)
            return self._policy.swap_weights_via_reshard(is_source=False)
        return self._policy.worker_group.run_all_workers_single_data(
            "update_weights_from_collective", refit_timeout_s=refit_timeout_s
        )

    def init_nccl_reshard_comm_group(
        self,
        *,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> list[ray.ObjectRef]:
        """Join every training PP stage's NCCL M-to-N communicator."""
        return self._policy.worker_group.run_all_workers_single_data(
            "init_nccl_reshard_comm_groups_generation",
            pp_ips=pp_ips,
            pp_ports=pp_ports,
            pp_size=pp_size,
            train_ranks_per_stage=train_ranks_per_stage,
            sub_world_size=sub_world_size,
        )

    def set_refit_membership(self, membership: "RefitMembership") -> None:
        """Record the inference ranks participating in nccl_reshard refits."""
        self._refit_membership = membership

    def _refit_ranked_workers(
        self, membership: Optional["RefitMembership"] = None
    ) -> list[tuple[Any, int, int]]:
        """Return ``(actor, rebuilt rank, original rank)`` for live workers."""
        active = membership or self._refit_membership
        if active is None:
            raise RuntimeError("Refit membership has not been initialized.")

        workers = self.worker_group.workers
        ranked_workers: list[tuple[Any, int, int]] = []
        for shard_idx, rank_prefix in active.shard_prefixes.items():
            worker_start = shard_idx * active.workers_per_shard
            for local_rank in range(active.workers_per_shard):
                worker_idx = worker_start + local_rank
                if worker_idx >= len(workers):
                    raise RuntimeError(
                        f"shard {shard_idx} maps to worker {worker_idx}, but the "
                        f"group has {len(workers)} workers"
                    )
                ranked_workers.append(
                    (workers[worker_idx], rank_prefix + local_rank, worker_idx)
                )
        return ranked_workers

    def rebuild_collective(
        self, membership: "RefitMembership", ip: str, port: int
    ) -> list[ray.ObjectRef]:
        """Build the misc-weight communicator over the selected Megatron ranks."""
        futures = []
        for worker, rank, original_rank in self._refit_ranked_workers(membership):
            futures.append(
                worker.init_collective.remote(
                    ip=ip,
                    port=port,
                    world_size=membership.world_size,
                    train_world_size=membership.train_world_size,
                    rank_offset=membership.train_world_size + rank - original_rank,
                    nccl_peer=self.get_collective_sender_spec().nccl_peer,
                )
            )
        return futures

    def rebuild_nccl_reshard_comm_group(
        self,
        membership: "RefitMembership",
        *,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> list[ray.ObjectRef]:
        """Build bulk communicators over the selected Megatron ranks."""
        return [
            worker.init_nccl_reshard_comm_groups_generation.remote(
                pp_ips=pp_ips,
                pp_ports=pp_ports,
                pp_size=pp_size,
                train_ranks_per_stage=train_ranks_per_stage,
                sub_world_size=sub_world_size,
                rank_prefix=rank,
            )
            for worker, rank, _original_rank in self._refit_ranked_workers(membership)
        ]

    def prepare_nccl_reshard_refit_info(self, refit_info: dict[str, Any]) -> None:
        """Build each inference worker's HF-to-Megatron M-to-N receive map."""
        if self._refit_membership is None:
            futures = self._policy.worker_group.run_all_workers_single_data(
                "prepare_nccl_reshard_refit_info", refit_info=refit_info
            )
        else:
            futures = [
                worker.prepare_nccl_reshard_refit_info.remote(refit_info=refit_info)
                for worker, _rank, _original_rank in self._refit_ranked_workers()
            ]
        ray.get(futures)

    def nccl_reshard_refit(
        self, refit_timeout_s: Optional[float] = None
    ) -> list[ray.ObjectRef]:
        """Receive one NCCL M-to-N refit on every Megatron inference worker."""
        if self._refit_membership is None:
            return self._policy.worker_group.run_all_workers_single_data(
                "nccl_reshard_refit", refit_timeout_s=refit_timeout_s
            )
        return [
            worker.nccl_reshard_refit.remote(refit_timeout_s=refit_timeout_s)
            for worker, _rank, _original_rank in self._refit_ranked_workers()
        ]

    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using the Megatron generation backend.

        mcore's data-parallel coordinator only accepts requests from DP rank 0 —
        the other workers' engine loops drain the coordinator queue but never
        receive a Python-side call. So we dispatch straight to worker 0.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths.
            greedy: Whether to use greedy decoding.

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec.
        """
        future = self._policy.worker_group.run_single_worker_single_data(
            method_name="generate",
            worker_idx=0,
            data=data,
            greedy=greedy,
        )
        return ray.get(future)

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate asynchronously, yielding `(index, batch)` tuples as they complete."""
        worker = self._policy.worker_group.workers[0]
        futures = worker.generate_async.options(num_returns="streaming").remote(
            data=data, greedy=greedy
        )
        async for result_ref in futures:
            index, result_batch = await result_ref
            result_batch["gen_leader_worker_idx"] = [0]
            yield index, result_batch

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Initialize / re-enter inference mode on every worker.

        First call starts the persistent inference engine, coordinator, and the OpenAI HTTP server.
        Subsequent calls re-enter inference mode after a refit.
        """
        futures = self._policy.worker_group.run_all_workers_single_data(
            "prepare_for_generation", **kwargs
        )
        ray.get(futures)
        if (
            not self.dp_openai_server_base_urls
            and self.cfg["mcore_generation_config"]["expose_http_server"]
        ):
            url_futures = self._policy.worker_group.run_all_workers_single_data(
                "report_dp_openai_server_base_url"
            )
            self.dp_openai_server_base_urls = [
                url for url in ray.get(url_futures) if url is not None
            ]
        return True

    def finish_generation(self, *, release_gpu: bool = True) -> bool:
        """Clean up after generation.

        When `release_gpu` is False, a colocated engine keeps serving instead of standing down.
        """
        futures = self._policy.worker_group.run_all_workers_single_data(
            "finish_generation", release_gpu=release_gpu
        )
        ray.get(futures)
        return True

    def blocks_training(self) -> bool:
        """Whether the engine must stand down before a training step.

        Colocated generation shares the training GPUs, so the training
        loop must wind the engine down before it can train.
        """
        return bool(self.cfg["colocated"]["enabled"])

    def wake_carries_weight_updates(self) -> bool:
        """The colocated wake reshards (or shares tensors); see the ABC."""
        return bool(self.cfg["colocated"]["enabled"])

    def invalidate_kv_cache(self) -> bool:
        """Report whether weight updates invalidate the KV cache.

        Under "recompute" mode the engine drops and rebuilds its KV cache
        across the suspend/resume that brackets every weight update, so
        invalidation is genuinely handled; report it truthfully instead of
        inheriting the interface's `False` (which makes the trajectory
        collector warn every step).
        """
        return (
            self.cfg["mcore_generation_config"].get("kv_cache_management_mode")
            == "recompute"
        )

    def preinit_nvshmem_collective(self) -> list[ray.ObjectRef]:
        """Pre-initialize NVShmem collectively outside CUDA graph capture.

        Must be called simultaneously on both training and inference workers.
        """
        if not self.uses_native_refit:
            raise RuntimeError(
                "NVSHMEM pre-initialization is only valid with refit_transport='mcore'."
            )
        return self._policy.preinit_nvshmem()

    def suspend_for_refit(self) -> None:
        """Suspend the inference engine for safe weight updates."""
        ray.get(
            self._policy.worker_group.run_all_workers_single_data("suspend_for_refit")
        )

    def resume_after_refit(self) -> None:
        """Resume the inference engine after weight updates."""
        ray.get(
            self._policy.worker_group.run_all_workers_single_data("resume_after_refit")
        )

    def prepare_refit_info(self, state_dict_info: Optional[dict[str, Any]]) -> None:
        """Prepare Bridge conversion tasks on every dedicated inference worker."""
        if not self._owns_policy or self.uses_native_refit:
            return
        if state_dict_info is None:
            raise ValueError("Megatron collective refit requires state_dict_info.")
        futures = self._policy.worker_group.run_all_workers_single_data(
            "prepare_refit_info", state_dict_info=state_dict_info
        )
        ray.get(futures)

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling on the dedicated inference workers.

        No-op when colocated: the shared workers are already profiled through the training policy.
        """
        if self._owns_policy:
            self._policy.start_gpu_profiling()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling on the dedicated inference workers."""
        if self._owns_policy:
            self._policy.stop_gpu_profiling()

    def shutdown(self) -> bool:
        """Shut down all inference workers and clean up resources."""
        if not self._owns_policy:
            return True
        return self._policy.shutdown()

    def __del__(self) -> None:
        """Safety net to ensure workers are shut down."""
        if hasattr(self, "_policy") and self._owns_policy:
            self._policy.shutdown()
