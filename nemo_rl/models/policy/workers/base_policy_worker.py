# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any, Optional

import ray
import torch
import zmq

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.interfaces import ReferenceLogprobOutputSpec
from nemo_rl.telemetry.setup import shutdown_telemetry
from nemo_rl.utils.nsys import wrap_with_nvtx_name


class AbstractPolicyWorker:
    """Base class for policy workers with shared functionality."""

    # None until init_collective builds it. Declared so a rebuild can release the
    # previous group without probing for the attribute's existence.
    model_update_group: Optional[Any] = None
    # Same, for the per-PP-stage group the nccl_reshard transport builds.
    pp_comm_group: Optional[Any] = None

    def init_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
        nccl_peer: str = "nemo",
    ) -> None:
        """Initialize the collective communication.

        Args:
            ip: IP address for the process group
            port: Port for the process group
            world_size: Total world size (train_world_size + inference_world_size)
            train_world_size: Number of training workers (used in inference cluster)
            nccl_peer: NCCL initialization protocol used by the inference workers
        """
        from nemo_rl.distributed.refit_watchdog import RELEASE_GRACE_S, release_within
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        # Printed on both sides, like the reshard rendezvous below, and for the same
        # reason: a connect timeout names only the address, so without this there is no
        # way to tell a trainer that never bound the port from two sides that disagree
        # about which port to use. rank 0 is a trainer and is this store's master.
        print(
            f"  refit: collective rendezvous [train] addr={ip}:{port} "
            f"rank={self.rank} world_size={world_size} "
            f"train_world_size={train_world_size} peer={nccl_peer} "
            f"master={self.rank == 0}",
            flush=True,
        )

        # BIND BEFORE RELEASING, and never the other way round. rank 0 is this store's
        # master, so every other rank is already counting down a 300s connect timeout
        # against this port; anything slow that runs first is spent from that budget.
        # Releasing first cost job 6518381 the whole run -- see release_within.
        old_group, self.model_update_group = (
            self.model_update_group,
            StatelessProcessGroup(
                master_address=ip, port=port, rank=self.rank, world_size=world_size
            ),
        )
        # Rebuilding is the recovery path for a dead generation rank, so this runs more
        # than once per job. Without the release, each rebuild would strand the previous
        # NCCL communicator and its TCPStore for the life of the worker. Bounded, because
        # the peer this is releasing may be frozen rather than dead.
        if old_group is not None:
            release_within(
                old_group.abort,
                RELEASE_GRACE_S,
                "the previous refit communicator",
            )
        device = torch.cuda.current_device()
        # Release unused cached allocator blocks before NCCL communicator
        # initialization so transport buffers have sufficient device-memory headroom.
        torch.cuda.empty_cache()
        self.model_update_group.init_nccl_communicator(device=device, peer=nccl_peer)

    def init_nccl_reshard_comm_group(
        self,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        my_pp_stage: int,
        sub_world_size: int,
        my_rank_in_group: int,
    ) -> None:
        """Bootstrap this train worker's nccl_reshard comm group.

        One comm group per PP stage; each train worker joins exactly its own
        stage's group (``pp_ips[my_pp_stage]`` / ``pp_ports[my_pp_stage]``).
        Non-PP is simply ``pp_size == 1`` that contains all the train ranks.
        """
        from nemo_rl.distributed.refit_watchdog import RELEASE_GRACE_S, release_within
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        # Released for the same reason init_collective releases model_update_group:
        # _build re-runs both communicator families on every reconcile, so without this
        # each recovery strands a NCCL communicator and a bound TCPStore per PP stage for
        # the life of the worker. Deferred past the rendezvous and bounded, same as there.
        stale_group, self.pp_comm_group = self.pp_comm_group, None

        # Printed on both sides of this rendezvous, because a mismatch here is invisible
        # otherwise: the party that got it wrong just waits, and the other reports a 300s
        # connect timeout naming only the address. Jobs 6512153 and 6513879 both died that
        # way after an ABORTED reshard refit -- the same rebuild works after a clean kill --
        # and two rounds of reasoning about which side was at fault produced two wrong
        # answers. Compare these lines against the gen side's before theorising.
        print(
            f"  refit: reshard rendezvous [train] stage={my_pp_stage} "
            f"addr={pp_ips[my_pp_stage]}:{pp_ports[my_pp_stage]} "
            f"rank={my_rank_in_group} world_size={sub_world_size}",
            flush=True,
        )
        self.pp_comm_group = StatelessProcessGroup(
            master_address=pp_ips[my_pp_stage],
            port=pp_ports[my_pp_stage],
            rank=my_rank_in_group,
            world_size=sub_world_size,
        )
        device = torch.cuda.current_device()
        # Free cached blocks so NCCL P2P buffers have headroom (see init_collective).
        torch.cuda.empty_cache()
        self.pp_comm_group.init_nccl_communicator(device=device)
        self.my_pp_stage = my_pp_stage

        if stale_group is not None:
            release_within(
                stale_group.abort,
                RELEASE_GRACE_S,
                "the previous reshard bulk communicator",
            )

    def prepare_nccl_reshard_refit_info(
        self,
        train_parallelism: dict[str, int],
        gen_parallelism: dict[str, int],
        train_world_size: int,
        gen_world_size: int,
    ) -> dict[str, Any]:
        """Prepare parameter metadata for NCCL reshard refit."""
        # This is a placeholder implementation.
        # Implementation should be located in each policy worker implementation.
        raise NotImplementedError(
            "prepare_nccl_reshard_refit_info is not implemented for this policy worker"
        )

    def nccl_reshard_refit(
        self,
        kv_scales: Optional[dict[str, float]] = None,
        refit_timeout_s: Optional[float] = None,
    ) -> None:
        """Transfer policy weights with NCCL reshard refit."""
        # This is a placeholder implementation.
        # Implementation should be located in each policy worker implementation.
        raise NotImplementedError(
            "nccl_reshard_refit is not implemented for this policy worker"
        )

    def _refit_transport_state(self, key: str) -> dict:
        """Return the mutable state dict for one refit transport, e.g. ``"sglang_ipc"``.

        Transports stash their bookkeeping here (gather groups, engine handles,
        weight versions) instead of each backend adding fields to the worker
        constructors. Created on first use because this base class has no
        ``__init__`` — the same lazy shape as ``maybe_init_zmq`` on the ZMQ path.
        """
        if not hasattr(self, "_refit_state"):
            self._refit_state: dict[str, dict] = {}
        return self._refit_state.setdefault(key, {})

    @torch.no_grad()
    @wrap_with_nvtx_name("policy_worker/connect_sglang_rollout_engines")
    def connect_sglang_rollout_engines(
        self,
        *,
        engine_gpu_counts: list[int],
        engine_gpu_offsets: Optional[list[int]] = None,
    ) -> None:
        """Set up the colocate Gloo gather topology for SGLang weight refit.

        Must be called collectively by every trainer rank when SGLang engines
        are added or recovered. Subsequent calls with the same layout are
        no-ops.
        """
        from nemo_rl.models.generation.sglang.utils.train_utils import (
            monkey_patch_torch_reductions,
        )
        from nemo_rl.models.policy.utils import connect_colocate_topology

        # Colocate refit serializes CUDA-IPC tensor handles for SGLang; the
        # torch reductions monkey patch must be in place before any tensor is
        # serialized. Idempotent, so installing on every connect is safe.
        monkey_patch_torch_reductions()

        connect_colocate_topology(
            engine_gpu_counts=list(engine_gpu_counts),
            engine_gpu_offsets=(
                list(engine_gpu_offsets) if engine_gpu_offsets is not None else None
            ),
            worker_state=self._refit_transport_state("sglang_ipc"),
        )

    def stand_down_refit_watchdog(self) -> int:
        """Cancel any in-flight refit deadline in this worker; returns how many.

        Called by the controller when a generation shard's process is confirmed GONE. A
        dead peer closes its sockets and NCCL unblocks the survivors on its own, so the
        deadline has nothing useful to do and firing it is actively harmful -- see
        stand_down_armed_watchdogs.

        Deliberately trivial: this runs on the actor's event loop while the refit itself
        runs off it, so it must not block or touch CUDA.
        """
        from nemo_rl.distributed.refit_watchdog import stand_down_armed_watchdogs

        return stand_down_armed_watchdogs()

    def is_alive(self) -> bool:
        """Check if the worker is alive."""
        return True

    def reset_peak_memory_stats(self) -> None:
        """Reset peak memory statistics."""
        torch.cuda.reset_peak_memory_stats()

    def get_gpu_info(self) -> dict[str, Any]:
        """Return information about the GPU being used by this worker."""
        from nemo_rl.models.policy.utils import get_gpu_info

        return get_gpu_info(self.model)

    def report_device_id(self) -> str:
        """Report the UUID of the current CUDA device using NVML.

        Returns:
            str: UUID of the device in the format "GPU-xxxxx"
        """
        from nemo_rl.utils.nvml import get_device_uuid

        # Get current device index from torch
        device_idx = torch.cuda.current_device()
        # Get device UUID using NVML
        return get_device_uuid(device_idx)

    def get_zmq_address(self) -> str:
        """Get the ZMQ address for the current device."""
        return f"ipc:///tmp/{self.report_device_id()}.sock"

    def maybe_init_zmq(self) -> None:
        """Initialize the ZMQ socket if it doesn't exist."""
        if not hasattr(self, "zmq_socket"):
            self.zmq_context = zmq.Context()
            self.zmq_socket = self.zmq_context.socket(zmq.REQ)
            self.zmq_socket.setsockopt(
                zmq.SNDTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(
                zmq.RCVTIMEO, 120000
            )  # set timeout to 120 seconds
            self.zmq_socket.setsockopt(zmq.LINGER, 0)
            self.zmq_socket.bind(self.get_zmq_address())

    def get_free_memory_bytes(self) -> int:
        """Get the available free memory."""
        from nemo_rl.utils.nvml import get_free_memory_bytes

        device_idx = torch.cuda.current_device()
        return get_free_memory_bytes(device_idx)

    def shutdown(self) -> bool:
        """Shutdown the policy."""
        try:
            # Clean up extension resources like ZMQ sockets
            if hasattr(self, "zmq_socket"):
                self.zmq_socket.close()
                self.zmq_context.term()
            return True
        except Exception:
            return False
        finally:
            # Flush buffered spans/metrics before the actor goes away. Only the
            # async GRPO trainer calls shutdown() today; elsewhere Ray reaps the
            # actor and this never runs, so worker telemetry relies on the batch
            # processor's periodic export.
            shutdown_telemetry()

    def start_gpu_profiling(self) -> None:
        """Start GPU profiling."""
        torch.cuda.profiler.start()

    def stop_gpu_profiling(self) -> None:
        """Stop GPU profiling."""
        torch.cuda.profiler.stop()

    def report_node_ip_and_gpu_id(self) -> tuple[str, int]:
        """Report the node IP and GPU ID of the current worker."""
        ip = ray._private.services.get_node_ip_address()
        # Workers that manage their own LOCAL_RANK will have an empty `ray.get_gpu_ids()`.
        gpu_ids = ray.get_gpu_ids()
        gpu_id = gpu_ids[0] if gpu_ids else torch.cuda.current_device()
        return (ip, gpu_id)

    # Temporary fix, 'data' is a kwarg due to some sort of ray bug
    @wrap_with_nvtx_name("policy_worker/get_reference_policy_logprobs")
    def get_reference_policy_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        """Get the logprobs from the reference policy for a batch of data.

        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "reference_logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        with self.use_reference_model():
            reference_logprobs = self.get_logprobs(
                data=data, micro_batch_size=micro_batch_size
            )

        return_data = BatchedDataDict[ReferenceLogprobOutputSpec]()
        return_data["reference_logprobs"] = reference_logprobs["logprobs"].cpu()
        return return_data

    def finalize_async_save(self) -> None:
        """Block until any in-flight async checkpoint write completes. No-op by default."""
        pass

    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        # Placeholder implementation
        pass
