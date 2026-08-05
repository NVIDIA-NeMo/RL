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
"""RefitWorker — sibling Ray actor that owns the cross-cluster weight-sync NCCL group.

Hosts the cross-cluster NCCL group in a separate process (same node + GPU as train
rank 0, independent CUDA context) so a gen peer dying mid-broadcast can't corrupt
train rank 0's primary context. Receives weight buffers from train rank 0 via CUDA
IPC + ZMQ and broadcasts them to gen workers via NCCL.
"""
import gc
import hashlib
import traceback
from typing import Any, Optional

import ray
import torch
import zmq

from nemo_rl.models.policy.utils import (
    IPCProtocol,
    rebuild_cuda_tensor_from_ipc,
)


# num_gpus=0: shares the physical GPU with train rank 0 via separate CUDA contexts + IPC.
@ray.remote(num_gpus=0)  # pragma: no cover
class RefitWorker:
    """Standalone process that owns the cross-cluster weight-sync NCCL group.

    Lives on the same node + same physical GPU as train rank 0, in a
    separate process. Receives weight buffers from train rank 0 via the
    existing CUDA IPC + ZMQ pipeline and forwards them onward to gen
    workers via NCCL.
    """

    def __init__(self, gpu_index: int) -> None:
        # Force-override CUDA_VISIBLE_DEVICES before the first CUDA op (Ray sets it to "").
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        if torch.cuda.is_initialized():
            print(
                "[refit_worker] WARNING: torch.cuda already initialized "
                "before CUDA_VISIBLE_DEVICES override; this should not "
                "happen with num_gpus=0 + clean process start",
                flush=True,
            )
        torch.cuda.set_device(0)
        self._gpu_index = gpu_index
        self._device = torch.device("cuda:0")
        self._group: Optional[Any] = None
        self._rank: int = 0
        self._world_size: int = 0
        self._zmq_context: Optional[zmq.Context] = None
        self._zmq_socket: Optional[Any] = None
        self._zmq_address: Optional[str] = None
        print(
            f"[refit_worker] initialized; gpu_index={gpu_index} "
            f"node_id={ray.get_runtime_context().get_node_id()}",
            flush=True,
        )

    def init_collective(
        self, ip: str, port: int, world_size: int, rank: int = 0
    ) -> None:
        """Bind a fresh ``StatelessProcessGroup`` and rendezvous over NCCL.

        Idempotent: if a previous group exists (e.g. from a prior fault
        cycle that didn't ``ray.kill`` this actor), it is destroyed first.
        Any queued NCCL/CUDA error from the prior group surfaces here and
        is left to propagate to the caller — the caller is then expected
        to ``ray.kill`` this actor and respawn fresh.
        """
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        old = self._group
        if old is not None:
            try:
                old.destroy()
            except Exception as e:  # noqa: BLE001
                print(
                    f"[refit_worker.init_collective] prior group destroy "
                    f"raised {type(e).__name__}: {e} — continuing with re-init",
                    flush=True,
                )
            try:
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                print(
                    f"[refit_worker.init_collective] post-destroy synchronize "
                    f"swallowed: {type(e).__name__}: {e}",
                    flush=True,
                )
            try:
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                print(
                    f"[refit_worker.init_collective] post-destroy empty_cache "
                    f"swallowed: {type(e).__name__}: {e}",
                    flush=True,
                )
            self._group = None

        self._rank = rank
        self._world_size = world_size
        print(
            f"[refit_worker.init_collective] binding StatelessProcessGroup "
            f"on {ip}:{port} rank={rank} world_size={world_size}",
            flush=True,
        )
        self._group = StatelessProcessGroup(
            master_address=ip, port=port, rank=rank, world_size=world_size
        )
        # Fixed device 0 — Ray pins this actor to a single GPU surfaced as
        # cuda:0 inside its CUDA_VISIBLE_DEVICES sandbox.
        self._group.init_nccl_communicator(device=0)
        print(
            f"[refit_worker.init_collective] rendezvous complete "
            f"(rank={rank}/{world_size})",
            flush=True,
        )

    def start_zmq_server(self) -> str:
        """Bind a REP socket and return its IPC address.

        Mirrors ``base_policy_worker.maybe_init_zmq`` (REQ side) but in
        REP role — train rank 0 will REQ-connect to this address. We
        use a unique IPC path tied to this actor's PID so multiple
        RefitWorkers on the same node don't collide.
        """
        if self._zmq_socket is not None:
            return self._zmq_address  # type: ignore[return-value]
        import os

        self._zmq_context = zmq.Context()
        self._zmq_socket = self._zmq_context.socket(zmq.REP)
        # 600s timeout covers first-chunk latency (~30-90s for large models); subsequent chunks are fast.
        self._zmq_socket.setsockopt(zmq.SNDTIMEO, 600000)
        self._zmq_socket.setsockopt(zmq.RCVTIMEO, 600000)
        self._zmq_socket.setsockopt(zmq.LINGER, 0)
        # PID-namespaced path avoids collisions with sibling workers on the same GPU.
        self._zmq_address = f"ipc:///tmp/refit-worker-{os.getpid()}.sock"
        self._zmq_socket.bind(self._zmq_address)
        print(
            f"[refit_worker.start_zmq_server] listening on {self._zmq_address}",
            flush=True,
        )
        return self._zmq_address

    def broadcast_weights_until_complete(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> bool:
        """Run the receive-and-broadcast loop until the sender signals COMPLETE.

        Mirrors ``vllm_backend.update_weights_via_ipc_zmq`` but instead of
        loading weights into a model, broadcasts the IPC-imported buffer
        to gen workers via NCCL.

        Returns True on clean completion, False on error (mirrors the
        boolean contract of ``update_weights_via_ipc_zmq``).
        """
        # ``kv_scales`` is consumed by the SENDER (Megatron worker) to
        # decide what to put into the byte stream. RefitWorker is
        # backend-agnostic and just forwards bytes — accept the kwarg so
        # the calling signature stays uniform.
        del kv_scales

        if self._group is None:
            print(
                "[refit_worker.broadcast_weights_until_complete] called "
                "before init_collective — refusing",
                flush=True,
            )
            return False
        if self._zmq_socket is None:
            print(
                "[refit_worker.broadcast_weights_until_complete] called "
                "before start_zmq_server — refusing",
                flush=True,
            )
            return False

        import os as _os

        _refit_pack_debug = _os.getenv("NRL_REFIT_PACKED_DEBUG", "0") == "1"
        buffer: Optional[torch.Tensor] = None
        ipc_view: Optional[torch.Tensor] = None
        chunk_idx = 0
        # Keep the cross-cluster group ALIVE on success; tear it down only on
        # failure (see the ``finally`` below for the reasoning).
        _ok = False
        try:
            while True:
                payload = self._zmq_socket.recv_pyobj()
                if payload == IPCProtocol.COMPLETE:
                    self._zmq_socket.send(IPCProtocol.ACK.value.encode())
                    break
                ipc_handle, _list_keys, _used_bytes = payload
                # Zero-copy view of sender's GPU memory (same physical GPU).
                ipc_view = rebuild_cuda_tensor_from_ipc(ipc_handle, 0)
                # Clone into our own allocation before ACKing so the sender can reuse its buffer
                # while NCCL still reads ours.
                buffer = ipc_view.clone()
                torch.cuda.synchronize()
                if _refit_pack_debug:
                    # Tiny head/tail fingerprint so we can compare
                    # bytes the producer wrote vs bytes RefitWorker
                    # imported vs bytes the gen consumer received.
                    # Three-way diff isolates the corruption stage:
                    # IPC, NCCL, or post-receive.
                    _src_sha = hashlib.sha1(
                        buffer[: min(64, buffer.numel())]
                        .cpu()
                        .numpy()
                        .tobytes()
                        + buffer[max(0, buffer.numel() - 64) :]
                        .cpu()
                        .numpy()
                        .tobytes()
                    ).hexdigest()[:12]
                    print(
                        f"[refit_diag.refitworker] chunk={chunk_idx} "
                        f"bytes={int(buffer.numel())} head_tail_sha={_src_sha}",
                        flush=True,
                    )
                # Release view before ACK so sender can reuse its buffer immediately.
                del ipc_view
                ipc_view = None
                # Now broadcast our LOCAL stable copy.
                self._group.broadcast(buffer, src=self._rank)
                torch.cuda.synchronize()
                del buffer
                buffer = None
                self._zmq_socket.send(IPCProtocol.ACK.value.encode())
                chunk_idx += 1
            _ok = True
            return True
        except Exception as e:  # noqa: BLE001
            # Log + return False so the caller (Policy) can ray.kill us
            # and respawn with a fresh CUDA context. Don't re-raise:
            # ZMQ errors from a wedged peer should not poison the actor
            # state in a way that ``ray.get`` then masks; the caller
            # owns the policy decision.
            print(
                f"[refit_worker.broadcast_weights_until_complete] error: "
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
                flush=True,
            )
            return False
        finally:
            if buffer is not None:
                del buffer
            if ipc_view is not None:
                del ipc_view
            # Keep the comm alive on success (avoids per-step rendezvous flakiness).
            # On failure, tear it down so a wedged group is never reused.
            if not _ok:
                try:
                    old = self._group
                    self._group = None
                    if old is not None:
                        old.destroy()
                except Exception as _e:  # noqa: BLE001
                    print(
                        f"[refit_worker] post-broadcast destroy raised "
                        f"{type(_e).__name__}: {_e} — group already gone",
                        flush=True,
                    )
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    def abort_collective(self) -> None:
        """Tear down the cross-cluster group and surface any queued NCCL error.

        Any ``cudaErrorLaunchFailure`` queued by a dead peer surfaces from
        ``StatelessProcessGroup.destroy()`` (which syncs the dedicated
        broadcast stream first). We re-raise so the caller knows the CUDA
        context of THIS actor is poisoned and the actor should be killed
        + respawned. The caller (lm_policy.abort_collective) does exactly
        that via ``ray.kill``, so the next ``init_collective`` lands on a
        fresh primary context.
        """
        old = self._group
        self._group = None
        if old is None:
            return
        try:
            old.destroy()
        except Exception as e:  # noqa: BLE001
            print(
                f"[refit_worker.abort_collective] destroy surfaced "
                f"{type(e).__name__}: {e} — CUDA context likely poisoned, "
                f"caller must ray.kill this actor",
                flush=True,
            )
            raise

    def is_alive(self) -> bool:
        """Liveness ping — used by orchestrator to verify the actor is up."""
        return True

    def get_node_id(self) -> str:
        """Return the Ray node id this actor was scheduled onto.

        Used by the orchestrator to verify hard-affinity placement landed
        on the same node as train rank 0.
        """
        return ray.get_runtime_context().get_node_id()

    def shutdown(self) -> bool:
        """Best-effort shutdown — close ZMQ + tear down NCCL."""
        try:
            if self._zmq_socket is not None:
                self._zmq_socket.close()
            if self._zmq_context is not None:
                self._zmq_context.term()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.abort_collective()
        except Exception:  # noqa: BLE001
            pass
        return True
