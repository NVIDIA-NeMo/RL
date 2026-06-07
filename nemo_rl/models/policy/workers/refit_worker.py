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

Architecture (RL-412 follow-up):

The cross-cluster ``model_update_group`` (StatelessProcessGroup, NCCL) used to
broadcast weights from train rank 0 to vLLM gen ranks lived inside the train
worker process itself. That meant a gen peer dying mid-broadcast would queue
``cudaErrorLaunchFailure`` into train rank 0's CUDA primary context, which is
shared device-wide across every NCCL group in the process (Megatron's
EP/TP/PP/DP groups all sit on the same context). Stream isolation contained
the symptom but not the root cause — a queued device error can surface on
any subsequent CUDA op submitted by that process.

The RefitWorker fixes this by hosting the cross-cluster group in a SEPARATE
PROCESS, scheduled with hard node affinity to the same node and the same
physical GPU as train rank 0. Two processes sharing one GPU is fine — they
have independent primary CUDA contexts, and CUDA IPC lets us hand a buffer
across the process boundary in zero-copy form.

Cross-cluster world layout becomes:
  - rank 0: RefitWorker (this actor)
  - ranks 1..N: gen workers
  - Megatron policy workers DO NOT participate.

Data path mirrors the proven colocated-vLLM ZMQ-IPC ping-pong pipeline
(``stream_weights_via_ipc_zmq_impl`` + ``vllm_backend.update_weights_via_ipc_zmq``):

  1. Train rank 0 packs params into one of two GPU "ping-pong" buffers,
     ``cuda.synchronize()``s, ``reduce_tensor``-exports a CUDA IPC handle,
     and sends ``(handle, param_names, used_bytes)`` over ZMQ REQ.
  2. RefitWorker (REP) ``recv_pyobj``s the handle, ``rebuild_cuda_tensor``s
     it (zero-copy view of train rank 0's GPU memory — same physical GPU),
     and broadcasts the buffer to gen ranks via NCCL on its own
     ``model_update_group``.
  3. RefitWorker syncs the broadcast stream (surfaces any queued NCCL
     error from a dead peer immediately, scoped to its own process) and
     ACKs back so the sender can swap to the other ping-pong buffer.
  4. When the sender is done it sends ``IPCProtocol.COMPLETE``; RefitWorker
     finishes, ACKs, and returns from ``broadcast_weights_until_complete``.
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


# Use num_gpus=0 — RefitWorker shares the physical GPU with train rank 0
# (separate processes, separate primary CUDA contexts; CUDA IPC bridges
# them). Pinning via NodeAffinitySchedulingStrategy in the orchestrator
# is what gets it onto the right node; here we just need Ray to NOT
# allocate a brand-new GPU.
@ray.remote(num_gpus=0)  # pragma: no cover
class RefitWorker:
    """Standalone process that owns the cross-cluster weight-sync NCCL group.

    Lives on the same node + same physical GPU as train rank 0, in a
    separate process. Receives weight buffers from train rank 0 via the
    existing CUDA IPC + ZMQ pipeline and forwards them onward to gen
    workers via NCCL.
    """

    def __init__(self, gpu_index: int) -> None:
        # ``num_gpus=0`` makes Ray set ``CUDA_VISIBLE_DEVICES=""``
        # (empty) on this actor's process, masking ALL GPUs. We
        # FORCE-override it to the physical GPU we want to share with
        # train rank 0 — ``setdefault`` is wrong here because Ray's
        # empty value already exists in os.environ; we need to
        # overwrite it. This must happen BEFORE the first CUDA op
        # (torch defers cuInit until the first cuda call), so do it
        # at the very top of ``__init__`` before ``set_device``.
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        # If torch.cuda was already initialized in this process with
        # the empty visibility list, force re-init by clearing the
        # internal cache. The first ``torch.cuda.set_device(0)`` below
        # will then re-read CUDA_VISIBLE_DEVICES.
        if torch.cuda.is_initialized():
            print(
                "[refit_worker] WARNING: torch.cuda already initialized "
                "before CUDA_VISIBLE_DEVICES override; this should not "
                "happen with num_gpus=0 + clean process start",
                flush=True,
            )
        # Bind the CUDA primary context to cuda:0 (which now points at
        # the physical GPU above).
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
        # Cross-cluster broadcast on a 30B-class model: first chunk
        # arrival is bounded below by Megatron's expert-parallel
        # all-gather + IPC handle export on rank 0 (~30-90s warm,
        # longer if Megatron hasn't materialized the optimizer state
        # yet). 600s gives plenty of headroom; subsequent chunks land
        # in <1s each.
        # peer doesn't get summarily killed. The outer broadcast retry
        # loop in grpo handles real wedges.
        self._zmq_socket.setsockopt(zmq.SNDTIMEO, 600000)
        self._zmq_socket.setsockopt(zmq.RCVTIMEO, 600000)
        self._zmq_socket.setsockopt(zmq.LINGER, 0)
        # /tmp is mounted into the pod and shared between processes
        # on the same node — we already use /tmp/{device_uuid}.sock for
        # the colocated path. Use a PID-namespaced path to avoid
        # collisions with a sibling worker on the same GPU.
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
                # Rebuild the tensor as a view onto the sender's GPU
                # memory. Same physical GPU → no cross-device traffic.
                ipc_view = rebuild_cuda_tensor_from_ipc(ipc_handle, 0)
                # DEFENSIVE COPY into RefitWorker's own allocation:
                # the IPC view aliases the sender's storage. Once we
                # ACK, the sender is free to reuse / reallocate that
                # storage. NCCL broadcast across an internode wire
                # may *kernel-complete* on the source rank before the
                # destination ranks have finished pulling — and even
                # if it doesn't, having the producer hold the buffer
                # for the duration of the entire collective is a
                # subtle dependency we don't want. ``clone()`` plus
                # ``torch.cuda.synchronize()`` guarantees the bytes
                # we're about to broadcast are stable in OUR memory
                # before we release the view back to the producer.
                #
                # 30B-class refit produces ~5GB chunks; one extra
                # buffer here costs ~5GB of GPU memory (we share the
                # GPU with train rank 0 which sits at ~70% util, so
                # there's headroom — Megatron's 30B-A3B leaves a few
                # tens of GB free per H100/B100 once the optimizer
                # state is shed during refit).
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
                # Release the view to the sender BEFORE the broadcast
                # so the sender can safely reuse its buffer the moment
                # ACK arrives. The clone above means the broadcast
                # reads from OUR memory, not the sender's.
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
            # Tear down the cross-cluster group ONLY on failure. On success we
            # KEEP the group alive across refits.
            #
            # Recreating the group every refit forces a fresh cross-cluster
            # TCPStore rendezvous on a NEW master port each step. On this
            # cluster that rendezvous is intermittently flaky (1-2 gen workers'
            # TCPStore clients fail to connect/validate → they never join the
            # NCCL group → the broadcast collective can never complete → every
            # survivor trips the 60s in-band abort and a healthy worker gets
            # evicted, all with NO injected fault). Holding the group alive
            # makes a steady-state refit a plain broadcast on the existing
            # comm (~2.5s, no rendezvous, reliable) — mirroring the non-disagg
            # path. ``ensure_collective_synced`` re-inits only when the world
            # size changes (a real fault); ``_last_synced_world_size`` gates it.
            #
            # Faults are still handled without a per-refit teardown: a peer
            # dying mid-broadcast trips the consumer's in-band abort
            # (StatelessProcessGroup.synchronize_or_abort) → this method
            # returns False → the grpo retry loop ray.kills + respawns this
            # RefitWorker (poisoned context dies with the process) and re-inits
            # at the new world. On FAILURE we MUST tear down here so a wedged
            # group is never reused.
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
