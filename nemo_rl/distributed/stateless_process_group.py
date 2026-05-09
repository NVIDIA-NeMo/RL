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

from datetime import timedelta
from typing import Optional

import torch
from nccl import bindings as _nccl_bindings
from nccl.core.communicator import Communicator, NCCLConfig
from nccl.core.utils import UniqueId, get_unique_id


# Bound TCPStore rendezvous + key-wait at 30s instead of the 300s
# default. The philosophy: rendezvous is sub-second when pods are
# healthy; minutes only happen when something is wrong. Fail fast and
# let ``ensure_collective_synced``'s retry loop handle the rare slow
# case (the retry creates a FRESH TCPStore on each attempt, so VPC
# routes warm up across retries — by attempt 2 connectivity that took
# 60-90s cold is sub-second).
#
# Tradeoff: a single bad pod blocks the rendezvous for 30s instead
# of 5 min. Across 4 retry attempts that's 2 min worst-case, vs the
# 20 min wedge we hit on the 300s default.
_RENDEZVOUS_TIMEOUT = timedelta(seconds=30)

# Bound NCCL bootstrap (the ncclCommInitRank handshake) at the same 30s.
# In blocking mode (NCCL default) this call BLOCKS in C land and ignores
# Python signals — a peer that dies mid-rendezvous wedges all survivors
# forever. We switch to non-blocking mode here so we can poll
# ``get_async_error`` and surface the failure as a clean Python exception
# the caller's actor can catch and recover from. See RL-412 cascade
# fix: without this, the survivors' Ray actor processes also die when
# one peer is SIGKILL'd during init_collective bootstrap.
_NCCL_BOOTSTRAP_TIMEOUT = timedelta(seconds=30)


class StatelessProcessGroup:
    def __init__(self, master_address: str, port: int, rank: int, world_size: int):
        self.master_address = master_address
        self.port = port
        self.rank = rank
        self.world_size = world_size
        self.tcp_store = torch.distributed.TCPStore(
            host_name=self.master_address,
            port=self.port,
            world_size=self.world_size,
            is_master=(self.rank == 0),
            timeout=_RENDEZVOUS_TIMEOUT,
        )

    def init_nccl_communicator(self, device: int):
        import time as _time

        UNIQUE_ID_KEY = "nccl_unique_id"

        _t = _time.monotonic()
        if self.rank == 0:
            unique_id = get_unique_id()
            unique_id_bytes = unique_id.as_bytes
            # Rank 0: store unique_id to TCPStore
            self.tcp_store.set(UNIQUE_ID_KEY, unique_id_bytes)
        else:
            # Other ranks: get unique_id from TCPStore. The wait is
            # bounded at the same 60s ceiling as the TCPStore rendezvous
            # — if rank 0 hasn't published the key by then, something is
            # wrong on the master side and we want to surface that fast
            # rather than block here for 5 min waiting on a key that
            # will never arrive.
            self.tcp_store.wait([UNIQUE_ID_KEY], _RENDEZVOUS_TIMEOUT)
            unique_id_bytes = self.tcp_store.get(UNIQUE_ID_KEY)
            unique_id = UniqueId.from_bytes(unique_id_bytes)
        _t_unique_id = _time.monotonic() - _t

        with torch.cuda.device(device):
            _t = _time.monotonic()
            # Non-blocking init: ncclCommInitRank in BLOCKING mode (the
            # default) is a synchronous C-land call that ignores Python
            # signals — if a peer dies during bootstrap, the call wedges
            # forever. Non-blocking mode returns immediately with an
            # ``ncclInProgress`` async-state which we poll; on peer
            # death the state flips to a real error code and we abort
            # the half-init'd communicator and raise. This converts a
            # cross-process cascade death into a clean Python exception
            # the caller's actor can catch and respawn from.
            #
            # We can't go through the high-level ``Communicator.init``
            # classmethod here: in non-blocking mode the comm pointer
            # comes back BEFORE the rendezvous completes, and the
            # ``Communicator`` constructor immediately calls
            # ``comm_count`` / ``comm_cu_device`` / ``comm_user_rank``
            # on the still-in-progress pointer, which raises
            # ``NCCLError(InvalidArgument)``. Instead we call the raw
            # binding, poll the async state to completion, and only
            # then wrap into the high-level ``Communicator`` (which can
            # now safely introspect the fully-initialized comm).
            cfg = NCCLConfig(blocking=False)
            comm_ptr = _nccl_bindings.comm_init_rank_scalable(
                int(self.world_size),
                int(self.rank),
                1,
                unique_id.ptr,
                cfg.ptr,
            )
            self._poll_raw_async(
                comm_ptr,
                phase="bootstrap",
                timeout=_NCCL_BOOTSTRAP_TIMEOUT,
            )
            # Bootstrap complete — now safe to wrap with the high-level
            # class and let it introspect the comm.
            self.nccl_communicator = Communicator(comm_ptr)
            _t_comm_init = _time.monotonic() - _t

            _t = _time.monotonic()
            # warmup and check if broadcast is working
            if self.rank == 0:
                data = torch.ones(1, device=device)
            else:
                data = torch.zeros(1, device=device)
            self.broadcast(data, 0)
            # Non-blocking comms return immediately from ``broadcast``;
            # the op may not be enqueued onto the stream yet. Poll the
            # async-error state until the enqueue is observed before
            # syncing the stream — without this the assertion below
            # races against the still-in-progress enqueue.
            self._poll_raw_async(
                self.nccl_communicator._comm,
                phase="warmup_broadcast",
                timeout=_NCCL_BOOTSTRAP_TIMEOUT,
            )
            torch.cuda.current_stream().synchronize()
            assert torch.allclose(data, torch.ones(1, device=device))
            _t_warmup = _time.monotonic() - _t

        _t_total = _t_unique_id + _t_comm_init + _t_warmup
        # Only log on slow inits (>3s).
        if _t_total > 3.0:
            print(
                f"[nccl_init_timing] rank={self.rank}/{self.world_size} "
                f"total={_t_total:.2f}s "
                f"unique_id={_t_unique_id:.2f}s "
                f"comm_init={_t_comm_init:.2f}s "
                f"warmup_bcast={_t_warmup:.2f}s",
                flush=True,
            )

    def _poll_raw_async(
        self,
        comm_ptr: int,
        phase: str,
        timeout: timedelta,
        poll_interval_s: float = 0.05,
    ) -> None:
        """Poll a raw NCCL comm pointer's async-error state.

        Non-blocking ``ncclCommInitRank`` returns immediately with the
        comm in ``ncclInProgress`` state. We poll until it transitions
        to ``Success`` (rendezvous complete) or any other state (peer
        death, NCCL error). On non-success we abort the half-init'd
        comm and raise a Python ``RuntimeError`` so the caller's actor
        can catch + recover instead of being SIGABRT'd or wedging.

        Operates at the bindings level (raw int pointer) because the
        high-level ``Communicator`` wrapper introspects via
        ``comm_count`` etc., which itself fails with
        ``NCCLError(InvalidArgument)`` on an in-progress comm.
        """
        import time as _time

        deadline = _time.monotonic() + timeout.total_seconds()
        success = int(_nccl_bindings.Result.Success)
        in_progress = int(_nccl_bindings.Result.InProgress)
        while True:
            state = int(_nccl_bindings.comm_get_async_error(comm_ptr))
            if state == success:
                return
            if state == in_progress:
                if _time.monotonic() > deadline:
                    self._abort_raw_quietly(comm_ptr)
                    raise RuntimeError(
                        f"NCCL {phase} timed out after "
                        f"{timeout.total_seconds()}s "
                        f"(rank={self.rank}/{self.world_size}) — likely a peer "
                        f"died mid-rendezvous"
                    )
                _time.sleep(poll_interval_s)
                continue
            # Any non-success, non-in-progress state is an error.
            try:
                err_msg = _nccl_bindings.get_last_error(comm_ptr)
            except Exception:  # noqa: BLE001
                err_msg = "<get_last_error failed>"
            self._abort_raw_quietly(comm_ptr)
            raise RuntimeError(
                f"NCCL {phase} failed (rank={self.rank}/{self.world_size}, "
                f"async_state={state}): {err_msg}"
            )

    @staticmethod
    def _abort_raw_quietly(comm_ptr: int) -> None:
        """Best-effort raw comm abort; swallow secondary errors."""
        try:
            _nccl_bindings.comm_abort(comm_ptr)
        except Exception:  # noqa: BLE001
            pass

    def broadcast(
        self, tensor: torch.Tensor, src: int, stream: Optional[torch.cuda.Stream] = None
    ):
        # Run on the caller's current stream by default. Stream isolation
        # used to be enforced here (a dedicated ``_broadcast_stream``
        # contained ``cudaErrorLaunchFailure`` from a dead gen peer so it
        # didn't poison Megatron's training streams) but that's no longer
        # needed: the cross-cluster broadcast now lives in ``RefitWorker``,
        # a sibling Ray actor in a separate process. A peer-death NCCL
        # error poisons RefitWorker's primary CUDA context and we
        # ``ray.kill`` + respawn it; Megatron's process is already
        # isolated by the process boundary. Letting the broadcast run
        # on the caller's stream means the consumer's reads (on the same
        # stream) are naturally ordered after the NCCL kernel — which
        # fixes a silent cross-stream race in ``packed_broadcast_consumer``
        # that was loading uninitialized bytes into model parameters
        # (KL=1.0, rewards=0 on Qwen3-30B-MoE).
        if stream is None:
            stream = torch.cuda.current_stream()
        self.nccl_communicator.broadcast(
            sendbuf=tensor, recvbuf=tensor, root=src, stream=int(stream.cuda_stream)
        )

    def destroy(self):
        """Tear down the NCCL communicator and TCP store so the group can be re-initialized.

        Safe to call even if the communicator was never initialized. Any errors raised by
        the NCCL library (e.g. the peers have already gone away) are swallowed so the
        caller can always bring the group back up with a fresh `init_nccl_communicator`.

        Whole-device sync before aborting drains any in-flight broadcast
        kernels so they finish (or surface their error) before we abort
        the communicator. RefitWorker is the only caller that runs this
        group; if a dead peer queued ``cudaErrorLaunchFailure``, it
        surfaces here and the caller (``RefitWorker.abort_collective``)
        ray.kills the actor — Megatron's process is unaffected because
        we're in a different process.
        """
        try:
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(
                f"[StatelessProcessGroup.destroy] synchronize surfaced "
                f"{type(e).__name__}: {e} — likely a peer died mid-"
                f"broadcast; caller should ray.kill this actor",
                flush=True,
            )
            comm = getattr(self, "nccl_communicator", None)
            if comm is not None:
                for method in ("abort", "destroy", "finalize"):
                    fn = getattr(comm, method, None)
                    if callable(fn):
                        try:
                            fn()
                            break
                        except Exception:
                            continue
                self.nccl_communicator = None  # type: ignore[assignment]
            raise
        comm = getattr(self, "nccl_communicator", None)
        if comm is not None:
            for method in ("abort", "destroy", "finalize"):
                fn = getattr(comm, method, None)
                if callable(fn):
                    try:
                        fn()
                        break
                    except Exception:
                        continue
            self.nccl_communicator = None  # type: ignore[assignment]
        store = getattr(self, "tcp_store", None)
        if store is not None:
            try:
                del self.tcp_store
            except Exception:
                pass
