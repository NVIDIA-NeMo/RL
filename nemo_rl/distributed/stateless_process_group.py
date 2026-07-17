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
from datetime import timedelta
from typing import Optional

import torch
from nccl import bindings as _nccl_bindings
from nccl.core.communicator import Communicator, NCCLConfig
from nccl.core.utils import UniqueId, get_unique_id


# Bound TCPStore rendezvous + key-wait at 30s instead of the 300s default.
# Rendezvous is sub-second when healthy; fail fast and let the retry loop handle
# slow cases (each retry creates a fresh TCPStore, so cold routes warm up across
# retries). Env-overridable via NRL_RENDEZVOUS_TIMEOUT_S.
_RENDEZVOUS_TIMEOUT = timedelta(
    seconds=float(os.environ.get("NRL_RENDEZVOUS_TIMEOUT_S", "30"))
)

# Bound NCCL bootstrap at 30s. Blocking mode (NCCL default) wedges all
# survivors forever if a peer dies mid-rendezvous. Non-blocking mode lets
# us poll ``get_async_error`` and surface failures as clean Python exceptions.
_NCCL_BOOTSTRAP_TIMEOUT = timedelta(seconds=30)

# Backstop timeout for waiting on an in-flight weight-broadcast collective.
# The primary failure signal is the comm's async-error state; this only
# fires if NCCL never reports the death (e.g. a peer GPU hang with the
# socket still alive).
_BROADCAST_SYNC_TIMEOUT = timedelta(
    seconds=float(os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "60"))
)



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
            # Other ranks: wait for rank 0 to publish the key, bounded by the rendezvous timeout.
            self.tcp_store.wait([UNIQUE_ID_KEY], _RENDEZVOUS_TIMEOUT)
            unique_id_bytes = self.tcp_store.get(UNIQUE_ID_KEY)
            unique_id = UniqueId.from_bytes(unique_id_bytes)
        _t_unique_id = _time.monotonic() - _t

        with torch.cuda.device(device):
            _t = _time.monotonic()
            # Non-blocking init: poll async state to completion rather than blocking in C.
            # We use the raw binding because the high-level Communicator constructor
            # introspects the comm immediately, which fails on an in-progress pointer.
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
            # Poll async-error state until the enqueue is observed before syncing —
            # without this, the assertion below races against the in-progress enqueue.
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

    def _poll_until_done(
        self,
        done_fn,
        comm_ptr: int,
        phase: str,
        timeout: timedelta,
        poll_interval_s: float = 0.005,
    ) -> None:
        """Block until ``done_fn()`` is truthy, watching the comm's async-error state.

        On peer death or timeout, aborts the comm and raises so the caller can
        recover rather than wedging forever.
        """
        import time as _time

        deadline = _time.monotonic() + timeout.total_seconds()
        success = int(_nccl_bindings.Result.Success)
        in_progress = int(_nccl_bindings.Result.InProgress)
        while not done_fn():
            state = int(_nccl_bindings.comm_get_async_error(comm_ptr))
            if state != success and state != in_progress:
                try:
                    err_msg = _nccl_bindings.get_last_error(comm_ptr)
                except Exception:  # noqa: BLE001
                    err_msg = "<get_last_error failed>"
                self._abort_raw_quietly(comm_ptr)
                raise RuntimeError(
                    f"NCCL {phase} failed (rank={self.rank}/{self.world_size}, "
                    f"async_state={state}): {err_msg} — peer likely died mid-collective"
                )
            if _time.monotonic() > deadline:
                self._abort_raw_quietly(comm_ptr)
                raise RuntimeError(
                    f"NCCL {phase} timed out after {timeout.total_seconds()}s "
                    f"(rank={self.rank}/{self.world_size}) — peer likely died mid-collective"
                )
            _time.sleep(poll_interval_s)

    def synchronize_or_abort(
        self,
        stream: Optional[torch.cuda.Stream] = None,
        timeout: Optional[timedelta] = None,
    ) -> None:
        """Interruptible replacement for ``stream.synchronize()`` on a stream
        carrying this group's NCCL collectives.

        Polls a CUDA completion event together with the comm's async-error
        state. On peer death or timeout, aborts the comm (unblocking the hung
        kernel) and raises so ``update_weights_from_collective`` can return
        and the engine RPC thread is freed for the recovery ``init_collective``.
        """
        if stream is None:
            stream = torch.cuda.current_stream()
        comm = getattr(self, "nccl_communicator", None)
        if comm is None:
            stream.synchronize()
            return
        if timeout is None:
            timeout = _BROADCAST_SYNC_TIMEOUT
        event = torch.cuda.Event()
        event.record(stream)
        self._poll_until_done(
            done_fn=event.query,
            comm_ptr=comm._comm,
            phase="broadcast_sync",
            timeout=timeout,
        )

    def broadcast(
        self, tensor: torch.Tensor, src: int, stream: Optional[torch.cuda.Stream] = None
    ):
        # Run on the caller's current stream so consumer reads are naturally ordered
        # after the NCCL kernel, avoiding a cross-stream race on model parameters.
        if stream is None:
            stream = torch.cuda.current_stream()
        self.nccl_communicator.broadcast(
            sendbuf=tensor, recvbuf=tensor, root=src, stream=int(stream.cuda_stream)
        )

    def destroy(self):
        """Tear down the NCCL communicator and TCP store so the group can be re-initialized.

        Safe to call even if the communicator was never initialized. Whole-device sync
        drains in-flight broadcast kernels before aborting; errors are surfaced to the
        caller so it can decide whether to kill and respawn the actor.
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
