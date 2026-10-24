# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""NCCL refit of an external vLLM deployment via vLLM's weight-transfer engine.

Setup builds one NCCL group: ranks ``[0, train_world_size)`` are the policy
workers, the vLLM ranks follow (``rank_offset = train_world_size``). Per refit::

    pause -> /start_weight_update
          -> /update_weights  ||  policy.broadcast_weights_for_collective
          -> /finish_weight_update(version) -> resume

HTTP calls that pair with a collective run concurrently on a side thread.
Controller deadlines and the policy broadcast watchdog bound failure handling.
After a partial update the synchronizer cannot be reused; serving stays paused.
"""

import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

import ray

from nemo_rl.models.generation.interfaces import CollectiveSenderSpec
from nemo_rl.models.generation.remote_vllm.preflight import validate_single_backend
from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


@dataclass(frozen=True)
class _WeightUpdateInfo:
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]


class RemoteVllmNcclWeightSynchronizer(WeightSynchronizer):
    """Broadcast policy weights to an external vLLM deployment over NCCL."""

    def __init__(
        self,
        policy: Any,
        generation: Any,
        train_cluster: Any,
        refit_timeout_s: Optional[float] = None,
    ) -> None:
        self._policy = policy
        self._generation = generation
        self._train_cluster = train_cluster
        self._refit = generation.remote_config.refit
        self._refit_timeout_s = (
            self._refit.timeout_s
            if refit_timeout_s is None
            else min(refit_timeout_s, self._refit.timeout_s)
        )
        if self._refit_timeout_s <= 0:
            raise ValueError("remote_vllm refit timeout must be positive")
        self._update_info: Optional[_WeightUpdateInfo] = None
        self._next_version = 1
        self._stale = True
        self._failed = False
        self._closed = False

    @property
    def is_stale(self) -> bool:
        return self._stale

    def _check_usable(self) -> None:
        if self._closed:
            raise RuntimeError("External NCCL synchronizer is shut down")
        if self._failed:
            raise RuntimeError(
                "External NCCL refit previously failed; restart the training job "
                "and vLLM deployment before retrying"
            )

    def _check_backend(self) -> None:
        validate_single_backend(
            self._generation.client.health(
                timeout_s=self._generation.remote_config.connect_timeout_s
            )
        )

    def _run_with_http(
        self, futures: list[ray.ObjectRef], call: Callable[[], None], what: str
    ) -> None:
        """Run ``call`` beside the policy collective; fail fast if it fails."""
        deadline = time.monotonic() + self._refit_timeout_s
        # Setup passes the initialized generation/synchronizer graph to a Ray
        # actor. Keep thread pools out of that graph, since they cannot be pickled.
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            http: Future[None] = executor.submit(call)
            pending = list(futures)
            while pending:
                http_error = http.exception() if http.done() else None
                if http_error is not None:
                    raise RuntimeError(
                        f"External vLLM {what} failed: {http_error}"
                    ) from http_error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"External vLLM {what} timed out")
                ready, pending = ray.wait(
                    pending, num_returns=1, timeout=min(1.0, remaining)
                )
                ray.get(ready)
            ray.get(list(futures))
            http.result(timeout=max(0.0, deadline - time.monotonic()))
        except BaseException:
            # A partial receive cannot be retried safely on the same engines.
            # Policy broadcasts also have a worker-side abort watchdog. HTTP
            # calls have socket timeouts; shutdown must not wait on that thread.
            self._failed = True
            raise
        finally:
            # Do not wait for a timed-out control request during failure cleanup.
            executor.shutdown(wait=False, cancel_futures=True)

    def init_communicator(self) -> None:
        self._check_usable()
        if self._update_info is not None:
            raise RuntimeError("External NCCL communicator is already initialized")
        self._check_backend()
        state_dict_info = self._policy.prepare_refit_info()
        if not state_dict_info:
            raise RuntimeError("policy.prepare_refit_info returned no tensors")
        # vLLM's NCCLWeightTransferUpdateInfo; "packed" belongs in init_info.
        update_info = _WeightUpdateInfo(
            names=list(state_dict_info),
            dtype_names=[
                str(dtype).removeprefix("torch.")
                for _, dtype in state_dict_info.values()
            ],
            shapes=[list(shape) for shape, _ in state_dict_info.values()],
        )

        engine_world_size = self._refit.nccl_engine_world_size
        train_world_size = self._train_cluster.world_size()
        world_size = train_world_size + engine_world_size
        ip, port = self._train_cluster.get_master_address_and_port()
        sender = self._generation.get_collective_sender_spec()
        init_info = {
            "master_address": ip,
            "master_port": port,
            "rank_offset": train_world_size,
            "world_size": world_size,
            "packed": True,
            "packed_buffer_size_bytes": sender.buffer_size_bytes,
            "packed_num_buffers": sender.num_buffers,
        }
        futures = self._policy.init_collective(
            ip,
            port,
            world_size,
            train_world_size=train_world_size,
            nccl_peer=sender.nccl_peer,
        )
        self._run_with_http(
            futures,
            lambda: self._generation.client.init_weight_transfer_engine(
                init_info, timeout_s=self._refit_timeout_s
            ),
            "init_weight_transfer_engine",
        )
        self._update_info = update_info

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        self._check_usable()
        if kv_scales is not None:
            raise NotImplementedError(
                "remote_vllm NCCL refit does not send separate KV scales"
            )
        if self._update_info is None:
            raise RuntimeError("init_communicator() must run before sync_weights()")
        update_info = asdict(self._update_info)
        self._stale = True
        self._check_backend()
        version = self._next_version
        self._next_version += 1
        sender = self._generation.get_collective_sender_spec()
        context = (
            timer.time("prepare_for_generation/nccl_refit_remote_vllm")
            if timer is not None
            else nullcontext()
        )
        try:
            with context:
                self._sync_weights(update_info, version, sender)
        except BaseException:
            self._failed = True
            raise

        self._stale = False

    def _sync_weights(
        self, update_info: dict[str, Any], version: int, sender: CollectiveSenderSpec
    ) -> None:
        client = self._generation.client
        self._generation.pause_generation_for_refit(
            clear_cache=True, timeout_s=self._refit_timeout_s
        )
        # Never resume after a partial receive or failed layerwise load.
        client.start_weight_update(timeout_s=self._refit_timeout_s)
        futures = self._policy.broadcast_weights_for_collective(
            refit_timeout_s=self._refit_timeout_s,
            buffer_size_bytes=sender.buffer_size_bytes,
            num_buffers=sender.num_buffers,
            use_legacy_packing=sender.use_legacy_packing,
        )
        self._run_with_http(
            futures,
            lambda: client.update_weights(update_info, timeout_s=self._refit_timeout_s),
            "update_weights",
        )
        client.finish_weight_update(str(version), timeout_s=self._refit_timeout_s)
        self._generation.resume_generation_after_refit(timeout_s=self._refit_timeout_s)

    def shutdown(self) -> None:
        """Close controller resources without resuming or terminating serving.

        NCCL communicators belong to the policy and vLLM GPU processes. Those
        processes must be restarted after a failed update; there is no native
        weight-transfer reset endpoint in the supported vLLM protocol.
        """
        self._closed = True
        self._update_info = None
