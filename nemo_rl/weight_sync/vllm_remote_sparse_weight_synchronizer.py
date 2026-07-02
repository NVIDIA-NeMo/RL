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

"""Shared S3/ZeroMQ sparse synchronizer for remote non-colocated vLLM refit."""

import time
import uuid
from contextlib import nullcontext
from typing import Any

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.utils.weight_transfer_remote_sparse import flush_vllm_refit_urls
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


class _VllmRemoteSparseWeightSynchronizer(WeightSynchronizer):
    def __init__(
        self,
        policy: Any,
        generation: Any,
        *,
        transport: str,
        api_key_env_var: str | None = None,
        request_timeout_s: float = 600.0,
    ) -> None:
        self._policy = policy
        self._generation = generation
        self._transport = transport
        self._refit_urls: list[str] = []
        self._refit_addresses: list[str] = []
        self._api_key_env_var = api_key_env_var
        self._request_timeout_s = request_timeout_s
        self._stale = True
        self._baseline_init_refs: list[Any] | None = None
        self._baseline_commit_refs: list[Any] | None = None

    def sync_weights(
        self,
        *,
        timer: Timer | None = None,
        kv_scales: dict[str, float] | None = None,
    ) -> None:
        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with timer_context:
            if self._baseline_commit_refs is not None:
                ray.get(self._baseline_commit_refs)
                self._baseline_commit_refs = None
            if not self._generation.invalidate_kv_cache():
                print(
                    f"vLLM KV cache invalidation failed before {self._transport} "
                    "weight update."
                )

            if self._baseline_init_refs is not None:
                ray.get(self._baseline_init_refs)
                self._baseline_init_refs = None
            succeeded = False
            try:
                transfer_id = uuid.uuid4().hex
                refs = self._policy.stream_remote_sparse_weights(
                    self._transport,
                    self._refit_urls
                    if self._transport == "s3"
                    else self._refit_addresses,
                    transfer_id=transfer_id,
                    api_key_env_var=self._api_key_env_var,
                    timeout_s=self._request_timeout_s,
                )
                results = ray.get(refs)
                payloads = sum(int(result["payloads"]) for result in results)
                if payloads:
                    started = time.perf_counter()
                    flush_vllm_refit_urls(
                        self._refit_urls,
                        api_key_env_var=self._api_key_env_var,
                        timeout_s=self._request_timeout_s,
                    )
                    print(
                        f"REFIT_{self._transport.upper()}_GLOBAL_COMMIT "
                        f"transfer_id={transfer_id} payloads={payloads} "
                        f"seconds={time.perf_counter() - started:.3f}",
                        flush=True,
                    )
                succeeded = True
            finally:
                self._baseline_commit_refs = (
                    self._policy.finish_remote_sparse_delta_sync(succeeded)
                )
        self._stale = False

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def init_communicator(self) -> None:
        self._baseline_init_refs = self._policy.init_remote_sparse_delta_baseline(
            self._transport
        )
        self._refit_urls = self._generation.report_refit_server_base_urls()
        if self._transport == "zmq":
            self._refit_addresses = self._generation.start_zmq_sparse_refit_relays()
        if not self._refit_urls or (
            self._transport == "zmq" and not self._refit_addresses
        ):
            raise ValueError(
                f"vLLM {self._transport} sparse refit endpoints are missing."
            )
        self._stale = False

    def shutdown(self) -> None:
        for ref in (self._baseline_init_refs or []) + (
            self._baseline_commit_refs or []
        ):
            ray.cancel(ref, force=False)
        if self._transport == "zmq":
            self._generation.stop_zmq_sparse_refit_relays()
        self._baseline_init_refs = None
        self._baseline_commit_refs = None
        self._refit_urls = []
        self._refit_addresses = []
        self._stale = True


class VllmS3SparseWeightSynchronizer(_VllmRemoteSparseWeightSynchronizer):
    def __init__(self, policy: Any, generation: Any, **kwargs: Any) -> None:
        super().__init__(policy, generation, transport="s3", **kwargs)


class VllmZmqSparseWeightSynchronizer(_VllmRemoteSparseWeightSynchronizer):
    def __init__(self, policy: Any, generation: Any, **kwargs: Any) -> None:
        super().__init__(policy, generation, transport="zmq", **kwargs)
