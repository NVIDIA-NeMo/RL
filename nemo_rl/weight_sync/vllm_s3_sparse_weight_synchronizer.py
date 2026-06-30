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

"""S3 manifest weight synchronizer for remote non-colocated vLLM refit."""

import time
from contextlib import nullcontext
from typing import Any

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.utils.weight_transfer_s3_manifest import flush_vllm_refit_urls
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


class VllmS3SparseWeightSynchronizer(WeightSynchronizer):
    def __init__(
        self,
        policy: Any,
        generation: Any,
        *,
        api_key_env_var: str | None = None,
        request_timeout_s: float = 600.0,
    ) -> None:
        self._policy = policy
        self._generation = generation
        self._refit_urls: list[str] = []
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
            flush_success = self._generation.invalidate_kv_cache()
            if not flush_success:
                print("vLLM KV cache invalidation failed before S3 weight update.")

            if self._baseline_init_refs is not None:
                ray.get(self._baseline_init_refs)
                self._baseline_init_refs = None
            succeeded = False
            try:
                results = ray.get(
                    self._policy.stream_sparse_weights_via_s3_manifest(
                        self._refit_urls,
                        api_key_env_var=self._api_key_env_var,
                        timeout_s=self._request_timeout_s,
                    )
                )
                payloads = sum(int(result["payloads"]) for result in results)
                if payloads:
                    started = time.perf_counter()
                    flush_vllm_refit_urls(
                        self._refit_urls,
                        api_key_env_var=self._api_key_env_var,
                        timeout_s=self._request_timeout_s,
                    )
                    print(
                        "REFIT_S3_GLOBAL_FLUSH "
                        f"payloads={payloads} "
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
        self._baseline_init_refs = self._policy.init_remote_sparse_delta_baseline()
        self._refit_urls = self._generation.report_refit_server_base_urls()
        if not self._refit_urls:
            raise ValueError(
                "vLLM S3 sparse refit requires expose_http_refit_server=true."
            )
        self._stale = False

    def shutdown(self) -> None:
        for ref in (self._baseline_init_refs or []) + (
            self._baseline_commit_refs or []
        ):
            ray.cancel(ref, force=False)
        self._baseline_init_refs = None
        self._baseline_commit_refs = None
        self._refit_urls = []
        self._stale = True
