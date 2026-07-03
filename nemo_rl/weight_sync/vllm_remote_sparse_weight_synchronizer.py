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
from contextlib import nullcontext, suppress
from typing import Any

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.utils.weight_transfer_remote_sparse import flush_vllm_refit_urls
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


class VllmRemoteSparseWeightSynchronizer(WeightSynchronizer):
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
        self._targets: list[str] = []
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
    ) -> dict[str, float]:
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
                raise RuntimeError(
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
                    self._targets,
                    transfer_id=transfer_id,
                    api_key_env_var=self._api_key_env_var,
                    timeout_s=self._request_timeout_s,
                )
                results = ray.get(refs)
                payloads = sum(result["payloads"] for result in results)
                changed_elements = sum(result["changed_elements"] for result in results)
                total_elements = sum(result["total_elements"] for result in results)
                changed_pct = 100.0 * changed_elements / max(total_elements, 1)
                print(
                    f"REFIT_{self._transport.upper()}_DELTA_CHANGE "
                    f"changed_elements={changed_elements} "
                    f"total_elements={total_elements} "
                    f"changed_pct={changed_pct:.8g}",
                    flush=True,
                )
                candidates = 0
                samples = 0
                exact_mismatches = 0
                mismatches = 0
                abs_sum = 0.0
                max_abs = 0.0
                commit_s = 0.0
                if payloads:
                    started = time.perf_counter()
                    receiver_results = flush_vllm_refit_urls(
                        self._refit_urls,
                        api_key_env_var=self._api_key_env_var,
                        timeout_s=self._request_timeout_s,
                    )
                    candidates = sum(
                        int(result.get("verification_candidates", 0))
                        for result in receiver_results
                    )
                    samples = sum(
                        int(result.get("verification_samples", 0))
                        for result in receiver_results
                    )
                    exact_mismatches = sum(
                        int(result.get("verification_exact_mismatches", 0))
                        for result in receiver_results
                    )
                    mismatches = sum(
                        int(result.get("verification_mismatches", 0))
                        for result in receiver_results
                    )
                    abs_sum = sum(
                        float(result.get("verification_abs_sum", 0.0))
                        for result in receiver_results
                    )
                    max_abs = max(
                        (
                            float(result.get("verification_max_abs", 0.0))
                            for result in receiver_results
                        ),
                        default=0.0,
                    )
                    if candidates or samples:
                        print(
                            f"REFIT_{self._transport.upper()}_DELTA_VERIFY "
                            f"candidates={candidates} samples={samples} "
                            f"exact_mismatches={exact_mismatches} "
                            f"mismatches={mismatches} "
                            f"mean_abs={abs_sum / max(samples, 1):.8g} "
                            f"max_abs={max_abs:.8g}",
                            flush=True,
                        )
                    if mismatches:
                        raise RuntimeError(
                            f"Sparse refit sampled {mismatches} mismatched deltas "
                            f"out of {samples}."
                        )
                    commit_s = time.perf_counter() - started
                    print(
                        f"REFIT_{self._transport.upper()}_GLOBAL_COMMIT "
                        f"transfer_id={transfer_id} payloads={payloads} "
                        f"seconds={commit_s:.3f}",
                        flush=True,
                    )
                succeeded = True
            finally:
                if not succeeded:
                    with suppress(Exception):
                        flush_vllm_refit_urls(
                            self._refit_urls,
                            api_key_env_var=self._api_key_env_var,
                            timeout_s=min(self._request_timeout_s, 60.0),
                        )
                self._baseline_commit_refs = (
                    self._policy.finish_remote_sparse_delta_sync(succeeded)
                )
        self._stale = False
        return {
            "delta/changed_elements": float(changed_elements),
            "delta/total_elements": float(total_elements),
            "delta/changed_pct": changed_pct,
            "delta_verify/candidates": float(candidates),
            "delta_verify/samples": float(samples),
            "delta_verify/exact_mismatches": float(exact_mismatches),
            "delta_verify/mismatches": float(mismatches),
            "delta_verify/mismatch_pct": 100.0 * mismatches / max(samples, 1),
            "delta_verify/mean_abs": abs_sum / max(samples, 1),
            "delta_verify/max_abs": max_abs,
            "transfer/payloads": float(payloads),
            "transfer/global_commit_s": commit_s,
        }

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
        self._targets = self._refit_urls
        if self._transport == "zmq":
            self._targets = self._generation.start_zmq_sparse_refit_relays(
                self._refit_urls
            )
        if not self._refit_urls or not self._targets:
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
        self._targets = []
        self._stale = True
