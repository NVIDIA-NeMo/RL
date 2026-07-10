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


def validate_vllm_remote_sparse_refit(
    config: Any,
    *,
    colocated: bool,
    megatron_enabled: bool,
) -> str | None:
    """Validate the optional transport without exposing its rules to GRPO."""
    transport = config.get("refit_transport")
    if transport not in (None, "vllm_s3_sparse", "vllm_zmq_sparse"):
        raise ValueError(f"Unsupported vLLM refit transport {transport!r}.")
    vllm_cfg = config["vllm_cfg"]
    if transport is not None and (
        colocated
        or not megatron_enabled
        or vllm_cfg["precision"] == "fp8"
        or vllm_cfg["kv_cache_dtype"].startswith("fp8")
        or not config.get("delta_compression")
        or config.get("quant_cfg")
        or config.get("real_quant")
    ):
        raise ValueError(
            f"{transport} requires a non-colocated Megatron policy, BF16/FP16 "
            "vLLM, delta compression, and an unquantized rollout."
        )
    return transport


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
                refs = self._run_policy_workers(
                    "stream_remote_sparse_weights",
                    transport=self._transport,
                    targets=self._targets,
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
                    self._policy.worker_group.run_all_workers_single_data(
                        "finish_remote_sparse_delta_sync",
                        succeeded=succeeded,
                    )
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

    def _run_policy_workers(self, method_name: str, **kwargs: Any) -> list[Any]:
        worker_group = self._policy.worker_group
        worker_count = len(worker_group.workers)
        return worker_group.run_all_workers_multiple_data(
            method_name,
            common_kwargs={**kwargs, "shard_count": worker_count},
            shard_rank=list(range(worker_count)),
        )

    def _run_generation_workers(self, method_name: str, **kwargs: Any) -> list[Any]:
        worker_group = self._generation.worker_group
        if not worker_group or not worker_group.workers:
            raise RuntimeError("vLLM worker group is not initialized.")
        return ray.get(
            worker_group.run_all_workers_single_data(
                method_name,
                run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
                **kwargs,
            )
        )

    def init_communicator(self) -> None:
        self._baseline_init_refs = self._run_policy_workers(
            "init_remote_sparse_delta_baseline",
            transport=self._transport,
        )
        self._refit_urls = [
            url
            for url in self._run_generation_workers("report_refit_server_base_url")
            if url
        ]
        self._targets = self._refit_urls
        if self._transport == "zmq":
            self._targets = [
                address
                for address in self._run_generation_workers(
                    "start_zmq_sparse_refit_relay", refit_urls=self._refit_urls
                )
                if address
            ]
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
            self._run_generation_workers("stop_zmq_sparse_refit_relay")
        self._baseline_init_refs = None
        self._baseline_commit_refs = None
        self._refit_urls = []
        self._targets = []
        self._stale = True
