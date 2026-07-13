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
from collections import defaultdict
from contextlib import nullcontext, suppress
from typing import Any

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.utils.weight_transfer_remote_sparse import (
    flush_vllm_refit_urls,
    merge_vllm_refit_metrics,
    prepare_vllm_sparse_refit_urls,
)
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

_REMOTE_SPARSE_TRANSPORTS = {
    "vllm_s3_sparse": "s3",
    "vllm_zmq_sparse": "zmq",
}


def validate_vllm_remote_sparse_refit(
    config: Any,
    *,
    colocated: bool,
    megatron_enabled: bool,
) -> str | None:
    """Validate the optional config and return its internal transport name."""
    transport = config.get("refit_transport")
    if transport is not None and transport not in _REMOTE_SPARSE_TRANSPORTS:
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
    return None if transport is None else _REMOTE_SPARSE_TRANSPORTS[transport]


class VllmRemoteSparseWeightSynchronizer(WeightSynchronizer):
    def __init__(
        self,
        policy: Any,
        generation: Any,
        *,
        transport: str,
        api_key_env_var: str | None = None,
        request_timeout_s: float = 600.0,
        baseline_init_refs: list[Any] | None = None,
    ) -> None:
        self._policy = policy
        self._generation = generation
        self._transport = transport
        self._api_key_env_var = api_key_env_var
        self._request_timeout_s = request_timeout_s
        self._refit_urls: list[str] = []
        self._targets: list[str] = []
        self._baseline_init_refs = list(baseline_init_refs or ())
        self._baseline_commit_refs: list[Any] = []
        self._stale = True

    def sync_weights(
        self,
        *,
        timer: Timer | None = None,
        kv_scales: dict[str, float] | None = None,
    ) -> dict[str, float]:
        context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer
            else nullcontext()
        )
        with context:
            if self._baseline_commit_refs:
                ray.get(self._baseline_commit_refs)
                self._baseline_commit_refs.clear()
            if not self._generation.invalidate_kv_cache():
                raise RuntimeError(
                    f"vLLM KV cache invalidation failed before {self._transport} "
                    "weight update."
                )
            if self._baseline_init_refs:
                ray.get(self._baseline_init_refs)
                self._baseline_init_refs.clear()

            succeeded = False
            try:
                transfer_id = uuid.uuid4().hex
                results = ray.get(
                    self._run_policy_workers(
                        "stream_remote_sparse_weights",
                        transport=self._transport,
                        targets=self._targets,
                        transfer_id=transfer_id,
                        api_key_env_var=self._api_key_env_var,
                        timeout_s=self._request_timeout_s,
                    )
                )
                payloads = sum(result["payloads"] for result in results)
                changed = sum(result["changed_elements"] for result in results)
                total = sum(result["total_elements"] for result in results)
                changed_pct = 100.0 * changed / max(total, 1)
                print(
                    f"REFIT_{self._transport.upper()}_DELTA_CHANGE "
                    f"changed_elements={changed} total_elements={total} "
                    f"changed_pct={changed_pct:.8g}",
                    flush=True,
                )

                verification: defaultdict[str, float] = defaultdict(float)
                commit_s = 0.0
                if payloads:
                    started = time.perf_counter()
                    verification.update(
                        merge_vllm_refit_metrics(
                            {},
                            flush_vllm_refit_urls(
                                self._refit_urls,
                                api_key_env_var=self._api_key_env_var,
                                timeout_s=self._request_timeout_s,
                            ),
                            maximum=True,
                            candidate_maximum=False,
                        )
                    )
                    candidates = int(verification["verification_candidates"])
                    samples = int(verification["verification_samples"])
                    exact = int(verification["verification_exact_mismatches"])
                    mismatches = int(verification["verification_mismatches"])
                    abs_sum = float(verification["verification_abs_sum"])
                    max_abs = float(verification["verification_max_abs"])
                    if candidates or samples:
                        print(
                            f"REFIT_{self._transport.upper()}_DELTA_VERIFY "
                            f"candidates={candidates} samples={samples} "
                            f"exact_mismatches={exact} mismatches={mismatches} "
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
                        "finish_remote_sparse_delta_sync", succeeded=succeeded
                    )
                )

        self._stale = False
        samples = int(verification["verification_samples"])
        mismatches = int(verification["verification_mismatches"])
        metrics = {
            "delta/changed_elements": float(changed),
            "delta/total_elements": float(total),
            "delta/changed_pct": changed_pct,
            "delta_verify/mismatch_pct": 100.0 * mismatches / max(samples, 1),
            "delta_verify/mean_abs": float(verification["verification_abs_sum"])
            / max(samples, 1),
            "transfer/payloads": float(payloads),
            "transfer/global_commit_s": commit_s,
        }
        metrics.update(
            {
                f"delta_verify/{key}": float(verification[f"verification_{key}"])
                for key in (
                    "candidates",
                    "samples",
                    "exact_mismatches",
                    "mismatches",
                    "max_abs",
                )
            }
        )
        return metrics

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def _run_policy_workers(self, method_name: str, **kwargs: Any) -> list[Any]:
        workers = self._policy.worker_group
        count = len(workers.workers)
        return workers.run_all_workers_multiple_data(
            method_name,
            common_kwargs={**kwargs, "shard_count": count},
            shard_rank=list(range(count)),
        )

    def _run_generation_workers(self, method_name: str, **kwargs: Any) -> list[Any]:
        workers = self._generation.worker_group
        return ray.get(
            workers.run_all_workers_single_data(
                method_name,
                run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
                **kwargs,
            )
        )

    @staticmethod
    def start_baseline(policy: Any, transport: str) -> list[Any]:
        workers = policy.worker_group
        count = len(workers.workers)
        return workers.run_all_workers_multiple_data(
            "init_remote_sparse_delta_baseline",
            common_kwargs={"transport": transport, "shard_count": count},
            shard_rank=list(range(count)),
        )

    @staticmethod
    def _merge_refit_info(parts: list[dict[str, Any]]) -> dict[str, Any]:
        merged = {}
        for part in parts:
            for name, info in part.items():
                if name in merged and merged[name] != info:
                    raise ValueError(f"Conflicting sparse refit metadata for {name!r}.")
                merged[name] = info
        return merged

    def init_communicator(self) -> None:
        if not self._baseline_init_refs:
            self._baseline_init_refs = self.start_baseline(
                self._policy, self._transport
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
        state_dict_info = self._merge_refit_info(
            ray.get(list(self._baseline_init_refs))
        )
        self._baseline_init_refs.clear()
        prepare_vllm_sparse_refit_urls(
            self._refit_urls,
            state_dict_info,
            api_key_env_var=self._api_key_env_var,
            timeout_s=self._request_timeout_s,
        )
        self._stale = False

    def shutdown(self) -> None:
        for ref in self._baseline_init_refs + self._baseline_commit_refs:
            ray.cancel(ref, force=False)
        if self._transport == "zmq":
            self._run_generation_workers("stop_zmq_sparse_refit_relay")
        self._baseline_init_refs.clear()
        self._baseline_commit_refs.clear()
        self._refit_urls.clear()
        self._targets.clear()
        self._stale = True
