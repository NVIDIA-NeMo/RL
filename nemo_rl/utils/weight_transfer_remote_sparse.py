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

"""Shared sparse payload pipeline and S3 control plane for vLLM refit."""

import io
import os
import threading
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import suppress
from functools import cache
from typing import Any, Callable

import requests
import torch
import zstandard
from urllib3.util.retry import Retry

from nemo_rl.utils.packed_tensor import get_target_packed_tensor_size
from nemo_rl.utils.weight_transfer_s3 import S3ObjectStore
from nemo_rl.utils.weight_transfer_sparse_codec import (
    DeltaCompressionTracker,
    NamedTensor,
    TensorBatch,
)

G_VLLM_REFIT_S3_MANIFEST_PATH = "/nemo-rl/refit/s3-manifest"
G_VLLM_REFIT_FLUSH_PATH = "/nemo-rl/refit/flush"
G_VLLM_REFIT_API_KEY_HEADER = "x-nemo-rl-refit-key"
_CONTROL_SESSION_LOCAL = threading.local()


def refit_env_int(name: str, default: int, min_value: int = 1) -> int:
    value = int(os.getenv(name) or default)
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")
    return value


def iter_sparse_weight_chunks(
    tensors: Iterable[NamedTensor], target_bytes: int
) -> Iterator[tuple[TensorBatch, float]]:
    iterator = iter(tensors)
    pending = None
    while True:
        started = time.perf_counter()
        chunk = [pending] if pending is not None else []
        size = pending[1].numel() * pending[1].element_size() if pending else 0
        pending = None
        for item in iterator:
            item_size = item[1].numel() * item[1].element_size()
            if chunk and size + item_size > target_bytes:
                pending = item
                break
            chunk.append(item)
            size += item_size
            if size >= target_bytes:
                break
        export_pull_s = time.perf_counter() - started
        if not chunk:
            return
        yield chunk, export_pull_s


def refit_http_session() -> requests.Session:
    session = getattr(_CONTROL_SESSION_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=64,
            pool_maxsize=64,
            max_retries=Retry(
                total=3,
                backoff_factor=0.25,
                status_forcelist=(500, 502, 503, 504),
                allowed_methods={"POST"},
            ),
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _CONTROL_SESSION_LOCAL.session = session
    return session


@cache
def _get_manifest_s3_store(bucket: str, region: str) -> S3ObjectStore:
    return S3ObjectStore(bucket=bucket, region=region)


def vllm_refit_api_key(api_key_env_var: str | None) -> str | None:
    if not api_key_env_var:
        return None
    token = os.environ.get(api_key_env_var)
    if not token:
        raise RuntimeError(
            "vLLM S3 refit API key env var "
            f"{api_key_env_var!r} is configured but unset or empty."
        )
    return token


def sparse_export_chunk_size(
    delta_tracker: DeltaCompressionTracker,
    transport: str,
) -> int:
    requested = refit_env_int(
        f"NRL_REFIT_{transport.upper()}_EXPORT_CHUNK_BYTES",
        default=(1024 if transport == "zmq" else 256) * 1024**2,
        min_value=1,
    )
    if torch.cuda.is_available():
        requested = min(requested, get_target_packed_tensor_size())
    return min(requested, delta_tracker.sparse_bucket_size_bytes)


@cache
def _executor(key: str, workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"nrl-{key}")


def _require_delta_tracker(
    delta_tracker: DeltaCompressionTracker | None,
) -> DeltaCompressionTracker:
    if delta_tracker is None:
        raise RuntimeError("Remote sparse refit requires delta compression.")
    return delta_tracker


def init_sparse_delta_baseline_from_iterator(
    iterator: Iterable[NamedTensor],
    *,
    delta_tracker: DeltaCompressionTracker | None,
    shard_rank: int = 0,
    shard_count: int = 1,
    transport: str = "s3",
) -> None:
    start_s = time.perf_counter()
    delta_tracker = _require_delta_tracker(delta_tracker)
    export_chunk_size = sparse_export_chunk_size(delta_tracker, transport)

    chunk_count = 0
    export_pull_s = snapshot_s = 0.0
    for chunk_index, (chunk, pull_s) in enumerate(
        iter_sparse_weight_chunks(iterator, export_chunk_size)
    ):
        chunk_count = chunk_index + 1
        export_pull_s += pull_s
        if chunk_index % shard_count != shard_rank:
            continue
        started = time.perf_counter()
        delta_tracker.snapshot_baseline(chunk)
        snapshot_s += time.perf_counter() - started
    print(
        "REFIT_BASELINE_INIT "
        f"event=end chunks={chunk_count} export_pull_s={export_pull_s:.3f} "
        f"snapshot_s={snapshot_s:.3f} "
        f"seconds={time.perf_counter() - start_s:.3f}",
        flush=True,
    )


def stream_sparse_delta_payloads(
    iterator: Iterable[NamedTensor],
    *,
    delta_tracker: DeltaCompressionTracker | None,
    transport: str,
    send_payload: Callable[[bytes, int], dict[str, Any]],
    transfer_workers: int,
    wire_metric: str,
    on_error: Callable[[], None] | None = None,
    shard_rank: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    delta_tracker = _require_delta_tracker(delta_tracker)
    prefix = transport.upper()
    encode_workers = refit_env_int(
        f"NRL_REFIT_{prefix}_ENCODE_WORKERS",
        default=max(2, min(8, os.cpu_count() or 8)),
    )
    pipeline_workers = max(encode_workers, transfer_workers)
    executor = _executor(f"refit-{transport}-pipeline", pipeline_workers)
    encode_slots = threading.Semaphore(encode_workers)
    transfer_slots = threading.Semaphore(transfer_workers)
    export_chunk_size = sparse_export_chunk_size(delta_tracker, transport)

    def process_chunk(chunk: TensorBatch, payload_index: int) -> dict[str, Any] | None:
        with encode_slots:
            started = time.perf_counter()
            payload = delta_tracker.prepare_sparse_delta_payload(chunk)
            encode_s = time.perf_counter() - started
            if not payload[2]:
                return None
            started = time.perf_counter()
            buffer = io.BytesIO()
            torch.save(payload, buffer)
            raw_body = buffer.getvalue()
            serialize_s = time.perf_counter() - started
            started = time.perf_counter()
            body = zstd_compress(raw_body, f"NRL_REFIT_{prefix}_ZSTD_THREADS")
            compress_s = time.perf_counter() - started
        with transfer_slots:
            result = send_payload(body, payload_index)
        result.update(
            body_size=len(body),
            encode_s=encode_s,
            serialize_s=serialize_s,
            compress_s=compress_s,
        )
        return result

    timing: dict[str, float] = {}
    receiver_timing: dict[str, float] = {}
    counts = {"payloads": 0, "wire_bytes": 0}
    chunk_count = 0
    export_pull_s = 0.0
    inflight: set[Any] = set()
    max_inflight = pipeline_workers * 2

    def collect_completed(future: Any) -> None:
        result = future.result()
        if result is None:
            return
        counts["payloads"] += 1
        counts["wire_bytes"] += int(result["body_size"])
        for key, value in result.items():
            if key.endswith("_s"):
                timing[key] = timing.get(key, 0.0) + float(value)
        merge_vllm_refit_receiver_timing(
            receiver_timing, [result["receiver"]], maximum=False
        )

    def drain_completed() -> None:
        completed, _ = wait(inflight, return_when=FIRST_COMPLETED)
        for future in completed:
            inflight.remove(future)
            collect_completed(future)

    payload_index = 0
    stream_start = time.perf_counter()
    try:
        for chunk_index, (chunk, pull_s) in enumerate(
            iter_sparse_weight_chunks(iterator, export_chunk_size)
        ):
            chunk_count = chunk_index + 1
            export_pull_s += pull_s
            if chunk_index % shard_count != shard_rank:
                continue
            if len(inflight) >= max_inflight:
                drain_completed()
            inflight.add(executor.submit(process_chunk, chunk, payload_index))
            payload_index += 1

        while inflight:
            drain_completed()
    except Exception:
        for future in inflight:
            future.cancel()
        wait(inflight)
        if on_error is not None:
            with suppress(Exception):
                on_error()
        raise

    timing = {
        "total_s": time.perf_counter() - stream_start,
        "export_pull_s": export_pull_s,
        **timing,
        "payloads": counts["payloads"],
        "chunks": chunk_count,
        wire_metric: counts["wire_bytes"] / 1e6,
        "pipeline_workers": pipeline_workers,
        "encode_workers": encode_workers,
        "export_chunk_mb": export_chunk_size / 1e6,
        "shard_rank": shard_rank,
        "shard_count": shard_count,
    }
    timing.update(receiver_timing)
    print(
        f"REFIT_{prefix}_TIMING "
        + " ".join(f"{key}={value}" for key, value in timing.items()),
        flush=True,
    )
    return {
        "ok": True,
        "payloads": counts["payloads"],
        "wire_bytes": counts["wire_bytes"],
    }


def stream_sparse_delta_payloads_via_s3_manifest(
    iterator: Iterable[NamedTensor],
    *,
    delta_tracker: DeltaCompressionTracker | None,
    refit_urls: Sequence[str],
    api_key_env_var: str | None = None,
    timeout_s: float = 600.0,
    shard_rank: int = 0,
    shard_count: int = 1,
) -> dict[str, Any]:
    urls = [url.strip().rstrip("/") for url in refit_urls if url.strip()]
    if not urls:
        raise ValueError("At least one vLLM S3 refit URL is required.")
    bucket = os.getenv("NRL_REFIT_S3_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("NRL_REFIT_S3_BUCKET must be set for S3 refit.")
    store = _get_manifest_s3_store(
        bucket,
        os.getenv("NRL_REFIT_S3_REGION", "us-east-1").strip() or "us-east-1",
    )
    endpoint_urls = [f"{url}{G_VLLM_REFIT_S3_MANIFEST_PATH}" for url in urls]
    object_prefix = os.getenv("NRL_REFIT_S3_PREFIX", "nemo-rl-refit").strip("/")
    run_prefix = (
        f"{object_prefix}/{uuid.uuid4().hex}" if object_prefix else uuid.uuid4().hex
    )

    def send_payload(body: bytes, payload_index: int) -> dict[str, Any]:
        key = f"{run_prefix}/{payload_index:06d}.pt"
        started = time.perf_counter()
        store.put_object(key, body)
        s3_put_s = time.perf_counter() - started
        try:
            started = time.perf_counter()
            responses = _post_refit_body_to_endpoint_urls(
                endpoint_urls,
                {"bucket": store.bucket, "region": store.region, "key": key},
                api_key_env_var=api_key_env_var,
                timeout_s=timeout_s,
            )
            manifest_post_s = time.perf_counter() - started
        finally:
            with suppress(Exception):
                store.delete_object(key)
        return {
            "s3_put_s": s3_put_s,
            "manifest_post_s": manifest_post_s,
            "receiver": merge_vllm_refit_receiver_timing({}, responses, maximum=True),
        }

    return stream_sparse_delta_payloads(
        iterator,
        delta_tracker=delta_tracker,
        transport="s3",
        send_payload=send_payload,
        transfer_workers=refit_env_int(
            "NRL_REFIT_S3_UPLOAD_WORKERS",
            default=max(4, min(32, os.cpu_count() or 32)),
        ),
        wire_metric="uploaded_mb",
        on_error=lambda: flush_vllm_refit_urls(
            urls,
            api_key_env_var=api_key_env_var,
            timeout_s=min(timeout_s, 60.0),
        ),
        shard_rank=shard_rank,
        shard_count=shard_count,
    )


def _post_refit_body_to_endpoint_urls(
    endpoint_urls: Sequence[str],
    body: Mapping[str, str],
    *,
    api_key_env_var: str | None,
    timeout_s: float,
) -> list[dict[str, Any]]:
    headers = {}
    if token := vllm_refit_api_key(api_key_env_var):
        headers[G_VLLM_REFIT_API_KEY_HEADER] = token

    def post(url: str) -> dict[str, Any]:
        response = refit_http_session().post(
            url,
            json=body,
            headers=headers,
            timeout=timeout_s,
        )
        result: dict[str, Any] = response.json() if response.content else {}
        if response.status_code >= 400 or result.get("ok") is not True:
            raise RuntimeError(f"vLLM refit failed for {url}: {result}")
        return result

    return list(_executor("refit-fanout", len(endpoint_urls)).map(post, endpoint_urls))


def flush_vllm_refit_urls(
    base_urls: Sequence[str],
    *,
    api_key_env_var: str | None,
    timeout_s: float,
) -> None:
    endpoint_urls = [
        f"{url}{G_VLLM_REFIT_FLUSH_PATH}"
        for url in (url.strip().rstrip("/") for url in base_urls if url.strip())
    ]
    _post_refit_body_to_endpoint_urls(
        endpoint_urls,
        {},
        api_key_env_var=api_key_env_var,
        timeout_s=timeout_s,
    )


def download_s3_refit_payload(
    manifest: Mapping[str, Any],
) -> bytes:
    bucket, region, key = (
        str(manifest[field]) for field in ("bucket", "region", "key")
    )
    return zstd_decompress(_get_manifest_s3_store(bucket, region).get_object(key))


def merge_vllm_refit_receiver_timing(
    result: dict[str, Any],
    timings: Iterable[Mapping[str, Any]],
    *,
    maximum: bool,
) -> dict[str, Any]:
    for timing in timings:
        for key, value in timing.items():
            if key.startswith("receiver_") and key.endswith("_s"):
                number = float(value)
                if key in result:
                    number = (
                        max(float(result[key]), number)
                        if maximum
                        else float(result[key]) + number
                    )
                result[key] = number
    return result


def zstd_compress(raw: bytes, threads_env: str) -> bytes:
    threads = refit_env_int(threads_env, default=0, min_value=0)
    compressors = getattr(_CONTROL_SESSION_LOCAL, "zstd_compressors", None)
    if compressors is None:
        compressors = {}
        _CONTROL_SESSION_LOCAL.zstd_compressors = compressors
    compressor = compressors.get(threads)
    if compressor is None:
        compressor = zstandard.ZstdCompressor(level=1, threads=threads)
        compressors[threads] = compressor
    return compressor.compress(raw)


def zstd_decompress(raw: bytes) -> bytes:
    decompressor = getattr(_CONTROL_SESSION_LOCAL, "zstd_decompressor", None)
    if decompressor is None:
        decompressor = zstandard.ZstdDecompressor()
        _CONTROL_SESSION_LOCAL.zstd_decompressor = decompressor
    return decompressor.decompress(raw)
