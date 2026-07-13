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

"""Shared sparse payload pipeline and control plane for remote vLLM refit."""

import hashlib
import io
import os
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from typing import Any, Literal, TypedDict
from urllib.parse import quote

import requests
import torch
import zstandard
from urllib3.util.retry import Retry

from nemo_rl.utils.packed_tensor import get_target_packed_tensor_size
from nemo_rl.utils.weight_transfer_sparse_codec import (
    DeltaCompressionTracker,
    NamedTensor,
    TensorBatch,
    TensorPayload,
    merge_sparse_payloads,
)

G_VLLM_REFIT_S3_MANIFEST_PATH = "/nemo-rl/refit/s3-manifest"
G_VLLM_REFIT_PREPARE_PATH = "/nemo-rl/refit/prepare"
G_VLLM_REFIT_FLUSH_PATH = "/nemo-rl/refit/flush"
G_VLLM_REFIT_API_KEY_HEADER = "x-nemo-rl-refit-key"
_CONTROL_SESSION_LOCAL = threading.local()
_S3_PART_SIZE = 64 * 1024**2
_S3_MEMORY_LIMIT = 2 * 1024**3


class SparseDeltaStreamResult(TypedDict):
    payloads: int
    changed_elements: int
    total_elements: int


SparsePartitionMode = Literal["chunks", "names", "none"]


@dataclass
class _SparsePayloadBucket:
    payloads: list[TensorPayload]
    dense_bytes: int = 0
    encode_s: float = 0.0
    next_index: int = 0


@cache
def _s3_client(region: str) -> Any:
    # Keep the AWS runtime unloaded for ZeroMQ-only jobs.
    from awscrt.auth import AwsCredentialsProvider
    from awscrt.io import ClientBootstrap, DefaultHostResolver, EventLoopGroup
    from awscrt.s3 import S3Client, create_default_s3_signing_config

    event_loop_group = EventLoopGroup()
    bootstrap = ClientBootstrap(
        event_loop_group,
        DefaultHostResolver(event_loop_group),
    )
    return S3Client(
        bootstrap=bootstrap,
        region=region,
        signing_config=create_default_s3_signing_config(
            region=region,
            credential_provider=AwsCredentialsProvider.new_default_chain(bootstrap),
        ),
        part_size=_S3_PART_SIZE,
        multipart_upload_threshold=_S3_PART_SIZE,
        throughput_target_gbps=10.0,
        memory_limit=_S3_MEMORY_LIMIT,
    )


class _S3ObjectStore:
    def __init__(self, *, bucket: str, region: str) -> None:
        from awscrt.s3 import S3RequestType

        self.bucket = bucket
        self.region = region
        self._client = _s3_client(region)
        self._request_type = S3RequestType

    def put(self, key: str, body: bytes) -> None:
        self._client.make_request(
            type=self._request_type.PUT_OBJECT,
            request=self._request("PUT", key, body),
        ).finished_future.result()

    def get(self, key: str) -> bytearray:
        from awscrt.http import HttpHeaders

        body = bytearray()

        def on_headers(
            status_code: int,
            headers: list[tuple[str, str]],
            **_kwargs: Any,
        ) -> None:
            nonlocal body
            if status_code != 200:
                raise RuntimeError(f"S3 GET returned HTTP {status_code}.")
            length = HttpHeaders(headers).get("content-length")
            if length is None:
                raise RuntimeError("S3 GET response omitted content-length.")
            body = bytearray(int(length))

        def on_body(chunk: bytes, offset: int, **_kwargs: Any) -> None:
            body[offset : offset + len(chunk)] = chunk

        self._client.make_request(
            type=self._request_type.GET_OBJECT,
            request=self._request("GET", key),
            on_headers=on_headers,
            on_body=on_body,
        ).finished_future.result()
        return body

    def delete(self, key: str) -> None:
        self._client.make_request(
            type=self._request_type.DEFAULT,
            request=self._request("DELETE", key),
            operation_name="DeleteObject",
        ).finished_future.result()

    def _request(self, method: str, key: str, body: bytes | None = None) -> Any:
        from awscrt.http import HttpHeaders, HttpRequest

        headers = HttpHeaders(
            [("host", f"{self.bucket}.s3.{self.region}.amazonaws.com")]
        )
        if body is not None:
            headers.add("content-length", str(len(body)))
            headers.add("content-type", "application/octet-stream")
        elif method == "DELETE":
            headers.add("content-length", "0")
        return HttpRequest(
            method,
            f"/{quote(key, safe='/~')}",
            headers,
            io.BytesIO(body) if body is not None else None,
        )


def refit_env_int(name: str, *, default: int, min_value: int = 1) -> int:
    value = int(os.getenv(name) or default)
    if value < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")
    return value


def sparse_payload_checksum(body: bytes | bytearray) -> str:
    return hashlib.blake2b(body, digest_size=16).hexdigest()


def decode_sparse_payload(body: bytes | bytearray, checksum: str) -> bytes:
    actual = sparse_payload_checksum(body)
    if actual != checksum:
        raise ValueError(
            f"Sparse refit payload checksum mismatch: expected={checksum}, actual={actual}."
        )
    decompressor = getattr(_CONTROL_SESSION_LOCAL, "zstd_decompressor", None)
    if decompressor is None:
        decompressor = zstandard.ZstdDecompressor()
        _CONTROL_SESSION_LOCAL.zstd_decompressor = decompressor
    return decompressor.decompress(body)


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


def sparse_name_shard(name: str, shard_count: int) -> int:
    return (
        int.from_bytes(hashlib.blake2b(name.encode(), digest_size=8).digest())
        % shard_count
    )


def vllm_refit_endpoints(base_urls: Sequence[str], path: str) -> list[str]:
    return list(
        dict.fromkeys(
            f"{url.strip().rstrip('/')}{path}" for url in base_urls if url.strip()
        )
    )


def _partition_sparse_weights_by_name(
    tensors: Iterable[NamedTensor], shard_rank: int, shard_count: int
) -> Iterator[NamedTensor]:
    for name, tensor in tensors:
        if sparse_name_shard(name, shard_count) == shard_rank:
            yield name, tensor


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
                status_forcelist=(502, 503, 504),
                allowed_methods={"POST"},
            ),
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        _CONTROL_SESSION_LOCAL.session = session
    return session


@cache
def _get_manifest_s3_store(bucket: str, region: str) -> _S3ObjectStore:
    return _S3ObjectStore(bucket=bucket, region=region)


def vllm_refit_api_key(api_key_env_var: str | None) -> str | None:
    if not api_key_env_var:
        return None
    token = os.environ.get(api_key_env_var)
    if not token:
        raise RuntimeError(
            "vLLM sparse refit API key env var "
            f"{api_key_env_var!r} is configured but unset or empty."
        )
    return token


def sparse_export_chunk_size(
    delta_tracker: DeltaCompressionTracker,
    transport: str,
) -> int:
    requested = refit_env_int(
        f"NRL_REFIT_{transport.upper()}_EXPORT_CHUNK_BYTES",
        default=256 * 1024**2,
        min_value=1,
    )
    if torch.cuda.is_available():
        requested = min(requested, get_target_packed_tensor_size())
    return min(requested, delta_tracker.sparse_bucket_size_bytes)


@cache
def _executor(key: str, workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"nrl-{key}")


def init_sparse_delta_baseline_from_iterator(
    iterator: Iterable[NamedTensor],
    *,
    delta_tracker: DeltaCompressionTracker,
    shard_rank: int,
    shard_count: int,
    transport: str,
    partition: SparsePartitionMode = "chunks",
) -> None:
    if partition == "names":
        iterator = _partition_sparse_weights_by_name(iterator, shard_rank, shard_count)
    start_s = time.perf_counter()
    export_chunk_size = sparse_export_chunk_size(delta_tracker, transport)

    chunk_count = 0
    export_pull_s = snapshot_s = 0.0
    for chunk_index, (chunk, pull_s) in enumerate(
        iter_sparse_weight_chunks(iterator, export_chunk_size)
    ):
        chunk_count = chunk_index + 1
        export_pull_s += pull_s
        if partition == "chunks" and chunk_index % shard_count != shard_rank:
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
    delta_tracker: DeltaCompressionTracker,
    transport: str,
    send_payload: Callable[[bytes, int], dict[str, Any]],
    transfer_workers: int,
    shard_rank: int,
    shard_count: int,
    partition: SparsePartitionMode = "chunks",
) -> SparseDeltaStreamResult:
    if partition == "names":
        iterator = _partition_sparse_weights_by_name(iterator, shard_rank, shard_count)
    prefix = transport.upper()
    encode_workers = refit_env_int(
        f"NRL_REFIT_{prefix}_ENCODE_WORKERS",
        default=max(2, min(8, os.cpu_count() or 8)),
    )
    encode_executor = _executor(f"refit-{transport}-encode", encode_workers)
    serialize_workers = min(4, encode_workers)
    serialize_executor = _executor(f"refit-{transport}-serialize", serialize_workers)
    transfer_executor = _executor(f"refit-{transport}-transfer", transfer_workers)
    export_chunk_size = sparse_export_chunk_size(delta_tracker, transport)

    def encode_chunk(
        chunk: TensorBatch,
    ) -> tuple[TensorPayload | None, float, int, int, int]:
        dense_bytes = sum(tensor.numel() * tensor.element_size() for _, tensor in chunk)
        started = time.perf_counter()
        payload, changed_elements, total_elements = (
            delta_tracker.prepare_sparse_delta_payload(chunk)
        )
        encode_s = time.perf_counter() - started
        return (
            payload if payload[2] else None,
            encode_s,
            changed_elements,
            total_elements,
            dense_bytes,
        )

    def serialize_payloads(
        payloads: tuple[TensorPayload, ...], encode_s: float
    ) -> tuple[bytes, dict[str, float]]:
        started = time.perf_counter()
        buffer = io.BytesIO()
        torch.save(merge_sparse_payloads(payloads), buffer)
        raw_body = buffer.getvalue()
        serialize_s = time.perf_counter() - started
        started = time.perf_counter()
        body = zstd_compress(raw_body, f"NRL_REFIT_{prefix}_ZSTD_THREADS")
        compress_s = time.perf_counter() - started
        return (
            body,
            {
                "encode_s": encode_s,
                "serialize_s": serialize_s,
                "compress_s": compress_s,
            },
        )

    def transfer_payload(
        encoded: tuple[bytes, dict[str, float]], payload_index: int
    ) -> dict[str, Any]:
        body, encode_timing = encoded
        result = send_payload(body, payload_index)
        result.update(
            body_size=len(body),
            **encode_timing,
        )
        return result

    timing: dict[str, float] = {}
    receiver_timing: dict[str, float] = {}
    counts = {
        "payloads": 0,
        "wire_bytes": 0,
        "changed_elements": 0,
        "total_elements": 0,
    }
    chunk_count = 0
    export_pull_s = 0.0
    encode_inflight: set[Any] = set()
    serialize_inflight: dict[Any, int] = {}
    transfer_inflight: set[Any] = set()
    worker_errors: list[Exception] = []
    max_encode_inflight = encode_workers * 2
    max_serialize_inflight = serialize_workers * 2
    max_transfer_inflight = transfer_workers * 2
    bucket = _SparsePayloadBucket([])

    def collect_transfers(*, block: bool) -> None:
        if not transfer_inflight:
            return
        if block:
            completed, _ = wait(transfer_inflight, return_when=FIRST_COMPLETED)
        else:
            completed = {future for future in transfer_inflight if future.done()}
        for future in completed:
            transfer_inflight.remove(future)
            try:
                result = future.result()
            except Exception as error:
                worker_errors.append(error)
                continue
            counts["payloads"] += 1
            counts["wire_bytes"] += int(result["body_size"])
            for key, value in result.items():
                if key.endswith("_s"):
                    timing[key] = timing.get(key, 0.0) + float(value)
            merge_vllm_refit_metrics(
                receiver_timing, [result["receiver"]], maximum=False
            )

    def collect_serialized(*, block: bool) -> None:
        if not serialize_inflight:
            return
        if block:
            completed, _ = wait(serialize_inflight, return_when=FIRST_COMPLETED)
        else:
            completed = {future for future in serialize_inflight if future.done()}
        for future in completed:
            index = serialize_inflight.pop(future)
            try:
                encoded = future.result()
            except Exception as error:
                worker_errors.append(error)
                continue
            while len(transfer_inflight) >= max_transfer_inflight:
                collect_transfers(block=True)
            transfer_inflight.add(
                transfer_executor.submit(transfer_payload, encoded, index)
            )
        collect_transfers(block=False)

    def submit_bucket() -> None:
        if not bucket.payloads:
            return
        while len(serialize_inflight) >= max_serialize_inflight:
            collect_serialized(block=True)
        serialize_inflight[
            serialize_executor.submit(
                serialize_payloads, tuple(bucket.payloads), bucket.encode_s
            )
        ] = bucket.next_index
        bucket.next_index += 1
        bucket.payloads.clear()
        bucket.dense_bytes = 0
        bucket.encode_s = 0.0
        collect_serialized(block=False)

    def consume_encoded(encoded: Any) -> None:
        payload, encode_s, changed_elements, total_elements, dense_bytes = encoded
        counts["changed_elements"] += changed_elements
        counts["total_elements"] += total_elements
        if payload is None:
            return
        if (
            bucket.payloads
            and bucket.dense_bytes + dense_bytes
            > delta_tracker.sparse_bucket_size_bytes
        ):
            submit_bucket()
        bucket.payloads.append(payload)
        bucket.dense_bytes += dense_bytes
        bucket.encode_s += encode_s
        if bucket.dense_bytes >= delta_tracker.sparse_bucket_size_bytes:
            submit_bucket()

    def drain_encodes() -> None:
        completed, _ = wait(encode_inflight, return_when=FIRST_COMPLETED)
        for future in completed:
            encode_inflight.remove(future)
            try:
                consume_encoded(future.result())
            except Exception as error:
                worker_errors.append(error)

    stream_start = time.perf_counter()
    try:
        for chunk_index, (chunk, pull_s) in enumerate(
            iter_sparse_weight_chunks(iterator, export_chunk_size)
        ):
            chunk_count = chunk_index + 1
            export_pull_s += pull_s
            if partition == "chunks" and chunk_index % shard_count != shard_rank:
                continue
            while len(encode_inflight) >= max_encode_inflight:
                drain_encodes()
            encode_inflight.add(encode_executor.submit(encode_chunk, chunk))

        while encode_inflight:
            drain_encodes()
        submit_bucket()
        while serialize_inflight:
            collect_serialized(block=True)
        while transfer_inflight:
            collect_transfers(block=True)
        if worker_errors:
            raise worker_errors[0]
    except Exception:
        for futures in (encode_inflight, serialize_inflight, transfer_inflight):
            for future in futures:
                future.cancel()
            if futures:
                wait(futures)
        raise

    report = {
        "total_s": time.perf_counter() - stream_start,
        "export_pull_s": export_pull_s,
        **timing,
        "payloads": counts["payloads"],
        "chunks": chunk_count,
        "wire_mb": counts["wire_bytes"] / 1e6,
        "encode_workers": encode_workers,
        "export_chunk_mb": export_chunk_size / 1e6,
        "shard_rank": shard_rank,
        "shard_count": shard_count,
        "partition": partition,
        "changed_elements": counts["changed_elements"],
        "total_elements": counts["total_elements"],
        "changed_pct": 100.0
        * counts["changed_elements"]
        / max(counts["total_elements"], 1),
    }
    report.update(receiver_timing)
    print(
        f"REFIT_{prefix}_TIMING "
        + " ".join(f"{key}={value}" for key, value in report.items()),
        flush=True,
    )
    return {
        "payloads": counts["payloads"],
        "changed_elements": counts["changed_elements"],
        "total_elements": counts["total_elements"],
    }


def stream_sparse_delta_payloads_via_s3_manifest(
    iterator: Iterable[NamedTensor],
    *,
    delta_tracker: DeltaCompressionTracker,
    refit_targets: Sequence[str],
    transfer_id: str,
    api_key_env_var: str | None,
    timeout_s: float,
    shard_rank: int,
    shard_count: int,
    partition: SparsePartitionMode = "chunks",
) -> SparseDeltaStreamResult:
    urls = [url.strip().rstrip("/") for url in refit_targets if url.strip()]
    if not urls:
        raise ValueError("At least one vLLM S3 refit URL is required.")
    bucket = os.getenv("NRL_REFIT_S3_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("NRL_REFIT_S3_BUCKET must be set for S3 refit.")
    store = _get_manifest_s3_store(
        bucket,
        os.getenv("NRL_REFIT_S3_REGION", "us-east-1").strip() or "us-east-1",
    )
    endpoint_urls = vllm_refit_endpoints(urls, G_VLLM_REFIT_S3_MANIFEST_PATH)
    object_prefix = os.getenv("NRL_REFIT_S3_PREFIX", "nemo-rl-refit").strip("/")
    run_prefix = (
        f"{object_prefix}/{transfer_id}/{shard_rank:06d}"
        if object_prefix
        else f"{transfer_id}/{shard_rank:06d}"
    )
    api_key = vllm_refit_api_key(api_key_env_var)

    def send_payload(body: bytes, payload_index: int) -> dict[str, Any]:
        key = f"{run_prefix}/{payload_index:06d}.pt"
        started = time.perf_counter()
        store.put(key, body)
        s3_put_s = time.perf_counter() - started
        try:
            started = time.perf_counter()
            responses = post_vllm_refit_endpoints(
                endpoint_urls,
                {
                    "bucket": store.bucket,
                    "region": store.region,
                    "key": key,
                    "checksum": sparse_payload_checksum(body),
                },
                api_key=api_key,
                timeout_s=timeout_s,
            )
            manifest_post_s = time.perf_counter() - started
        finally:
            with suppress(Exception):
                store.delete(key)
        return {
            "s3_put_s": s3_put_s,
            "manifest_post_s": manifest_post_s,
            "receiver": merge_vllm_refit_metrics({}, responses, maximum=True),
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
        shard_rank=shard_rank,
        shard_count=shard_count,
        partition=partition,
    )


def post_vllm_refit_endpoints(
    endpoint_urls: Sequence[str],
    body: Mapping[str, Any] | bytes,
    *,
    api_key: str | None,
    timeout_s: float,
    headers: Mapping[str, str] | None = None,
    executor: ThreadPoolExecutor | None = None,
) -> list[dict[str, Any]]:
    request_headers = dict(headers or {})
    if api_key:
        request_headers[G_VLLM_REFIT_API_KEY_HEADER] = api_key
    request_kwargs: dict[str, Any] = (
        {"data": body} if isinstance(body, bytes) else {"json": body}
    )

    def post(url: str) -> dict[str, Any]:
        response = refit_http_session().post(
            url,
            **request_kwargs,
            headers=request_headers,
            timeout=timeout_s,
        )
        result: dict[str, Any] = response.json() if response.content else {}
        if response.status_code >= 400 or result.get("ok") is not True:
            raise RuntimeError(f"vLLM refit failed for {url}: {result}")
        return result

    pool = executor or _executor("refit-fanout", len(endpoint_urls))
    futures = [pool.submit(post, url) for url in endpoint_urls]
    return [future.result() for future in futures]


def flush_vllm_refit_urls(
    base_urls: Sequence[str],
    *,
    api_key_env_var: str | None,
    timeout_s: float,
) -> list[dict[str, Any]]:
    return post_vllm_refit_endpoints(
        vllm_refit_endpoints(base_urls, G_VLLM_REFIT_FLUSH_PATH),
        {},
        api_key=vllm_refit_api_key(api_key_env_var),
        timeout_s=timeout_s,
    )


def prepare_vllm_sparse_refit_urls(
    base_urls: Sequence[str],
    state_dict_info: Mapping[str, tuple[tuple[int, ...], torch.dtype]],
    *,
    api_key_env_var: str | None,
    timeout_s: float,
) -> list[dict[str, Any]]:
    tensors = {
        name: [list(shape), str(dtype).removeprefix("torch.")]
        for name, (shape, dtype) in state_dict_info.items()
    }
    return post_vllm_refit_endpoints(
        vllm_refit_endpoints(base_urls, G_VLLM_REFIT_PREPARE_PATH),
        {"tensors": tensors},
        api_key=vllm_refit_api_key(api_key_env_var),
        timeout_s=timeout_s,
    )


def download_s3_refit_payload(
    manifest: Mapping[str, Any],
) -> bytes:
    bucket, region, key, checksum = (
        str(manifest[field]) for field in ("bucket", "region", "key", "checksum")
    )
    body = _get_manifest_s3_store(bucket, region).get(key)
    return decode_sparse_payload(body, checksum)


def merge_vllm_refit_metrics(
    result: dict[str, Any],
    metrics: Iterable[Mapping[str, Any]],
    *,
    maximum: bool,
    candidate_maximum: bool | None = None,
) -> dict[str, Any]:
    for metric in metrics:
        for key, value in metric.items():
            if key.startswith("receiver_") and key.endswith("_s"):
                number, use_maximum = float(value), maximum
            elif candidate_maximum is not None and key.startswith("verification_"):
                number = value
                use_maximum = key == "verification_max_abs" or (
                    key == "verification_candidates" and candidate_maximum
                )
            else:
                continue
            if key in result:
                number = (
                    max(result[key], number) if use_maximum else result[key] + number
                )
            result[key] = number
    return result


def zstd_compress(raw: bytes, threads_env: str) -> bytes:
    compressor = getattr(_CONTROL_SESSION_LOCAL, "zstd_compressor", None)
    if compressor is None:
        compressor = zstandard.ZstdCompressor(
            level=1,
            threads=refit_env_int(threads_env, default=0, min_value=0),
        )
        _CONTROL_SESSION_LOCAL.zstd_compressor = compressor
    return compressor.compress(raw)
