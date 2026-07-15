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

"""Shared sparse payload pipeline and S3 transport for remote vLLM refit."""

import hashlib
import io
import os
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from typing import Any, Protocol
from urllib.parse import quote

import torch
import zstandard

from nemo_rl.utils.packed_tensor import get_target_packed_tensor_size
from nemo_rl.utils.weight_transfer_http import (
    G_VLLM_REFIT_S3_MANIFEST_PATH,
    merge_vllm_refit_metrics,
    post_vllm_refit_endpoints,
    vllm_refit_api_key,
    vllm_refit_endpoints,
)
from nemo_rl.utils.weight_transfer_sparse_codec import (
    DeltaCompressionTracker,
    NamedTensor,
    TensorBatch,
    TensorPayload,
    merge_sparse_payloads,
)

_STREAM_LOCAL = threading.local()
_S3_PART_SIZE = 64 * 1024**2
_S3_MEMORY_LIMIT = 2 * 1024**3


class SparseRefitTransport(Protocol):
    """Payload delivery owned by one generic stream invocation."""

    name: str
    transfer_workers: int

    def send(
        self, body: bytes, payload_id: int, verification_candidates: int
    ) -> dict[str, Any]: ...

    def cleanup(self) -> None:
        """Release resources created on the current transfer worker."""
        ...


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


class _S3ManifestTransport:
    name = "s3"

    def __init__(
        self,
        *,
        store: _S3ObjectStore,
        endpoint_urls: Sequence[str],
        run_prefix: str,
        api_key: str | None,
        timeout_s: float,
        transfer_workers: int,
    ) -> None:
        self._store = store
        self._endpoint_urls = endpoint_urls
        self._run_prefix = run_prefix
        self._api_key = api_key
        self._timeout_s = timeout_s
        self.transfer_workers = transfer_workers
        self._keys = threading.local()

    def send(
        self, body: bytes, payload_id: int, verification_candidates: int
    ) -> dict[str, Any]:
        key = f"{self._run_prefix}/{payload_id:06d}.pt"
        keys = getattr(self._keys, "values", None)
        if keys is None:
            keys = []
            self._keys.values = keys
        keys.append(key)

        try:
            started = time.perf_counter()
            self._store.put(key, body)
            s3_put_s = time.perf_counter() - started
            started = time.perf_counter()
            responses = post_vllm_refit_endpoints(
                self._endpoint_urls,
                {
                    "bucket": self._store.bucket,
                    "region": self._store.region,
                    "key": key,
                    "checksum": sparse_payload_checksum(body),
                    "verification_candidates": verification_candidates,
                },
                api_key=self._api_key,
                timeout_s=self._timeout_s,
            )
            result = {
                "s3_put_s": s3_put_s,
                "manifest_post_s": time.perf_counter() - started,
                "receiver": merge_vllm_refit_metrics({}, responses, maximum=True),
            }
        finally:
            try:
                self._store.delete(key)
            except Exception:
                pass
            else:
                keys.remove(key)
        return result

    def cleanup(self) -> None:
        for key in getattr(self._keys, "values", ()):
            with suppress(Exception):
                self._store.delete(key)
        self._keys.values = []


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
    decompressor = getattr(_STREAM_LOCAL, "zstd_decompressor", None)
    if decompressor is None:
        decompressor = zstandard.ZstdDecompressor()
        _STREAM_LOCAL.zstd_decompressor = decompressor
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


@cache
def _get_manifest_s3_store(bucket: str, region: str) -> _S3ObjectStore:
    return _S3ObjectStore(bucket=bucket, region=region)


def sparse_export_chunk_size(
    delta_tracker: DeltaCompressionTracker,
    transport: str,
) -> int:
    default_mib = 64 if transport == "s3" else 256
    requested = refit_env_int(
        f"NRL_REFIT_{transport.upper()}_EXPORT_CHUNK_BYTES",
        default=default_mib * 1024**2,
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
) -> None:
    start_s = time.perf_counter()
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
    delta_tracker: DeltaCompressionTracker,
    transport: SparseRefitTransport,
    shard_rank: int,
    shard_count: int,
) -> dict[str, int]:
    prefix = transport.name.upper()
    encode_workers = refit_env_int(
        f"NRL_REFIT_{prefix}_ENCODE_WORKERS",
        default=max(2, min(8, os.cpu_count() or 8)),
    )
    encode_executor = _executor(f"refit-{transport.name}-encode", encode_workers)
    serialize_workers = min(4, encode_workers)
    serialize_executor = _executor(
        f"refit-{transport.name}-serialize", serialize_workers
    )
    transfer_executor = ThreadPoolExecutor(
        max_workers=transport.transfer_workers,
        thread_name_prefix=f"nrl-refit-{transport.name}-transfer",
    )
    export_chunk_size = sparse_export_chunk_size(delta_tracker, transport.name)

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
    ) -> tuple[bytes, int, dict[str, float]]:
        started = time.perf_counter()
        buffer = io.BytesIO()
        merged = merge_sparse_payloads(payloads)
        torch.save(merged, buffer)
        raw_body = buffer.getvalue()
        serialize_s = time.perf_counter() - started
        started = time.perf_counter()
        body = zstd_compress(raw_body, f"NRL_REFIT_{prefix}_ZSTD_THREADS")
        compress_s = time.perf_counter() - started
        return (
            body,
            sum(int(item.get("verification_samples", 0)) for item in merged[2]),
            {
                "encode_s": encode_s,
                "serialize_s": serialize_s,
                "compress_s": compress_s,
            },
        )

    def transfer_payload(
        encoded: tuple[bytes, int, dict[str, float]], payload_index: int
    ) -> dict[str, Any]:
        body, verification_candidates, encode_timing = encoded
        result = transport.send(body, payload_index, verification_candidates)
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
    encode_inflight: dict[Any, None] = {}
    serialize_inflight: dict[Any, int] = {}
    transfer_inflight: dict[Any, None] = {}
    transfer_submitted = False
    max_encode_inflight = encode_workers * 2
    max_serialize_inflight = serialize_workers * 2
    max_transfer_inflight = transport.transfer_workers * 2
    bucket = _SparsePayloadBucket([])

    def resolved(inflight: dict[Any, Any], *, block: bool) -> Iterator[tuple[Any, Any]]:
        if not inflight:
            return
        completed = (
            wait(inflight, return_when=FIRST_COMPLETED)[0]
            if block
            else tuple(future for future in inflight if future.done())
        )
        for future in completed:
            metadata = inflight.pop(future)
            yield metadata, future.result()

    def collect_transfers(*, block: bool) -> None:
        for _, result in resolved(transfer_inflight, block=block):
            counts["payloads"] += 1
            counts["wire_bytes"] += int(result["body_size"])
            for key, value in result.items():
                if key.endswith("_s"):
                    timing[key] = timing.get(key, 0.0) + float(value)
            merge_vllm_refit_metrics(
                receiver_timing, [result["receiver"]], maximum=False
            )

    def collect_serialized(*, block: bool) -> None:
        nonlocal transfer_submitted
        for index, encoded in resolved(serialize_inflight, block=block):
            while len(transfer_inflight) >= max_transfer_inflight:
                collect_transfers(block=True)
            transfer_submitted = True
            transfer_inflight[
                transfer_executor.submit(transfer_payload, encoded, index)
            ] = None
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
        for _, encoded in resolved(encode_inflight, block=True):
            consume_encoded(encoded)

    stream_start = time.perf_counter()
    try:
        for chunk_index, (chunk, pull_s) in enumerate(
            iter_sparse_weight_chunks(iterator, export_chunk_size)
        ):
            chunk_count = chunk_index + 1
            export_pull_s += pull_s
            if chunk_index % shard_count != shard_rank:
                continue
            while len(encode_inflight) >= max_encode_inflight:
                drain_encodes()
            encode_inflight[encode_executor.submit(encode_chunk, chunk)] = None

        while encode_inflight:
            drain_encodes()
        submit_bucket()
        while serialize_inflight:
            collect_serialized(block=True)
        while transfer_inflight:
            collect_transfers(block=True)
    except Exception:
        for futures in (encode_inflight, serialize_inflight, transfer_inflight):
            for future in futures:
                future.cancel()
            if futures:
                wait(futures)
        raise
    finally:
        try:
            if transfer_submitted:
                barrier = threading.Barrier(transport.transfer_workers)

                def cleanup_transport(_index: int) -> None:
                    barrier.wait()
                    transport.cleanup()

                list(
                    transfer_executor.map(
                        cleanup_transport, range(transport.transfer_workers)
                    )
                )
        finally:
            transfer_executor.shutdown(wait=True, cancel_futures=True)

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
) -> dict[str, int]:
    endpoint_urls = vllm_refit_endpoints(refit_targets, G_VLLM_REFIT_S3_MANIFEST_PATH)
    if not endpoint_urls:
        raise ValueError("At least one vLLM S3 refit URL is required.")
    bucket = os.getenv("NRL_REFIT_S3_BUCKET", "").strip()
    if not bucket:
        raise RuntimeError("NRL_REFIT_S3_BUCKET must be set for S3 refit.")
    store = _get_manifest_s3_store(
        bucket,
        os.getenv("NRL_REFIT_S3_REGION", "us-east-1").strip() or "us-east-1",
    )
    object_prefix = os.getenv("NRL_REFIT_S3_PREFIX", "nemo-rl-refit").strip("/")
    run_prefix = (
        f"{object_prefix}/{transfer_id}/{shard_rank:06d}"
        if object_prefix
        else f"{transfer_id}/{shard_rank:06d}"
    )
    return stream_sparse_delta_payloads(
        iterator,
        delta_tracker=delta_tracker,
        transport=_S3ManifestTransport(
            store=store,
            endpoint_urls=endpoint_urls,
            run_prefix=run_prefix,
            api_key=vllm_refit_api_key(api_key_env_var),
            timeout_s=timeout_s,
            transfer_workers=refit_env_int(
                "NRL_REFIT_S3_UPLOAD_WORKERS",
                default=max(4, min(32, os.cpu_count() or 32)),
            ),
        ),
        shard_rank=shard_rank,
        shard_count=shard_count,
    )


def download_s3_refit_payload(
    manifest: Mapping[str, Any],
) -> bytes:
    body = _get_manifest_s3_store(str(manifest["bucket"]), str(manifest["region"])).get(
        str(manifest["key"])
    )
    return decode_sparse_payload(body, str(manifest["checksum"]))


def zstd_compress(raw: bytes, threads_env: str) -> bytes:
    compressor = getattr(_STREAM_LOCAL, "zstd_compressor", None)
    if compressor is None:
        compressor = zstandard.ZstdCompressor(
            level=1,
            threads=refit_env_int(threads_env, default=0, min_value=0),
        )
        _STREAM_LOCAL.zstd_compressor = compressor
    return compressor.compress(raw)
