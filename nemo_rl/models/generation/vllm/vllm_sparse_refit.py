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

"""Remote sparse-refit receiver lifecycle for vLLM generation workers."""

import asyncio
import io
import os
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Literal, NamedTuple, cast

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GENERATION_PORT_RANGE_HIGH,
    DEFAULT_GENERATION_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.utils import weight_transfer_sparse_codec as sparse_codec
from nemo_rl.utils.weight_transfer_remote_sparse import (
    G_VLLM_REFIT_API_KEY_HEADER,
    G_VLLM_REFIT_FLUSH_PATH,
    G_VLLM_REFIT_PREPARE_PATH,
    G_VLLM_REFIT_S3_MANIFEST_PATH,
    decode_sparse_payload,
    download_s3_refit_payload,
    merge_vllm_refit_metrics,
    refit_env_int,
    vllm_refit_api_key,
)
from nemo_rl.utils.weight_transfer_zmq import (
    G_VLLM_REFIT_CHECKSUM_HEADER,
    G_VLLM_REFIT_PAYLOAD_HEADER,
    G_VLLM_REFIT_PRODUCER_HEADER,
    G_VLLM_REFIT_TRANSFER_HEADER,
    G_VLLM_REFIT_ZMQ_PAYLOAD_PATH,
    ZmqSparseRefitServer,
)


def _decode_staged_payload(
    serialized: bytes,
) -> tuple[sparse_codec.DecodedSparsePayload, int]:
    payload = cast(
        sparse_codec.TensorPayload,
        torch.load(io.BytesIO(serialized), map_location="cpu", weights_only=True),
    )
    return (
        sparse_codec.decode_sparse_tensor_payload_for_staging(payload),
        sum(int(item.get("verification_samples", 0)) for item in payload[2]),
    )


class _StagedSparsePayload(NamedTuple):
    path: str
    started_at: float
    finished_at: float
    deserialize_s: float
    save_s: float
    candidates: int


def _stage_sparse_payload(
    serialized: bytes,
    staging_dir: str,
) -> _StagedSparsePayload:
    started_at = time.perf_counter()
    decoded, candidates = _decode_staged_payload(serialized)
    deserialize_s = time.perf_counter() - started_at
    descriptor, path = tempfile.mkstemp(
        prefix="nemo_rl_refit_", suffix=".pt", dir=staging_dir
    )
    os.close(descriptor)
    started = time.perf_counter()
    try:
        torch.save(decoded, path)
    except Exception:
        os.unlink(path)
        raise
    finished_at = time.perf_counter()
    return _StagedSparsePayload(
        path,
        started_at,
        finished_at,
        deserialize_s,
        finished_at - started,
        candidates,
    )


class VllmSparseRefitReceiver:
    """Own the optional transport server, apply queue, and relay resources."""

    def __init__(self, worker: Any) -> None:
        self._worker = worker
        self._refit_apply_queue_condition = threading.Condition()
        self._refit_apply_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="nrl-vllm-sparse-refit",
        )
        self._refit_apply_futures: list[Future[dict[str, Any]]] = []
        self._refit_apply_pending_payloads: list[
            bytes | Future[_StagedSparsePayload]
        ] = []
        self._refit_seen_payloads: dict[tuple[str, int, int], str] = {}
        self._refit_workers_share_node = False
        self._refit_apply_queue_depth = refit_env_int(
            "NRL_REFIT_APPLY_QUEUE_DEPTH", default=32
        )
        self._refit_apply_batch_size = refit_env_int(
            "NRL_REFIT_APPLY_BATCH_SIZE", default=8
        )
        self._refit_partition_executor = ThreadPoolExecutor(
            max_workers=refit_env_int(
                "NRL_REFIT_PARTITION_WORKERS",
                default=max(2, min(8, os.cpu_count() or 8)),
            ),
            thread_name_prefix="nrl-vllm-sparse-partition",
        )
        self._refit_verification_candidates = 0
        self._refit_batch_staging_dir = (
            os.getenv("NRL_REFIT_BATCH_STAGING_DIR") or "/dev/shm"
        )
        self._refit_http_server: tuple[Any, threading.Thread, str] | None = None
        self._zmq_refit_server: tuple[ZmqSparseRefitServer, str] | None = None
        self._refit_async_loop: asyncio.AbstractEventLoop | None = None

    def set_worker_hostnames(self, hostnames: list[str]) -> None:
        self._refit_workers_share_node = len(set(hostnames)) == 1

    def start_sync_server(self) -> None:
        llm = self._worker.llm
        if llm is None:
            raise RuntimeError("vLLM is not initialized on this worker.")
        self.set_worker_hostnames(llm.collective_rpc("report_node_hostname", args=()))
        self._setup_vllm_refit_server()

    def shutdown(self) -> None:
        self.stop_zmq_sparse_refit_relay()
        if self._refit_http_server is not None:
            self._refit_http_server[0].should_exit = True

        self._flush_queued_sparse_payloads()
        self._refit_apply_executor.shutdown(wait=True)
        self._refit_partition_executor.shutdown(wait=True)

        if self._refit_http_server is not None:
            self._refit_http_server[1].join(timeout=5.0)
            self._refit_http_server = None

    def _enqueue_sparse_payload_apply(
        self,
        payload: bytes,
        payload_key: tuple[str, int, int],
        checksum: str,
    ) -> dict[str, Any]:
        completed: list[Future[dict[str, Any]]] = []
        with self._refit_apply_queue_condition:
            seen_checksum = self._refit_seen_payloads.get(payload_key)
            if seen_checksum is not None:
                if seen_checksum != checksum:
                    raise ValueError(
                        "A sparse refit payload ID was reused with different data."
                    )
                return {"ok": True, "payloads": 0, "duplicate": True}
            while (
                len(self._refit_apply_futures) >= self._refit_apply_queue_depth
                and not self._refit_apply_futures[0].done()
            ):
                self._refit_apply_queue_condition.wait()
            while self._refit_apply_futures and self._refit_apply_futures[0].done():
                completed.append(self._refit_apply_futures.pop(0))
            response = self._collect_refit_apply_results(completed)
            self._refit_seen_payloads[payload_key] = checksum
            pending: bytes | Future[_StagedSparsePayload] = payload
            if self._refit_workers_share_node:
                pending = self._refit_partition_executor.submit(
                    _stage_sparse_payload,
                    payload,
                    self._refit_batch_staging_dir,
                )
            self._refit_apply_pending_payloads.append(pending)
            if len(self._refit_apply_pending_payloads) == self._refit_apply_batch_size:
                self._submit_pending_sparse_payloads()
        return response

    def _submit_pending_sparse_payloads(self) -> Future[dict[str, Any]]:
        payloads = tuple(self._refit_apply_pending_payloads)
        self._refit_apply_pending_payloads.clear()
        apply = (
            self.update_weights_from_staged_sparse_payloads
            if self._refit_workers_share_node
            else self.update_weights_from_serialized_sparse_payloads
        )
        future = self._refit_apply_executor.submit(cast(Any, apply), payloads)
        self._refit_apply_futures.append(future)
        future.add_done_callback(self._notify_refit_apply_waiters)
        return future

    def _notify_refit_apply_waiters(self, _future: Future[dict[str, Any]]) -> None:
        with self._refit_apply_queue_condition:
            self._refit_apply_queue_condition.notify_all()

    def _collect_refit_apply_results(
        self,
        futures: list[Future[dict[str, Any]]],
    ) -> dict[str, Any]:
        results = [future.result() for future in futures]
        return {
            "ok": True,
            "payloads": sum(int(result.get("payloads", 0)) for result in results),
            **merge_vllm_refit_metrics({}, results, maximum=False),
        }

    @staticmethod
    def _refit_collective_response(worker_results: Any) -> dict[str, Any]:
        results = cast(list[dict[str, Any]], worker_results)
        return {
            "ok": True,
            **merge_vllm_refit_metrics(
                {}, results, maximum=True, candidate_maximum=True
            ),
        }

    def _refit_collective_rpc(
        self,
        method: str,
        args: tuple[Any, ...],
    ) -> Any:
        llm = self._worker.llm
        if llm is None:
            raise RuntimeError("vLLM is not initialized on this worker.")
        if not self._worker.cfg["vllm_cfg"]["async_engine"]:
            return llm.collective_rpc(method, args=args)
        if self._refit_async_loop is None:
            raise RuntimeError("The async vLLM refit server loop is not initialized.")
        return asyncio.run_coroutine_threadsafe(
            llm.collective_rpc(method, args=args),
            self._refit_async_loop,
        ).result()

    def update_weights_from_serialized_sparse_payloads(
        self,
        serialized_payloads: tuple[bytes, ...],
    ) -> dict[str, Any]:
        """Apply a FIFO batch of sparse deltas through one collective RPC."""

        def decode_for_rpc(serialized: bytes) -> tuple[bytes, int]:
            decoded, candidates = _decode_staged_payload(serialized)
            buffer = io.BytesIO()
            torch.save(decoded, buffer)
            return buffer.getvalue(), candidates

        decoded_payloads = list(
            self._refit_partition_executor.map(decode_for_rpc, serialized_payloads)
        )
        self._refit_verification_candidates += sum(
            candidates for _, candidates in decoded_payloads
        )
        response = self._refit_collective_response(
            self._refit_collective_rpc(
                "update_weights_from_decoded_sparse_payload",
                tuple(payload for payload, _ in decoded_payloads),
            )
        )
        response["payloads"] = len(serialized_payloads)
        return response

    def update_weights_from_staged_sparse_payloads(
        self,
        staged_payloads: tuple[Future[_StagedSparsePayload], ...],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        staged: list[_StagedSparsePayload] = []
        try:
            stage_error = None
            for future in staged_payloads:
                try:
                    staged.append(future.result())
                except Exception as exc:
                    stage_error = stage_error or exc
            if stage_error is not None:
                raise stage_error
            stage_wait_s = time.perf_counter() - started
            self._refit_verification_candidates += sum(
                payload.candidates for payload in staged
            )
            try:
                response = self._refit_collective_response(
                    self._refit_collective_rpc(
                        "update_weights_from_decoded_sparse_payload_files",
                        tuple(payload.path for payload in staged),
                    )
                )
            except Exception:
                # Drain peers before removing shared batch files.
                self._refit_collective_rpc("synchronize_device", ())
                raise
        finally:
            for payload in staged:
                os.unlink(payload.path)
        worker_total_s = float(response.get("receiver_total_s", 0.0))
        response.update(
            receiver_node_deserialize_s=max(
                (payload.deserialize_s for payload in staged), default=0.0
            ),
            receiver_stage_s=(
                max(payload.finished_at for payload in staged)
                - min(payload.started_at for payload in staged)
            ),
            receiver_stage_save_s=max(
                (payload.save_s for payload in staged), default=0.0
            ),
            receiver_stage_wait_s=stage_wait_s,
        )
        response["receiver_worker_total_s"] = worker_total_s
        response["receiver_total_s"] = time.perf_counter() - started
        response["payloads"] = len(staged_payloads)
        return response

    def _flush_queued_sparse_payloads(self) -> dict[str, Any]:
        started = time.perf_counter()
        with self._refit_apply_queue_condition:
            if self._refit_apply_pending_payloads:
                self._submit_pending_sparse_payloads()
            futures = list(self._refit_apply_futures)
            self._refit_apply_futures.clear()
            self._refit_apply_queue_condition.notify_all()
            payload_count = len(self._refit_seen_payloads)
            batch_count = (
                payload_count + self._refit_apply_batch_size - 1
            ) // self._refit_apply_batch_size
        response = self._collect_refit_apply_results(futures)
        if futures:
            verification = self._refit_collective_response(
                self._refit_collective_rpc("finish_sparse_delta_refit", ())
            )
            if self._refit_verification_candidates:
                verification["verification_candidates"] = (
                    self._refit_verification_candidates
                )
            response.update(verification)
        with self._refit_apply_queue_condition:
            self._refit_seen_payloads.clear()
            self._refit_verification_candidates = 0
        response.update(
            payloads=payload_count,
            batches=batch_count,
            seconds=time.perf_counter() - started,
        )
        if futures:
            print(
                "REFIT_RECEIVER_TIMING "
                f"payloads={payload_count} batches={batch_count} "
                f"total_s={response['seconds']:.3f} "
                f"payload_total_s={response.get('receiver_total_s', 0.0):.3f} "
                f"delta_verify_candidates="
                f"{response.get('verification_candidates', 0)} "
                f"delta_verify_samples={response.get('verification_samples', 0)} "
                f"delta_verify_exact_mismatches="
                f"{response.get('verification_exact_mismatches', 0)} "
                f"delta_verify_mismatches="
                f"{response.get('verification_mismatches', 0)} "
                f"delta_verify_max_abs="
                f"{response.get('verification_max_abs', 0.0):.8g}",
                flush=True,
            )
        return response

    def _prepare_sparse_refit_info(self, request: dict[str, Any]) -> dict[str, Any]:
        started = time.perf_counter()
        state_dict_info = {
            name: (tuple(shape), sparse_codec.dtype_from_name(dtype))
            for name, (shape, dtype) in request["tensors"].items()
        }
        self._refit_collective_rpc(
            "prepare_sparse_delta_refit_info", (state_dict_info,)
        )
        seconds = time.perf_counter() - started
        print(
            f"REFIT_RECEIVER_PREWARM tensors={len(state_dict_info)} "
            f"seconds={seconds:.3f}",
            flush=True,
        )
        return {"ok": True, "tensors": len(state_dict_info), "seconds": seconds}

    async def _apply_s3_manifest_payload(
        self,
        manifest: dict[str, Any],
    ) -> dict[str, Any]:
        started = time.perf_counter()
        body = await asyncio.to_thread(download_s3_refit_payload, manifest)
        download_s = time.perf_counter() - started
        key = str(manifest["key"])
        checksum = str(manifest["checksum"])
        result = await asyncio.to_thread(
            self._enqueue_sparse_payload_apply,
            body,
            (key, -1, -1),
            checksum,
        )
        result["receiver_s3_download_s"] = download_s
        return result

    async def _apply_zmq_payload(self, raw_request: Any) -> dict[str, Any]:
        headers = raw_request.headers
        transfer_id = headers.get(G_VLLM_REFIT_TRANSFER_HEADER, "")
        producer_id = int(headers.get(G_VLLM_REFIT_PRODUCER_HEADER, "-1"))
        payload_id = int(headers.get(G_VLLM_REFIT_PAYLOAD_HEADER, "-1"))
        checksum = headers.get(G_VLLM_REFIT_CHECKSUM_HEADER, "")
        if not transfer_id or producer_id < 0 or payload_id < 0 or not checksum:
            raise ValueError("Missing or invalid ZeroMQ sparse refit payload headers.")
        compressed = await raw_request.body()
        started = time.perf_counter()
        payload = await asyncio.to_thread(
            decode_sparse_payload,
            compressed,
            checksum,
        )
        decode_s = time.perf_counter() - started
        result = await asyncio.to_thread(
            self._enqueue_sparse_payload_apply,
            payload,
            (transfer_id, producer_id, payload_id),
            checksum,
        )
        result["receiver_zmq_decode_s"] = decode_s
        return result

    def setup_api_server(self, app: Any) -> None:
        cfg = self._worker.cfg
        token = vllm_refit_api_key(cfg["vllm_cfg"].get("http_refit_api_key_env_var"))

        async def respond(
            raw_request: Request,
            action: Literal["prepare", "s3", "flush", "zmq"],
        ) -> JSONResponse:
            if cfg["vllm_cfg"]["async_engine"]:
                self._refit_async_loop = asyncio.get_running_loop()
            if (
                token is not None
                and raw_request.headers.get(G_VLLM_REFIT_API_KEY_HEADER) != token
            ):
                return JSONResponse(
                    content={"ok": False, "error": "unauthorized"}, status_code=403
                )
            try:
                if action == "prepare":
                    result = await asyncio.to_thread(
                        self._prepare_sparse_refit_info,
                        await raw_request.json(),
                    )
                elif action == "s3":
                    result = await self._apply_s3_manifest_payload(
                        await raw_request.json()
                    )
                elif action == "zmq":
                    result = await self._apply_zmq_payload(raw_request)
                else:
                    result = await asyncio.to_thread(self._flush_queued_sparse_payloads)
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            return JSONResponse(
                content=result,
                status_code=200 if result.get("ok") is True else 500,
            )

        def endpoint(action: Literal["prepare", "s3", "flush", "zmq"]):
            async def handle(raw_request: Request) -> JSONResponse:
                return await respond(raw_request, action)

            return handle

        for path, action in (
            (G_VLLM_REFIT_S3_MANIFEST_PATH, "s3"),
            (G_VLLM_REFIT_PREPARE_PATH, "prepare"),
            (G_VLLM_REFIT_FLUSH_PATH, "flush"),
            (G_VLLM_REFIT_ZMQ_PAYLOAD_PATH, "zmq"),
        ):
            app.add_api_route(path, endpoint(action), methods=["POST"])

    def report_refit_server_base_url(self) -> str | None:
        if self._refit_http_server is not None:
            return self._refit_http_server[2]
        base_url = getattr(self._worker, "base_url", None)
        return base_url.removesuffix("/v1") if base_url else None

    def start_zmq_sparse_refit_relay(self, refit_urls: list[str]) -> str:
        if self._zmq_refit_server is not None:
            return self._zmq_refit_server[1]
        cfg = self._worker.cfg
        port = cfg["vllm_cfg"].get("zmq_refit_server_port") or _get_free_port_local(
            cfg.get("port_range_low", DEFAULT_GENERATION_PORT_RANGE_LOW),
            cfg.get("port_range_high", DEFAULT_GENERATION_PORT_RANGE_HIGH),
        )
        server = ZmqSparseRefitServer(
            refit_urls,
            bind_address=f"tcp://0.0.0.0:{port}",
            api_key_env_var=cfg["vllm_cfg"].get("http_refit_api_key_env_var"),
            timeout_s=float(os.getenv("NRL_REFIT_ZMQ_TIMEOUT_S") or 600.0),
        )
        server.start()
        address = f"tcp://{_get_node_ip_local()}:{port}"
        self._zmq_refit_server = (server, address)
        print(f"Starting vLLM ZeroMQ refit relay on {address}", flush=True)
        return address

    def stop_zmq_sparse_refit_relay(self) -> None:
        if self._zmq_refit_server is not None:
            self._zmq_refit_server[0].close()
        self._zmq_refit_server = None

    def _setup_vllm_refit_server(self) -> None:
        app = FastAPI()
        self.setup_api_server(app)
        cfg = self._worker.cfg
        port = cfg["vllm_cfg"].get("http_refit_server_port") or _get_free_port_local(
            cfg.get("port_range_low", DEFAULT_GENERATION_PORT_RANGE_LOW),
            cfg.get("port_range_high", DEFAULT_GENERATION_PORT_RANGE_HIGH),
        )
        server = uvicorn.Server(
            uvicorn.Config(
                app,
                host="0.0.0.0",
                port=port,
                timeout_keep_alive=120,
            )
        )
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        base_url = f"http://{_get_node_ip_local()}:{port}"
        self._refit_http_server = (server, thread, base_url)
        print(f"Starting vLLM refit server on {base_url}", flush=True)
