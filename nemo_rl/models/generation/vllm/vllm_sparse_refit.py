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
import hmac
import logging
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from functools import cache
from typing import Any, Literal, NamedTuple, cast

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GENERATION_PORT_RANGE_HIGH,
    DEFAULT_GENERATION_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.models.generation.vllm.config import VllmRefitConfig
from nemo_rl.utils import weight_transfer_sparse_codec as sparse_codec
from nemo_rl.utils.weight_transfer_http import (
    G_VLLM_REFIT_API_KEY_HEADER,
    G_VLLM_REFIT_FLUSH_PATH,
    G_VLLM_REFIT_PREPARE_PATH,
    G_VLLM_REFIT_S3_MANIFEST_PATH,
    G_VLLM_REFIT_ZMQ_FLUSH_PATH,
    merge_vllm_refit_metrics,
    vllm_refit_api_key,
)
from nemo_rl.utils.weight_transfer_stream import (
    decode_sparse_payload,
    download_s3_refit_payload,
)
from nemo_rl.utils.weight_transfer_zmq import (
    ZmqSparseRefitServer,
)

logger = logging.getLogger(__name__)


@cache
def _warn_unauthenticated_refit_server(transport: str) -> None:
    logger.warning(
        "%s sparse-refit server is binding 0.0.0.0 without an API key; "
        "weight-write endpoints are reachable from the network.",
        transport,
    )


class _StagedSparsePayload(NamedTuple):
    path: str
    started_at: float
    finished_at: float
    save_s: float


def _stage_sparse_payload(
    serialized: bytes,
    staging_dir: str,
) -> _StagedSparsePayload:
    started_at = time.perf_counter()
    descriptor, path = tempfile.mkstemp(
        prefix="nemo_rl_refit_", suffix=".pt", dir=staging_dir
    )
    started = time.perf_counter()
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
        save_s = time.perf_counter() - started
    except Exception:
        os.unlink(path)
        raise
    finished_at = time.perf_counter()
    return _StagedSparsePayload(
        path,
        started_at,
        finished_at,
        save_s,
    )


class VllmSparseRefitReceiver:
    """Own the optional transport server, apply queue, and relay resources."""

    def __init__(self, worker: Any) -> None:
        self._worker = worker
        self._refit_config = VllmRefitConfig.model_validate(
            worker.cfg.get("refit_cfg") or {}
        )
        tuning = self._refit_config.tuning
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
        self._refit_apply_queue_depth = tuning.apply_queue_depth
        self._refit_apply_batch_size = tuning.apply_batch_size
        self._refit_partition_executor = ThreadPoolExecutor(
            max_workers=tuning.partition_workers,
            thread_name_prefix="nrl-vllm-sparse-partition",
        )
        self._refit_verification_candidates = 0
        self._refit_batch_staging_dir = self._refit_config.storage.staging_dir
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
        verification_candidates: int = 0,
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
            self._refit_verification_candidates += verification_candidates
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

    def _submit_pending_sparse_payloads(self) -> None:
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
        response = self._refit_collective_response(
            self._refit_collective_rpc(
                "update_weights_from_decoded_sparse_payload",
                serialized_payloads,
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
            try:
                response = self._refit_collective_response(
                    self._refit_collective_rpc(
                        "update_weights_from_decoded_sparse_payload",
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
            receiver_node_deserialize_s=0.0,
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
        worker_results = self._refit_collective_rpc(
            "prepare_sparse_delta_refit_info", (state_dict_info,)
        )
        overwrite_names = sorted({name for names in worker_results for name in names})
        seconds = time.perf_counter() - started
        print(
            f"REFIT_RECEIVER_PREWARM tensors={len(state_dict_info)} "
            f"overwrite_tensors={len(overwrite_names)} seconds={seconds:.3f}",
            flush=True,
        )
        return {
            "ok": True,
            "tensors": len(state_dict_info),
            "overwrite_names": overwrite_names,
            "seconds": seconds,
        }

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
            int(manifest["verification_candidates"]),
        )
        result["receiver_s3_download_s"] = download_s
        return result

    def _apply_zmq_payload(
        self, compressed: bytes, metadata: Mapping[str, Any]
    ) -> dict[str, Any]:
        started = time.perf_counter()
        checksum = str(metadata["checksum"])
        payload = decode_sparse_payload(compressed, checksum)
        decode_s = time.perf_counter() - started
        result = self._enqueue_sparse_payload_apply(
            payload,
            (
                str(metadata["transfer_id"]),
                int(metadata["producer_id"]),
                int(metadata["payload_id"]),
            ),
            checksum,
            int(metadata["verification_candidates"]),
        )
        result["receiver_zmq_decode_s"] = decode_s
        return result

    def setup_api_server(self, app: Any) -> None:
        cfg = self._worker.cfg
        token = vllm_refit_api_key(cfg["vllm_cfg"].get("http_refit_api_key_env_var"))

        async def respond(
            raw_request: Request,
            action: Literal["prepare", "s3", "flush", "zmq_flush"],
        ) -> JSONResponse:
            if cfg["vllm_cfg"]["async_engine"]:
                self._refit_async_loop = asyncio.get_running_loop()
            supplied_token = raw_request.headers.get(G_VLLM_REFIT_API_KEY_HEADER)
            if token is not None and (
                supplied_token is None or not hmac.compare_digest(token, supplied_token)
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
                elif action == "zmq_flush":
                    body = await raw_request.json()
                    result = await asyncio.to_thread(
                        self.flush_zmq_sparse_refit_relay,
                        str(body["transfer_id"]),
                        int(body.get("expected_payloads", 0)),
                    )
                else:
                    result = await asyncio.to_thread(self._flush_queued_sparse_payloads)
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            return JSONResponse(
                content=result,
                status_code=200 if result.get("ok") is True else 500,
            )

        def endpoint(
            action: Literal["prepare", "s3", "flush", "zmq_flush"],
        ):
            async def handle(raw_request: Request) -> JSONResponse:
                return await respond(raw_request, action)

            return handle

        for path, action in (
            (G_VLLM_REFIT_S3_MANIFEST_PATH, "s3"),
            (G_VLLM_REFIT_PREPARE_PATH, "prepare"),
            (G_VLLM_REFIT_FLUSH_PATH, "flush"),
            (G_VLLM_REFIT_ZMQ_FLUSH_PATH, "zmq_flush"),
        ):
            app.add_api_route(path, endpoint(action), methods=["POST"])

    def report_refit_server_base_url(self) -> str | None:
        if self._refit_http_server is not None:
            return self._refit_http_server[2]
        base_url = getattr(self._worker, "base_url", None)
        return base_url.removesuffix("/v1") if base_url else None

    def start_zmq_sparse_refit_relay(self) -> str:
        if self._zmq_refit_server is not None:
            return self._zmq_refit_server[1]
        cfg = self._worker.cfg
        port = cfg["vllm_cfg"].get("zmq_refit_server_port") or _get_free_port_local(
            cfg.get("port_range_low", DEFAULT_GENERATION_PORT_RANGE_LOW),
            cfg.get("port_range_high", DEFAULT_GENERATION_PORT_RANGE_HIGH),
        )
        server = ZmqSparseRefitServer(
            self._apply_zmq_payload,
            bind_address=f"tcp://0.0.0.0:{port}",
            api_key_env_var=cfg["vllm_cfg"].get("http_refit_api_key_env_var"),
            timeout_s=self._refit_config.request_timeout_s,
            tuning=self._refit_config.tuning,
        )
        if (
            vllm_refit_api_key(cfg["vllm_cfg"].get("http_refit_api_key_env_var"))
            is None
        ):
            _warn_unauthenticated_refit_server("ZeroMQ")
        server.start()
        address = f"tcp://{_get_node_ip_local()}:{port}"
        self._zmq_refit_server = (server, address)
        print(f"Starting vLLM ZeroMQ refit relay on {address}", flush=True)
        return address

    def configure_zmq_sparse_refit_relay(self, relay_addresses: list[str]) -> None:
        if self._zmq_refit_server is None:
            raise RuntimeError("ZeroMQ sparse refit relay is not running.")
        server, own_address = self._zmq_refit_server
        server.configure_tree(
            relay_addresses,
            own_address=own_address,
        )

    def stop_zmq_sparse_refit_relay(self) -> None:
        if self._zmq_refit_server is not None:
            self._zmq_refit_server[0].close()
        self._zmq_refit_server = None

    def flush_zmq_sparse_refit_relay(
        self, transfer_id: str, expected_payloads: int = 0
    ) -> dict[str, Any]:
        if self._zmq_refit_server is None:
            raise RuntimeError("ZeroMQ sparse refit relay is not running.")
        return self._zmq_refit_server[0].flush(transfer_id, expected_payloads)

    def _setup_vllm_refit_server(self) -> None:
        app = FastAPI()
        self.setup_api_server(app)
        cfg = self._worker.cfg
        port = cfg["vllm_cfg"].get("http_refit_server_port") or _get_free_port_local(
            cfg.get("port_range_low", DEFAULT_GENERATION_PORT_RANGE_LOW),
            cfg.get("port_range_high", DEFAULT_GENERATION_PORT_RANGE_HIGH),
        )
        if (
            vllm_refit_api_key(cfg["vllm_cfg"].get("http_refit_api_key_env_var"))
            is None
        ):
            _warn_unauthenticated_refit_server("HTTP")
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
