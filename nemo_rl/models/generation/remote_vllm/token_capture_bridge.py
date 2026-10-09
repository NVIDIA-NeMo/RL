# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP bridge from external vLLM servers to the controller's token staging.

External rollout servers live in a different Ray cluster, so they cannot attach
to the TransferQueue controller actor directly. This bridge keeps the data-plane
client in the SingleController process and exposes only the two operations the
serving plugin needs: resolve a staged prefix and commit one completed model
call. The plugin waits for commit acknowledgement before returning the response,
preserving Gym's stage-before-response ordering.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import threading
import time
from typing import Any
import urllib.error
import urllib.request

import ray
import uvicorn
from fastapi import FastAPI, HTTPException, Request

from nemo_rl.data_plane.token_staging_wire import (
    TokenCaptureAdmission,
    build_staged_token_record,
    commit_coords,
    failed_commit_coords,
)
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource

LOGGER = logging.getLogger(__name__)
WEIGHT_VERSION_PATH = "/v1/nemo-rl/token-capture/weight-version"


def set_remote_token_capture_weight_version(
    *,
    bridge_url: str,
    auth_token: str,
    version: int,
    timeout_s: float = 30.0,
) -> None:
    """Rotate the bridge version without carrying its live server across Ray."""
    if type(version) is not int or version < 0:
        raise ValueError(f"weight version must be a non-negative int, got {version!r}")
    request = urllib.request.Request(
        f"{bridge_url.rstrip('/')}{WEIGHT_VERSION_PATH}",
        data=json.dumps({"weight_version": version}).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response.read()
    except urllib.error.HTTPError as error:
        detail = error.read(2048).decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Token-capture bridge returned HTTP {error.code} for "
            f"{request.full_url}: {detail}"
        ) from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Could not reach token-capture bridge at {request.full_url}: "
            f"{error.reason}"
        ) from error


class RemoteVllmTokenCaptureBridge:
    """Serve token-prefix reads and staged-call commits to external vLLM."""

    def __init__(
        self,
        *,
        dp_client: Any,
        staging_partition: str,
        auth_token: str,
    ) -> None:
        self._source = TQTokenSource(
            dp_client,
            staging_partition=staging_partition,
        )
        self._sink = TQTokenSink(
            dp_client,
            staging_partition=staging_partition,
        )
        self._auth_token = auth_token
        self._weight_version = 0
        self._weight_version_lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.base_url: str | None = None

    def _current_weight_version(self) -> int:
        with self._weight_version_lock:
            return self._weight_version

    def set_weight_version(self, version: int) -> None:
        """Set the policy version stamped on future completed calls."""
        if type(version) is not int or version < 0:
            raise ValueError(
                f"weight version must be a non-negative int, got {version!r}"
            )
        with self._weight_version_lock:
            self._weight_version = version

    def _authorize(self, request: Request) -> None:
        if request.headers.get("Authorization") != f"Bearer {self._auth_token}":
            raise HTTPException(
                status_code=401, detail="invalid token-capture credential"
            )

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/health")
        async def health(request: Request) -> dict[str, Any]:
            self._authorize(request)
            return {
                "status": "ok",
                "weight_version": self._current_weight_version(),
            }

        @app.post("/v1/nemo-rl/token-capture/prefix")
        async def fetch_prefix(
            body: dict[str, Any], request: Request
        ) -> dict[str, Any]:
            self._authorize(request)
            staging_chain = body.get("staging_chain")
            if not isinstance(staging_chain, list) or any(
                not isinstance(key, str) or not key for key in staging_chain
            ):
                raise HTTPException(
                    status_code=400,
                    detail="staging_chain must be a list of non-empty strings",
                )
            try:
                token_ids = await asyncio.to_thread(
                    self._source.fetch_prefix_token_ids,
                    staging_chain,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            return {"prefix_token_ids": token_ids}

        @app.post("/v1/nemo-rl/token-capture/commit")
        async def commit_call(body: dict[str, Any], request: Request) -> dict[str, Any]:
            self._authorize(request)
            try:
                admission_body = body["admission"]
                if not isinstance(admission_body, dict):
                    raise TypeError("admission must be an object")
                admission = TokenCaptureAdmission.from_wire(admission_body)
                prefix_token_ids = [
                    int(value) for value in body.get("prefix_token_ids", [])
                ]
                prompt_token_ids = [int(value) for value in body["prompt_token_ids"]]
                generated_token_ids = [
                    int(value) for value in body["generated_token_ids"]
                ]
                generated_logprobs = [
                    float(value) for value in body["generated_logprobs"]
                ]
                extras = body.get("extras")
                if extras is not None and not isinstance(extras, dict):
                    raise TypeError("extras must be an object or null")
            except (KeyError, TypeError, ValueError) as error:
                raise HTTPException(status_code=400, detail=str(error)) from error

            weight_version = self._current_weight_version()
            try:
                record = build_staged_token_record(
                    admission=admission,
                    prefix_token_ids=prefix_token_ids,
                    prompt_token_ids=prompt_token_ids,
                    generated_token_ids=generated_token_ids,
                    generated_logprobs=generated_logprobs,
                    weight_version=weight_version,
                    extras=extras,
                )
            except (TypeError, ValueError, OverflowError):
                LOGGER.exception(
                    "token capture could not build rollout %s call %s",
                    admission.rollout_id,
                    admission.model_call_id,
                )
                return {
                    "ng_commit_coords": failed_commit_coords(
                        admission,
                        weight_version=weight_version,
                    )
                }
            result = await asyncio.to_thread(self._sink.stage_wire, record)
            if not result.ok:
                LOGGER.warning(
                    "token staging sink rejected rollout %s call %s: %s",
                    admission.rollout_id,
                    admission.model_call_id,
                    result.error,
                )
                coords = failed_commit_coords(
                    admission,
                    weight_version=weight_version,
                )
            else:
                coords = commit_coords(record)
            return {"ng_commit_coords": coords}

        @app.post(WEIGHT_VERSION_PATH)
        async def set_weight_version(
            body: dict[str, Any], request: Request
        ) -> dict[str, Any]:
            self._authorize(request)
            version = body.get("weight_version")
            if not isinstance(version, int) or isinstance(version, bool) or version < 0:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"weight version must be a non-negative int, got {version!r}"
                    ),
                )
            self.set_weight_version(version)
            return {"weight_version": version}

        return app

    def start(self) -> None:
        """Start the bridge on the controller node and wait until it is ready."""
        if self._thread is not None:
            raise RuntimeError("token-capture bridge is already running")
        bind_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        bind_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        bind_socket.bind(("0.0.0.0", 0))
        bind_socket.listen(2048)
        port = int(bind_socket.getsockname()[1])
        node_ip = ray.util.get_node_ip_address().strip("[]")
        server = uvicorn.Server(
            uvicorn.Config(
                self._build_app(),
                host="0.0.0.0",
                port=port,
                log_level="warning",
                access_log=False,
            )
        )
        thread = threading.Thread(
            target=server.run,
            kwargs={"sockets": [bind_socket]},
            name="remote-vllm-token-capture-bridge",
            daemon=True,
        )
        self._socket = bind_socket
        self._server = server
        self._thread = thread
        self.base_url = f"http://{node_ip}:{port}"
        thread.start()
        deadline = time.monotonic() + 30.0
        while not server.started and thread.is_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            self.stop()
            raise RuntimeError("token-capture bridge did not start within 30 seconds")

    def stop(self) -> None:
        """Stop the bridge and release its listening socket."""
        server = self._server
        thread = self._thread
        if server is not None:
            server.should_exit = True
        if thread is not None:
            thread.join(timeout=10.0)
        if self._socket is not None:
            self._socket.close()
        self._socket = None
        self._server = None
        self._thread = None
        self.base_url = None
