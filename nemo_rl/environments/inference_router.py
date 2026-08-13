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

import subprocess
import sys
import time
from typing import Literal
from urllib.error import URLError
from urllib.request import urlopen

from pydantic import BaseModel, model_validator

RouterBackend = Literal["vllm_router", "smg"]
RouterPolicy = Literal[
    "random",
    "round_robin",
    "cache_aware",
    "power_of_two",
    "consistent_hash",
    "rendezvous_hash",
    "passthrough",
    "least_load",
    "manual",
    "prefix_hash",
]

_BACKEND_POLICIES = {
    "vllm_router": frozenset(
        {
            "random",
            "round_robin",
            "cache_aware",
            "power_of_two",
            "consistent_hash",
        }
    ),
    "smg": frozenset(
        {
            "random",
            "round_robin",
            "cache_aware",
            "power_of_two",
            "consistent_hash",
            "passthrough",
            "least_load",
            "manual",
            "prefix_hash",
        }
    ),
}


class InferenceRouterConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    backend: RouterBackend = "vllm_router"
    policy: RouterPolicy = "consistent_hash"

    @model_validator(mode="after")
    def validate_backend_policy(self) -> "InferenceRouterConfig":
        if self.policy not in _BACKEND_POLICIES[self.backend]:
            supported = ", ".join(sorted(_BACKEND_POLICIES[self.backend]))
            raise ValueError(
                f"Router backend {self.backend!r} does not support policy "
                f"{self.policy!r}; supported policies: {supported}"
            )
        return self


class InferenceRouterProcess:
    def __init__(
        self,
        worker_base_urls: list[str],
        host: str,
        port: int,
        prometheus_port: int,
        config: InferenceRouterConfig,
    ) -> None:
        self.worker_base_urls = [
            base_url.rstrip("/").removesuffix("/v1") for base_url in worker_base_urls
        ]
        self.host = host
        self.port = port
        self.prometheus_port = prometheus_port
        self.config = config
        self._process: subprocess.Popen | None = None

    @property
    def name(self) -> str:
        return "vLLM Router" if self.config.backend == "vllm_router" else "SMG"

    @property
    def session_affinity_header(self) -> str:
        if self.config.backend == "vllm_router":
            return "X-Session-ID"
        return "X-SMG-Routing-Key"

    @property
    def command(self) -> list[str]:
        if self.config.backend == "vllm_router":
            module = "vllm_router.launch_router"
            policy = self.config.policy
        else:
            module = "smg.launch_router"
            policy = (
                "consistent_hashing"
                if self.config.policy == "consistent_hash"
                else self.config.policy
            )

        return [
            sys.executable,
            "-m",
            module,
            "--worker-urls",
            *self.worker_base_urls,
            "--policy",
            policy,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--prometheus-port",
            str(self.prometheus_port),
        ]

    @property
    def openai_base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    @property
    def readiness_url(self) -> str:
        return f"http://{self.host}:{self.port}/readiness"

    def start(self) -> None:
        if self._process is not None:
            raise RuntimeError(f"{self.name} process has already been started")
        self._process = subprocess.Popen(self.command)

    def wait_until_ready(
        self,
        timeout: float = 600.0,
        poll_interval: float = 1.0,
    ) -> None:
        process = self._process
        if process is None:
            raise RuntimeError(f"{self.name} process has not been started")

        deadline = time.monotonic() + timeout
        while True:
            return_code = process.poll()
            if return_code is not None:
                self._process = None
                raise RuntimeError(
                    f"{self.name} process exited with code {return_code} "
                    "before becoming ready"
                )

            try:
                with urlopen(
                    self.readiness_url,
                    timeout=poll_interval,
                ) as response:
                    if response.status == 200:
                        return
            except (URLError, TimeoutError):
                pass

            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"{self.name} did not become ready within {timeout} seconds"
                )

            time.sleep(poll_interval)

    def stop(self, timeout: float = 10.0) -> None:
        process = self._process
        if process is None:
            return

        self._process = None
        if process.poll() is not None:
            return

        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
