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
from urllib.error import URLError
from urllib.request import urlopen

from pydantic import BaseModel


class VllmRouterConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    policy: str = "consistent_hash"


class VllmRouterProcess:
    def __init__(
        self,
        worker_base_urls: list[str],
        host: str,
        port: int,
        prometheus_port: int,
        config: VllmRouterConfig,
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
    def command(self) -> list[str]:
        return [
            sys.executable,
            "-m",
            "vllm_router.launch_router",
            "--worker-urls",
            *self.worker_base_urls,
            "--policy",
            self.config.policy,
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
            raise RuntimeError("vLLM Router process has already been started")
        self._process = subprocess.Popen(self.command)

    def wait_until_ready(
        self,
        timeout: float = 600.0,
        poll_interval: float = 1.0,
    ) -> None:
        process = self._process
        if process is None:
            raise RuntimeError("vLLM Router process has not been started")

        deadline = time.monotonic() + timeout
        while True:
            return_code = process.poll()
            if return_code is not None:
                self._process = None
                raise RuntimeError(
                    f"vLLM Router process exited with code {return_code} "
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
                    f"vLLM Router did not become ready within {timeout} seconds"
                )

            time.sleep(poll_interval)

    def stop(self, timeout: float = 5.0) -> None:
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
