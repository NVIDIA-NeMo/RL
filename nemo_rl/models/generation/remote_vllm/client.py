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

"""Small stdlib HTTP client for vLLM's stock control endpoints."""

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from nemo_rl.models.generation.remote_vllm.config import RemoteVllmServiceConfig


def service_url(base_url: str, absolute_path: str) -> str:
    """Resolve a control path against the origin of an OpenAI base URL."""
    parsed = urlsplit(base_url)
    return urlunsplit((parsed.scheme, parsed.netloc, absolute_path, "", ""))


class RemoteVllmClient:
    """Synchronous control client used only on setup/refit paths."""

    def __init__(self, config: RemoteVllmServiceConfig) -> None:
        self.config = config

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        env_var = self.config.api_key_env_var
        if env_var is None:
            return headers
        api_key = os.environ.get(env_var)
        if not api_key:
            raise RuntimeError(
                "External rollout API key environment variable "
                f"{env_var!r} is configured but unset or empty"
            )
        headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def request(
        self,
        method: str,
        path: str,
        *,
        query: dict[str, str] | None = None,
        body: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        control: bool = False,
    ) -> bytes:
        base_url = (
            self.config.effective_control_base_url if control else self.config.base_url
        )
        url = service_url(base_url, path)
        if query:
            url = f"{url}?{urllib.parse.urlencode(query)}"
        headers = self._headers()
        payload = None
        if body is not None:
            payload = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            url, data=payload, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout_s or self.config.request_timeout_s
            ) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            detail = error.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(
                f"External vLLM returned HTTP {error.code} for "
                f"{request.full_url}: {detail}"
            ) from error
        except urllib.error.URLError as error:
            raise RuntimeError(
                f"Could not reach external vLLM at {request.full_url}: {error.reason}"
            ) from error

    def get_json(
        self,
        path: str,
        *,
        timeout_s: float | None = None,
        control: bool = False,
    ) -> dict[str, Any]:
        payload = self.request("GET", path, timeout_s=timeout_s, control=control)
        base_url = (
            self.config.effective_control_base_url if control else self.config.base_url
        )
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"External vLLM returned invalid JSON from "
                f"{service_url(base_url, path)}"
            ) from error
        if not isinstance(decoded, dict):
            raise RuntimeError(
                f"External vLLM returned a non-object from "
                f"{service_url(base_url, path)}"
            )
        return decoded

    def health(
        self,
        *,
        timeout_s: float | None = None,
        control: bool = False,
    ) -> dict[str, Any] | None:
        payload = self.request(
            "GET", self.config.health_path, timeout_s=timeout_s, control=control
        )
        if not payload:
            return None
        try:
            decoded = json.loads(payload)
        except (UnicodeDecodeError, json.JSONDecodeError):
            # Stock vLLM does not promise a JSON health body. A successful
            # status code is sufficient when there is no recognized payload.
            return None
        return decoded if isinstance(decoded, dict) else None

    def pause(self, *, mode: str, clear_cache: bool) -> None:
        self.request(
            "POST",
            self.config.pause_path,
            query={
                "mode": mode,
                "clear_cache": str(clear_cache).lower(),
            },
            control=True,
        )

    def reload_weights(self, weights_path: str) -> None:
        self.request(
            "POST",
            self.config.collective_rpc_path,
            body={
                "method": "reload_weights",
                "kwargs": {"weights_path": weights_path},
            },
            timeout_s=self.config.refit.timeout_s,
            control=True,
        )

    def reset_prefix_cache(self) -> None:
        self.request("POST", self.config.reset_prefix_cache_path, control=True)

    def configure_token_capture(self, *, bridge_url: str, auth_token: str) -> None:
        """Configure every external backend to use the controller bridge."""
        self.request(
            "POST",
            "/v1/nemo-rl/token-capture/configure",
            body={"bridge_url": bridge_url, "auth_token": auth_token},
            control=True,
        )

    def resume(self) -> None:
        self.request("POST", self.config.resume_path, control=True)
