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

"""HTTP control-plane utilities shared by remote sparse-refit transports."""

import os
import threading
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Any

import requests
from urllib3.util.retry import Retry

G_VLLM_REFIT_S3_MANIFEST_PATH = "/nemo-rl/refit/s3-manifest"
G_VLLM_REFIT_PREPARE_PATH = "/nemo-rl/refit/prepare"
G_VLLM_REFIT_FLUSH_PATH = "/nemo-rl/refit/flush"
G_VLLM_REFIT_ZMQ_FLUSH_PATH = "/nemo-rl/refit/zmq-flush"
G_VLLM_REFIT_API_KEY_HEADER = "x-nemo-rl-refit-key"
_HTTP_LOCAL = threading.local()
_HTTP_ADAPTER = requests.adapters.HTTPAdapter(
    pool_connections=64,
    pool_maxsize=64,
    max_retries=Retry(
        total=3,
        backoff_factor=0.25,
        status_forcelist=(502, 503, 504),
        allowed_methods={"POST"},
    ),
)


def vllm_refit_endpoints(base_urls: Sequence[str], path: str) -> list[str]:
    return list(
        dict.fromkeys(
            f"{url.strip().rstrip('/')}{path}" for url in base_urls if url.strip()
        )
    )


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


def refit_http_session() -> requests.Session:
    session = getattr(_HTTP_LOCAL, "session", None)
    if session is None:
        session = requests.Session()
        session.mount("http://", _HTTP_ADAPTER)
        session.mount("https://", _HTTP_ADAPTER)
        _HTTP_LOCAL.session = session
    return session


@cache
def _http_executor(workers: int) -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="nrl-refit-http")


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
        try:
            result: dict[str, Any] = response.json() if response.content else {}
        except requests.exceptions.JSONDecodeError:
            result = {}
        if response.status_code >= 400 or result.get("ok") is not True:
            raise RuntimeError(
                f"vLLM refit failed for {url}: HTTP {response.status_code}: "
                f"{response.text[:512]}"
            )
        return result

    pool = executor or _http_executor(len(endpoint_urls))
    return list(pool.map(post, endpoint_urls))


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
