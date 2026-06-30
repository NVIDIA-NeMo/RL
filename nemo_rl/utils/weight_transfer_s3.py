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

"""AWS CRT object transport for S3 refit payloads."""

import io
from functools import cache
from typing import Any
from urllib.parse import quote

_PART_SIZE = 64 * 1024**2
_MEMORY_LIMIT = 2 * 1024**3


@cache
def _s3_client(region: str) -> Any:
    from awscrt.auth import AwsCredentialsProvider
    from awscrt.io import ClientBootstrap, DefaultHostResolver, EventLoopGroup
    from awscrt.s3 import S3Client, create_default_s3_signing_config

    event_loop_group = EventLoopGroup()
    bootstrap = ClientBootstrap(
        event_loop_group,
        DefaultHostResolver(event_loop_group),
    )
    credentials = AwsCredentialsProvider.new_default_chain(bootstrap)
    return S3Client(
        bootstrap=bootstrap,
        region=region,
        signing_config=create_default_s3_signing_config(
            region=region,
            credential_provider=credentials,
        ),
        part_size=_PART_SIZE,
        multipart_upload_threshold=_PART_SIZE,
        throughput_target_gbps=10.0,
        memory_limit=_MEMORY_LIMIT,
    )


class S3ObjectStore:
    """Blocking object operations backed by CRT's asynchronous S3 client."""

    def __init__(self, *, bucket: str, region: str) -> None:
        from awscrt.s3 import S3RequestType

        self.bucket = bucket
        self.region = region
        self._client = _s3_client(region)
        self._request_type = S3RequestType

    def put_object(self, key: str, body: bytes) -> None:
        request = self._request("PUT", key, body)
        self._client.make_request(
            type=self._request_type.PUT_OBJECT,
            request=request,
        ).finished_future.result()

    def get_object(self, key: str) -> bytes:
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
        return bytes(body)

    def delete_object(self, key: str) -> None:
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
