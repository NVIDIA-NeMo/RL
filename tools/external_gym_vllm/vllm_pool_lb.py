#!/usr/bin/env python3
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

"""Lightweight async reverse proxy for an external Gym vLLM backend pool.

Reads the backend registry file and forwards OpenAI-compatible requests
to healthy backends using least-outstanding-requests routing.

Supports:
  - Dynamic backend discovery (re-reads registry every few seconds)
  - Health checks (GET /health on each backend)
  - Automatic failover when a backend goes down
  - Single URL for Gym integration

Usage:
    python vllm_pool_lb.py --port 8080 --registry-dir /path/to/state --group-id default

The load balancer exposes:
    http://<host>:8080/v1/...   →  proxied to backends
    http://<host>:8080/health   →  LB health + backend summary
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

MAX_RSS_MB = 4096
SHUTDOWN_TIMEOUT_SECONDS = 120

print(f"[LB] Python: {sys.executable}", flush=True)
try:
    import aiohttp
    from aiohttp import web
except Exception as e:
    print(
        f"ERROR: Failed to import aiohttp: {type(e).__name__}: {e} (python={sys.executable})",
        flush=True,
    )
    import traceback

    traceback.print_exc()
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("vllm_pool_lb")


# HTTP statuses that indicate the upstream backend is sick (crashed EngineCore,
# overloaded, etc). We fail over to a different backend and quarantine the sick
# one until the next health probe clears it.
RETRYABLE_UPSTREAM_STATUSES = {500, 502, 503, 504}


def _read_current_rss_mb() -> float | None:
    """Read the process's current resident memory from procfs."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError):
        log.exception("Failed to read current RSS from /proc/self/status")
    return None


class UpstreamRetryableStatus(Exception):
    """Raised from _proxy_once when a non-streaming upstream returned a 5xx we want to retry."""

    def __init__(self, status: int, body: bytes, headers: dict[str, str]) -> None:
        super().__init__(f"upstream status {status}")
        self.status = status
        self.body = body
        self.headers = headers


class Backend:
    __slots__ = ("job_id", "host", "port", "healthy", "inflight", "last_check")

    def __init__(self, job_id: str, host: str, port: int) -> None:
        self.job_id = job_id
        self.host = host
        self.port = port
        self.healthy = True
        self.inflight = 0
        self.last_check = 0.0

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def __repr__(self) -> str:
        status = "UP" if self.healthy else "DOWN"
        return f"Backend({self.job_id}, {self.host}:{self.port}, {status}, inflight={self.inflight})"


class BackendPool:
    """Manages the set of backends, reads registry, performs health checks."""

    def __init__(
        self, registry_dir: str, group_id: str, health_interval: float = 5.0
    ) -> None:
        self.registry_file = Path(registry_dir) / f".registry_{group_id}"
        self.health_interval = health_interval
        self.backends: dict[str, Backend] = {}  # job_id -> Backend
        self._session: aiohttp.ClientSession | None = None
        self._running = False

    async def start(self) -> None:
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5))
        self._running = True
        asyncio.create_task(self._refresh_loop())
        asyncio.create_task(self._health_check_loop())

    async def stop(self) -> None:
        self._running = False
        if self._session:
            await self._session.close()

    def _read_registry(self) -> dict[str, tuple[str, int]] | None:
        """Read registry file. Returns {job_id: (host, port)}."""
        result: dict[str, tuple[str, int]] = {}
        if not self.registry_file.exists():
            return result
        try:
            lines = self.registry_file.read_text().strip().splitlines()
        except (OSError, UnicodeError) as e:
            log.warning("Failed to read registry: %s", e)
            return None

        for line in lines:
            parts = line.split()
            if len(parts) >= 5 and parts[4] == "ready":
                try:
                    result[parts[0]] = (parts[1], int(parts[2]))
                except ValueError:
                    log.warning("Skipping malformed registry entry: %s", line)
        return result

    async def _refresh_loop(self) -> None:
        """Periodically re-read the registry to discover new/removed backends."""
        while self._running:
            try:
                registered = self._read_registry()
                if registered is not None:
                    # Add new backends
                    for job_id, (host, port) in registered.items():
                        if job_id not in self.backends:
                            b = Backend(job_id, host, port)
                            self.backends[job_id] = b
                            log.info("Discovered new backend: %s", b)
                    # Remove gone backends
                    gone = set(self.backends) - set(registered)
                    for job_id in gone:
                        b = self.backends.pop(job_id)
                        log.info("Removed backend: %s", b)
            except Exception as e:
                log.warning("Refresh error: %s", e)
            await asyncio.sleep(self.health_interval)

    async def _check_backend_health(self, backend: Backend) -> None:
        session = self._session
        if session is None:
            return
        try:
            async with session.get(f"{backend.base_url}/health") as response:
                backend.healthy = response.status == 200
        except Exception:
            backend.healthy = False
        backend.last_check = time.time()

    async def _health_check_loop(self) -> None:
        """Periodically check /health on each backend."""
        while self._running:
            await asyncio.gather(
                *(
                    self._check_backend_health(backend)
                    for backend in list(self.backends.values())
                )
            )
            rss_mb = _read_current_rss_mb()
            if rss_mb is not None and rss_mb > MAX_RSS_MB:
                log.error(
                    "Current RSS %.0f MB exceeds %d MB; stopping new requests and "
                    "draining in-flight requests before restart",
                    rss_mb,
                    MAX_RSS_MB,
                )
                self._running = False
                os.kill(os.getpid(), signal.SIGTERM)
                return
            await asyncio.sleep(self.health_interval)

    def pick(
        self, exclude: set[str] | None = None, affinity_key: str | None = None
    ) -> Backend | None:
        """Pick a healthy backend.

        If affinity_key is set, use consistent hashing to prefer the same backend
        for requests with the same prefix (enables vLLM prefix caching).
        Falls back to least-outstanding-requests if the preferred backend is
        excluded or unhealthy.
        """
        exclude = exclude or set()
        healthy = [
            b for b in self.backends.values() if b.healthy and b.job_id not in exclude
        ]
        if not healthy:
            return None

        if affinity_key and len(healthy) > 1:
            # Consistent hash: sort by hash(affinity_key + job_id) to get a
            # stable preference order. Pick the first one (preferred), but if
            # it's heavily loaded compared to the least-loaded, fall back.
            h = hashlib.md5(affinity_key.encode()).hexdigest()
            ranked = sorted(
                healthy, key=lambda b: hashlib.md5((h + b.job_id).encode()).hexdigest()
            )
            preferred = ranked[0]
            least_loaded = min(healthy, key=lambda b: b.inflight)
            # Use preferred backend unless it has 2x+ more inflight than the
            # least loaded — avoids hotspots when one prefix dominates.
            if preferred.inflight <= least_loaded.inflight * 2 + 10:
                return preferred
            return least_loaded

        return min(healthy, key=lambda b: b.inflight)

    def summary(self) -> list[dict[str, str | bool | int]]:
        return [
            {
                "job_id": b.job_id,
                "url": b.base_url,
                "healthy": b.healthy,
                "inflight": b.inflight,
            }
            for b in self.backends.values()
        ]


class LoadBalancer:
    def __init__(self, pool: BackendPool, port: int) -> None:
        self.pool = pool
        self.port = port
        self._proxy_session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        # limit=5000: enough for 26 backends × ~50 concurrent reqs, but capped
        # to prevent unbounded memory growth that caused OOM at 31GB with limit=0.
        self._proxy_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=5000, limit_per_host=150),
            timeout=aiohttp.ClientTimeout(total=1800),
        )
        await self.pool.start()

    async def stop(self) -> None:
        await self.pool.stop()
        if self._proxy_session:
            await self._proxy_session.close()

    async def handle_health(self, request: web.Request) -> web.Response:
        backends = self.pool.summary()
        healthy_count = sum(1 for b in backends if b["healthy"])
        return web.json_response(
            {
                "status": "ok" if healthy_count > 0 else "no_healthy_backends",
                "healthy_backends": healthy_count,
                "total_backends": len(backends),
                "backends": backends,
            }
        )

    async def _proxy_once(
        self,
        backend: Backend,
        request_method: str,
        path_qs: str,
        headers: dict[str, str],
        body: bytes,
        request: web.Request,
    ) -> web.StreamResponse:
        """Attempt to proxy a single request to one backend."""
        target_url = f"{backend.base_url}{path_qs}"
        backend.inflight += 1
        try:
            if self._proxy_session is None:
                raise RuntimeError("Load balancer has not been started")
            async with self._proxy_session.request(
                method=request_method,
                url=target_url,
                headers=headers,
                data=body,
            ) as upstream_resp:
                content_type = upstream_resp.headers.get("Content-Type", "")
                is_streaming = "text/event-stream" in content_type

                if is_streaming:
                    response = web.StreamResponse(
                        status=upstream_resp.status,
                        headers={
                            "Content-Type": "text/event-stream",
                            "Cache-Control": "no-cache",
                            "Connection": "keep-alive",
                        },
                    )
                    await response.prepare(request)
                    iterator = upstream_resp.content.iter_any().__aiter__()
                    while True:
                        try:
                            chunk = await anext(iterator)
                        except StopAsyncIteration:
                            break
                        except Exception as e:
                            backend.healthy = False
                            log.warning(
                                "Upstream stream from backend %s failed after response "
                                "headers were sent; returning the partial response without "
                                "retrying: %s: %s",
                                backend,
                                type(e).__name__,
                                e,
                            )
                            break
                        try:
                            await response.write(chunk)
                        except Exception as e:
                            log.info(
                                "Downstream disconnected while streaming from backend %s: "
                                "%s: %s",
                                backend,
                                type(e).__name__,
                                e,
                            )
                            return response
                    try:
                        await response.write_eof()
                    except Exception as e:
                        log.info(
                            "Could not finish downstream stream from backend %s: %s: %s",
                            backend,
                            type(e).__name__,
                            e,
                        )
                    return response
                resp_body = await upstream_resp.read()
                resp_headers = {
                    k: v
                    for k, v in upstream_resp.headers.items()
                    if k.lower()
                    not in ("transfer-encoding", "content-encoding", "content-length")
                }
                if upstream_resp.status in RETRYABLE_UPSTREAM_STATUSES:
                    raise UpstreamRetryableStatus(
                        status=upstream_resp.status,
                        body=resp_body,
                        headers=resp_headers,
                    )
                return web.Response(
                    status=upstream_resp.status,
                    headers=resp_headers,
                    body=resp_body,
                )
        finally:
            backend.inflight -= 1
        raise RuntimeError("Upstream request exited without producing a response")

    @staticmethod
    def _extract_affinity_key(body: bytes) -> str | None:
        """Extract a prefix-affinity key from the request body.

        For chat completions, hash the messages array (the shared prompt prefix).
        This ensures requests with the same prompt go to the same backend,
        maximizing vLLM prefix cache hits.
        """
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                return None
            messages = data.get("messages")
            if messages:
                # Hash messages content only (not metadata/response pairs which vary)
                return hashlib.md5(
                    json.dumps(messages, sort_keys=True).encode()
                ).hexdigest()
        except (json.JSONDecodeError, TypeError, UnicodeError):
            pass
        return None

    async def handle_proxy(self, request: web.Request) -> web.StreamResponse:
        """Forward request to a backend, retrying on a different backend if one fails."""
        body = await request.read()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "transfer-encoding")
        }

        # Extract affinity key for prefix-cache-aware routing
        affinity_key = (
            self._extract_affinity_key(body) if request.method == "POST" else None
        )

        tried: set[str] = set()
        last_error: Exception | None = None
        last_upstream_5xx: UpstreamRetryableStatus | None = None

        # Try up to MAX_RETRIES different healthy backends. `pool.pick` filters
        # on `healthy` and excludes anything in `tried`, so the loop also
        # terminates early if every non-quarantined backend has been attempted.
        # The per-attempt aiohttp ClientTimeout bounds the worst-case latency
        # of a wedged backend.
        MAX_RETRIES = 5
        for attempt in range(1, MAX_RETRIES + 1):
            backend = self.pool.pick(exclude=tried, affinity_key=affinity_key)
            if backend is None:
                log.warning(
                    "[proxy %s %s] no more healthy untried backends after %d attempt(s); giving up",
                    request.method,
                    request.path_qs,
                    attempt - 1,
                )
                break

            tried.add(backend.job_id)
            try:
                resp = await self._proxy_once(
                    backend,
                    request.method,
                    request.path_qs,
                    headers,
                    body,
                    request,
                )
                if attempt > 1:
                    log.warning(
                        "[proxy %s %s] succeeded on attempt %d/%d via backend %s after failing on %s",
                        request.method,
                        request.path_qs,
                        attempt,
                        MAX_RETRIES,
                        backend,
                        sorted(tried - {backend.job_id}),
                    )
                return resp
            except UpstreamRetryableStatus as e:
                log.warning(
                    "[proxy %s %s] attempt %d/%d: backend %s returned %d, "
                    "failing over without changing backend health. Body: %s",
                    request.method,
                    request.path_qs,
                    attempt,
                    MAX_RETRIES,
                    backend,
                    e.status,
                    e.body[:500],
                )
                last_upstream_5xx = e
            except Exception as e:
                log.warning(
                    "[proxy %s %s] attempt %d/%d: backend %s raised %s: %s, "
                    "quarantining and failing over",
                    request.method,
                    request.path_qs,
                    attempt,
                    MAX_RETRIES,
                    backend,
                    type(e).__name__,
                    e,
                )
                backend.healthy = False
                last_error = e

        # All retries exhausted. If the last failure was an upstream 5xx with a real
        # response body, forward that body verbatim so callers see the true error.
        # Otherwise synthesize a 502/503.
        if last_upstream_5xx is not None:
            log.error(
                "[proxy %s %s] all %d attempt(s) failed; returning last upstream status %d. "
                "Tried backends: %s",
                request.method,
                request.path_qs,
                len(tried),
                last_upstream_5xx.status,
                sorted(tried),
            )
            return web.Response(
                status=last_upstream_5xx.status,
                headers=last_upstream_5xx.headers,
                body=last_upstream_5xx.body,
            )
        log.error(
            "[proxy %s %s] all %d attempt(s) failed with connection errors; "
            "last_error=%s. Tried backends: %s",
            request.method,
            request.path_qs,
            len(tried),
            last_error,
            sorted(tried),
        )
        return web.json_response(
            {"error": f"All backends failed. Last error: {last_error}"},
            status=502 if last_error else 503,
        )

    def make_app(self) -> web.Application:
        # Judge payloads can include many long candidate responses. Let each
        # upstream model enforce its own request/model-length limits.
        app = web.Application(client_max_size=0)
        app.router.add_get("/health", self.handle_health)
        # Catch-all: proxy everything else
        app.router.add_route("*", "/{path:.*}", self.handle_proxy)

        async def on_startup(_app: web.Application) -> None:
            await self.start()

        async def on_cleanup(_app: web.Application) -> None:
            await self.stop()

        app.on_startup.append(on_startup)
        # Cleanup runs after aiohttp has stopped accepting requests and drained
        # active handlers, so the upstream client session remains usable while
        # long generations finish.
        app.on_cleanup.append(on_cleanup)
        return app


def main() -> None:
    parser = argparse.ArgumentParser(description="External vLLM Pool Load Balancer")
    parser.add_argument("--port", type=int, default=8080, help="LB listen port")
    parser.add_argument(
        "--registry-dir",
        default=os.environ.get(
            "EXTERNAL_VLLM_STATE_DIR", os.path.dirname(os.path.abspath(__file__))
        ),
        help="Directory containing the registry file",
    )
    parser.add_argument(
        "--group-id",
        default=os.environ.get("EXTERNAL_VLLM_GROUP_ID", "default"),
        help="Server group ID",
    )
    parser.add_argument(
        "--health-interval",
        type=float,
        default=5.0,
        help="Seconds between health checks / registry refresh",
    )
    args = parser.parse_args()

    pool = BackendPool(args.registry_dir, args.group_id, args.health_interval)
    lb = LoadBalancer(pool, args.port)
    app = lb.make_app()

    log.info(
        "Starting external vLLM load balancer on port %d (group=%s)",
        args.port,
        args.group_id,
    )
    log.info("Registry: %s", pool.registry_file)

    # Stop accepting new connections immediately, but give in-flight requests
    # a bounded window to finish before the watchdog restarts the process.
    web.run_app(
        app,
        port=args.port,
        print=log.info,
        shutdown_timeout=SHUTDOWN_TIMEOUT_SECONDS,
    )


if __name__ == "__main__":
    main()
