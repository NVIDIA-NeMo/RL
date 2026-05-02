import asyncio
import json
import logging

import httpx
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_rl.models.generation.sglang.config import SGLangConfig

logger = logging.getLogger(__name__)


async def _post(client, url, payload, max_retries=10, action="post"):
    retry_count = 0
    while retry_count < max_retries:
        try:
            if action in ("delete", "get"):
                assert not payload
                response = await getattr(client, action)(url)
            else:
                response = await getattr(client, action)(url, json=payload or {})
            response.raise_for_status()
            try:
                output = response.json()
            except json.JSONDecodeError:
                output = response.text
        except Exception as e:
            retry_count += 1

            if isinstance(e, httpx.HTTPStatusError):
                response_text = e.response.text
            else:
                response_text = None

            logger.info(
                f"Error: {e}, retrying... (attempt {retry_count}/{max_retries}, url={url}, response={response_text})"
            )
            if retry_count >= max_retries:
                logger.info(
                    f"Max retries ({max_retries}) reached, failing... (url={url})"
                )
                raise e
            await asyncio.sleep(1)
            continue
        break

    return output


class HttpClient:
    """HTTP client wrapper with optional Ray-based distributed POST dispatch."""

    def __init__(self, args: SGLangConfig | None = None):
        self._client: httpx.AsyncClient | None = None
        self._client_concurrency: int = 0
        self._distributed_post_enabled: bool = False
        self._post_actors: list[object] = []
        self._post_actor_idx: int = 0

        if args is not None:
            self.init(args)

    def init(self, args: SGLangConfig) -> None:
        """Configure HTTP client limits and optional distributed POST actors."""
        server_cfg = args.get("sglang_server") or {}
        if not server_cfg.get("num_gpus"):
            return

        self._client_concurrency = (
            server_cfg["sglang_server_concurrency"]
            * server_cfg["num_gpus"]
            // server_cfg["num_gpus_per_engine"]
        )

        router_cfg = args.get("sglang_router") or {}
        if router_cfg.get("use_distributed_post"):
            self._init_ray_distributed_post(args)
            self._distributed_post_enabled = True

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            if self._client_concurrency > 0:
                self._client = httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=self._client_concurrency),
                    timeout=httpx.Timeout(None),
                )
            else:
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(None))
        return self._client

    def _next_actor(self):
        if not self._post_actors:
            return None
        actor = self._post_actors[self._post_actor_idx % len(self._post_actors)]
        self._post_actor_idx = (self._post_actor_idx + 1) % len(self._post_actors)
        return actor

    def _init_ray_distributed_post(self, args: SGLangConfig) -> None:
        """Initialize one or more Ray async actors per node for HTTP POST."""
        if self._post_actors:
            return  # Already initialized

        # Discover alive, schedulable nodes. Filter out nodes with CPU=0
        # (e.g. an unschedulable head node) — placing an actor there would hang.
        nodes = [
            n
            for n in ray.nodes()
            if n.get("Alive") and n.get("Resources", {}).get("CPU", 0) > 0
        ]
        if not nodes:
            raise RuntimeError("No alive Ray nodes to place HTTP POST actors.")

        @ray.remote
        class _HttpPosterActor:
            def __init__(self, concurrency: int):
                # Lazy creation to this actor's event loop
                self._client = httpx.AsyncClient(
                    limits=httpx.Limits(max_connections=max(1, concurrency)),
                    timeout=httpx.Timeout(None),
                )

            async def do_post(self, url, payload, max_retries=10, action="post"):
                return await _post(
                    self._client, url, payload, max_retries, action=action
                )

        created = []
        per_actor_conc = max(1, (self._client_concurrency + len(nodes)) // len(nodes))

        for node in nodes:
            node_id = node["NodeID"]
            scheduling = NodeAffinitySchedulingStrategy(node_id=node_id, soft=False)
            for _ in range(args["sglang_server"]["num_gpus_per_engine"]):
                actor = _HttpPosterActor.options(
                    name=None,
                    lifetime="detached",
                    scheduling_strategy=scheduling,
                    max_concurrency=per_actor_conc,
                    # Use tiny CPU to schedule
                    num_cpus=0.001,
                ).remote(per_actor_conc)
                created.append(actor)

        self._post_actors = created

    # TODO may generalize the name since it now contains http DELETE/GET etc (with retries and remote-execution)
    async def post(self, url, payload, max_retries=10, action="post"):
        if self._distributed_post_enabled and self._post_actors:
            try:
                actor = self._next_actor()
                if actor is not None:
                    # Use a thread to avoid blocking the event loop on ray.get
                    obj_ref = actor.do_post.remote(
                        url, payload, max_retries, action=action
                    )
                    return await asyncio.to_thread(ray.get, obj_ref)
            except Exception as e:
                logger.info(
                    f"[http_utils] Distributed POST failed, falling back to local: {e} (url={url})"
                )
                # fall through to local

        return await _post(self._get_client(), url, payload, max_retries, action=action)

    # TODO unify w/ `post` to add retries and remote-execution
    async def get(self, url):
        response = await self._get_client().get(url)
        response.raise_for_status()
        output = response.json()
        return output

    def shutdown(self) -> None:
        """Kill HTTP POST actors created by this client."""
        if not self._post_actors:
            return

        for actor in self._post_actors:
            try:
                ray.kill(actor)
            except Exception:
                pass
        self._post_actors = []
        self._post_actor_idx = 0
        self._distributed_post_enabled = False

    async def aclose(self) -> None:
        """Close local HTTP resources and kill distributed POST actors."""
        self.shutdown()
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def init_http_client(args: SGLangConfig) -> HttpClient:
    """Create an HTTP client for SGLang requests."""
    return HttpClient(args)


async def post(url, payload, max_retries=10, action="post"):
    """Issue a one-off HTTP request without shared module-level state."""
    client = HttpClient()
    try:
        return await client.post(url, payload, max_retries, action=action)
    finally:
        await client.aclose()


async def get(url):
    """Issue a one-off GET request without shared module-level state."""
    client = HttpClient()
    try:
        return await client.get(url)
    finally:
        await client.aclose()
