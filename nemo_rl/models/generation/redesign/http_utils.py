import asyncio
import json
import logging

import httpx

from nemo_rl.models.generation.redesign.config import SGLangConfig

logger = logging.getLogger(__name__)


_http_client: httpx.AsyncClient | None = None
_client_concurrency: int = 0

# Optional Ray-based distributed POST dispatch
_distributed_post_enabled: bool = False
_post_actors: list[object] = []
_post_actor_idx: int = 0


def _next_actor():
    global _post_actor_idx
    if not _post_actors:
        return None
    actor = _post_actors[_post_actor_idx % len(_post_actors)]
    _post_actor_idx = (_post_actor_idx + 1) % len(_post_actors)
    return actor


async def _post(client, url, payload, max_retries=60, action="post"):
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
                logger.info(f"Max retries ({max_retries}) reached, failing... (url={url})")
                raise e
            await asyncio.sleep(1)
            continue
        break

    return output


def init_http_client(args: SGLangConfig):
    """Initialize HTTP client and optionally enable distributed POST via Ray."""
    global _http_client, _client_concurrency, _distributed_post_enabled
    if not args.get("sglang_server").get("num_gpus"):
        return

    _client_concurrency = (
        args["sglang_server"]["sglang_server_concurrency"]
        * args["sglang_server"]["num_gpus"]
        // args["sglang_server"]["num_gpus_per_engine"]
    )
    if _http_client is None:
        _http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=_client_concurrency),
            timeout=httpx.Timeout(None),
        )

    # Optionally initialize distributed POST via Ray without changing interfaces
    if args.get("sglang_router").get("use_distributed_post"):
        _init_ray_distributed_post(args)
        _distributed_post_enabled = True


def _init_ray_distributed_post(args: SGLangConfig):
    """Initialize one or more Ray async actors per node for HTTP POST.
    """
    global _post_actors
    if _post_actors:
        return  # Already initialized

    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    # Discover alive nodes
    nodes = [n for n in ray.nodes() if n.get("Alive")]
    if not nodes:
        raise RuntimeError("No alive Ray nodes to place HTTP POST actors.")

    # Define the async actor
    @ray.remote
    class _HttpPosterActor:
        def __init__(self, concurrency: int):
            # Lazy creation to this actor's event loop
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=max(1, concurrency)),
                timeout=httpx.Timeout(None),
            )

        async def do_post(self, url, payload, max_retries=60, action="post"):
            return await _post(self._client, url, payload, max_retries, action=action)

    # Create actors per node
    created = []
    # Distribute client concurrency across actors (at least 1 per actor)
    per_actor_conc = (_client_concurrency + len(nodes)) // len(nodes)

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

    _post_actors = created


# TODO may generalize the name since it now contains http DELETE/GET etc (with retries and remote-execution)
async def post(url, payload, max_retries=60, action="post"):
    # If distributed mode is enabled and actors exist, dispatch via Ray.
    if _distributed_post_enabled and _post_actors:
        try:
            import ray

            actor = _next_actor()
            if actor is not None:
                # Use a thread to avoid blocking the event loop on ray.get
                obj_ref = actor.do_post.remote(url, payload, max_retries, action=action)
                return await asyncio.to_thread(ray.get, obj_ref)
        except Exception as e:
            logger.info(f"[http_utils] Distributed POST failed, falling back to local: {e} (url={url})")
            # fall through to local

    return await _post(_http_client, url, payload, max_retries, action=action)


# TODO unify w/ `post` to add retries and remote-execution
async def get(url):
    response = await _http_client.get(url)
    response.raise_for_status()
    output = response.json()
    return output
