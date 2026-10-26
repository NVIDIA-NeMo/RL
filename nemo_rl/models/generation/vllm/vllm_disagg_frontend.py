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
"""One replica's prefill/decode disaggregation frontend for the vLLM backend.

The TRT-LLM backend fronts a replica with TRT-LLM's own ``OpenAIDisaggServer``;
vLLM ships no deployable equivalent, so this module is that component: a small
OpenAI-compatible relay that turns one NeMo-Gym request into a prefill leg and
a decode leg against the replica's two engine pools. It invents no protocol.
Towards Gym it is an ordinary ``/v1/chat/completions``; towards the engines it
speaks vLLM's ``kv_transfer_params`` handshake, relaying the prefill engine's
handshake dict into the decode request without reading its keys (connectors
differ in what they put there; see ``kv_connector_profiles``).

Started like ``trtllm_disagg_server``: a uvicorn app in a daemon thread,
returning the URL NeMo-Gym talks to. NeMo RL hands it the two address pools
and the router policies; everything below that -- the KV transfer itself -- is
the connector's.
"""

import asyncio
import copy
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import ray

from nemo_rl.models.generation.openai_server_utils import (
    decode_token_ids_b64,
    encode_token_ids_b64,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CHAT_PATH",
    "COMPLETIONS_PATH",
    "BlockHashIndex",
    "ConversationKVCache",
    "ConversationRouter",
    "DisaggFrontendActor",
    "DisaggFrontendImpl",
    "EngineEndpoint",
    "FrontendRelay",
    "KvCacheAwareRouter",
    "LoadBalancingRouter",
    "RelayError",
    "RelaySettings",
    "RequestBlockHasher",
    "RoundRobinRouter",
    "build_decode_leg",
    "build_prefill_leg",
    "conversation_key",
    "decode_token_ids_b64",
    "encode_token_ids_b64",
    "make_router",
    "start_server",
    "wait_ready",
]

CHAT_PATH = "/v1/chat/completions"
COMPLETIONS_PATH = "/v1/completions"
_RELAYED_PATHS = frozenset({CHAT_PATH, COMPLETIONS_PATH})

# Request fields that must not reach the prefill leg. The leg generates one
# token nobody reads, so logprob plumbing is waste; ``ng_capture`` is Gym's
# token-capture handle and may only ever complete on the decode leg, which is
# where the trajectory's tokens come from; ``conversation_params`` is consumed
# here (routing) and is a TRT-LLM extension vLLM does not declare.
_PREFILL_LEG_DROPPED_FIELDS = (
    "stream_options",
    "min_tokens",
    "min_completion_tokens",
    "logprobs",
    "top_logprobs",
    "return_tokens_as_token_ids",
    "ng_capture",
    "conversation_params",
)
# The Gym token fields on assistant messages. The engine's request model
# derives ``required_prefix_token_ids`` from them, which must not happen on a
# leg whose prompt ids are already spliced.
_MESSAGE_TOKEN_FIELDS = (
    "prompt_token_ids",
    "generation_token_ids",
    "generation_log_probs",
    "routed_experts",
)
# Hosts a connector may advertise when its side-channel host was left at the
# default; the frontend substitutes the engine's actual host.
_LOCAL_HOSTS = frozenset({None, "", "localhost", "127.0.0.1", "0.0.0.0", "::1"})


# --------------------------------------------------------------------------- #
#  Conversation identity
# --------------------------------------------------------------------------- #


def conversation_key(body: dict[str, Any]) -> Optional[str]:
    """The key a trajectory's turns share, or None for a one-off request.

    NeMo-Gym sends its session id as ``conversation_params.conversation_id``
    when ``forward_session_id_as_conversation_id`` is on; that is the stable
    key. Without it the first message of the conversation is hashed: every
    turn of a trajectory repeats its opening message verbatim, so the hash is
    the same across turns while unrelated conversations differ.
    """
    params = body.get("conversation_params")
    if isinstance(params, dict) and params.get("conversation_id"):
        return str(params["conversation_id"])
    messages = body.get("messages")
    if isinstance(messages, list) and messages:
        opening = {
            k: v for k, v in messages[0].items() if k not in _MESSAGE_TOKEN_FIELDS
        }
        return hashlib.sha256(
            json.dumps(opening, sort_keys=True, default=str).encode()
        ).hexdigest()
    return None


# --------------------------------------------------------------------------- #
#  The two legs
# --------------------------------------------------------------------------- #


def _strip_message_token_fields(leg: dict[str, Any]) -> None:
    messages = leg.get("messages")
    if not isinstance(messages, list):
        return
    leg["messages"] = [
        {k: v for k, v in message.items() if k not in _MESSAGE_TOKEN_FIELDS}
        if isinstance(message, dict)
        else message
        for message in messages
    ]


def _attach_prompt_ids(
    leg: dict[str, Any],
    token_ids: list[int],
    *,
    as_b64: bool,
    strip_message_token_fields: bool,
) -> None:
    """Hand the engine prompt ids so it skips rendering and tokenization.

    The JSON form rides vLLM's own ``kv_transfer_params.prompt_token_ids``,
    which its renderer consumes directly; the base64 form is NeMo RL's
    extension, decoded by the engine's HTTP server into the same place. The
    engine treats supplied ids as final: it neither splices a prefix into them
    nor derives one from the Gym token fields on the messages. Those fields are
    dropped from the decode leg (the relayed ids carry the prefix) but kept on
    the prefill leg, where the engine's shadow validation recomputes the ids
    from them.
    """
    if as_b64:
        leg["prompt_token_ids_b64"] = encode_token_ids_b64(token_ids)
    else:
        params = dict(leg.get("kv_transfer_params") or {})
        params["prompt_token_ids"] = list(token_ids)
        leg["kv_transfer_params"] = params
    leg.pop("required_prefix_token_ids", None)
    if strip_message_token_fields:
        _strip_message_token_fields(leg)


def build_prefill_leg(
    body: dict[str, Any],
    *,
    initial_params: dict[str, Any],
    cached_decode_params: Optional[dict[str, Any]] = None,
    frontend_prompt_ids: Optional[list[int]] = None,
    ids_as_b64: bool = False,
    return_prompt_ids_b64: bool = False,
) -> dict[str, Any]:
    """The prefill engine's request for ``body``.

    Prefill only: one token (vLLM runs a request only if it generates at least
    one), non-streaming, and ``return_token_ids`` so the response carries the
    prompt ids the decode leg reuses. ``kv_transfer_params`` is the connector's
    cold marker, or -- bidirectional KV transfer -- the decode engine's handshake
    from the previous turn re-flagged so the prefill engine pulls that KV back.
    """
    leg = copy.deepcopy(body)
    for name in _PREFILL_LEG_DROPPED_FIELDS:
        leg.pop(name, None)
    leg["stream"] = False
    leg["max_tokens"] = 1
    leg.pop("max_completion_tokens", None)
    leg["return_token_ids"] = True
    if return_prompt_ids_b64:
        leg["return_prompt_token_ids_b64"] = True

    if cached_decode_params:
        params = dict(cached_decode_params)
        params["do_remote_decode"] = True
        params["do_remote_prefill"] = False
    else:
        params = dict(initial_params)
    leg["kv_transfer_params"] = params

    if frontend_prompt_ids is not None:
        _attach_prompt_ids(
            leg,
            frontend_prompt_ids,
            as_b64=ids_as_b64,
            strip_message_token_fields=False,
        )
    return leg


class RelayError(Exception):
    """A leg failed in a way the frontend reports to the client itself."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def prefill_response_prompt_ids(response: dict[str, Any]) -> Optional[list[int]]:
    """The prompt ids a prefill response carries, in either encoding."""
    if response.get("prompt_token_ids_b64"):
        return decode_token_ids_b64(response["prompt_token_ids_b64"])
    ids = response.get("prompt_token_ids")
    return list(ids) if ids else None


def build_decode_leg(
    body: dict[str, Any],
    *,
    prefill_response: dict[str, Any],
    prefill_host: str,
    ids_as_b64: bool = False,
    strip_message_history: bool = False,
) -> dict[str, Any]:
    """The decode engine's request: ``body`` plus the prefill engine's handshake.

    The handshake dict is relayed as the connector returned it; only a
    default-valued ``remote_host`` is replaced by the engine's real host, since
    a connector left at its defaults advertises ``localhost``. The prompt ids
    the prefill engine rendered go along so the decode engine's sequence is
    exactly the one whose KV it receives.
    """
    params = prefill_response.get("kv_transfer_params")
    if not isinstance(params, dict) or not params:
        raise RelayError(
            502,
            "prefill response carried no kv_transfer_params; the prefill engine "
            "is most likely running without a KV connector",
        )
    params = dict(params)
    if params.get("remote_host") in _LOCAL_HOSTS and "remote_host" in params:
        params["remote_host"] = prefill_host

    leg = copy.deepcopy(body)
    leg.pop("conversation_params", None)
    leg["kv_transfer_params"] = params

    prompt_ids = prefill_response_prompt_ids(prefill_response)
    if "messages" in leg:
        if prompt_ids is None:
            raise RelayError(
                502,
                "prefill response carried no prompt_token_ids; the decode leg "
                "cannot reuse the prefilled sequence",
            )
        _attach_prompt_ids(
            leg, prompt_ids, as_b64=ids_as_b64, strip_message_token_fields=True
        )
        messages = leg.get("messages")
        if strip_message_history and isinstance(messages, list) and len(messages) > 1:
            # The relayed ids carry the whole prefix; the decode engine only
            # reads the last message (tool parsing, continue_final_message).
            leg["messages"] = messages[-1:]
    elif prompt_ids is not None:
        # /v1/completions: a token prompt is the native way to hand over ids.
        leg["prompt"] = prompt_ids
    return leg


# --------------------------------------------------------------------------- #
#  Routers
# --------------------------------------------------------------------------- #


class RoundRobinRouter:
    """Stateless: engines in turn."""

    def __init__(self, num_engines: int) -> None:
        assert num_engines >= 1
        self.num_engines = num_engines
        self._next = 0
        self.inflight = [0] * num_engines

    def pick(self, key: Optional[str], prompt_ids: Optional[list[int]] = None) -> int:
        del key, prompt_ids
        idx = self._next
        self._next = (self._next + 1) % self.num_engines
        return idx

    def acquire(self, idx: int) -> None:
        self.inflight[idx] += 1

    def release(self, idx: int) -> None:
        self.inflight[idx] -= 1


class LoadBalancingRouter(RoundRobinRouter):
    """Stateless: the engine with the fewest requests in flight, ties in turn."""

    def pick(self, key: Optional[str], prompt_ids: Optional[list[int]] = None) -> int:
        del key, prompt_ids
        least = min(self.inflight)
        # Rotate the tie-break so equal-load engines share new requests.
        for offset in range(self.num_engines):
            idx = (self._next + offset) % self.num_engines
            if self.inflight[idx] == least:
                self._next = (idx + 1) % self.num_engines
                return idx
        raise AssertionError("unreachable: some engine has the minimum load")


class ConversationRouter(LoadBalancingRouter):
    """Stateful: a conversation keeps the engine its first turn landed on.

    The first turn goes to the least-loaded engine; later turns follow it, so
    the prefill engine holding the conversation's prefix in its cache serves
    them. The table is an LRU of ``table_size`` conversations and lives in
    this frontend only -- replicas are disjoint and Gym pins a session to one
    frontend, so nothing is shared.
    """

    def __init__(self, num_engines: int, *, table_size: int = 100_000) -> None:
        super().__init__(num_engines)
        self._table: OrderedDict[str, int] = OrderedDict()
        self._table_size = table_size

    def pick(self, key: Optional[str], prompt_ids: Optional[list[int]] = None) -> int:
        if key is None:
            return super().pick(None, prompt_ids)
        idx = self._table.get(key)
        if idx is None:
            idx = super().pick(None, prompt_ids)
        else:
            self._table.move_to_end(key)
        self._table[key] = idx
        while len(self._table) > self._table_size:
            self._table.popitem(last=False)
        return idx

    def forget_all(self) -> None:
        self._table.clear()


class BlockHashIndex:
    """Which engines hold which prefix-cache blocks, from their KV events.

    vLLM chains block hashes over the whole prefix, so one hash fingerprints a
    conversation's prompt up to that block boundary. A request is matched by
    walking its own block hashes from the longest prefix down and stopping at
    the first hash some engine holds.
    """

    def __init__(self, num_engines: int) -> None:
        self._engines_by_hash: dict[Any, set[int]] = {}
        self._hashes_by_engine: list[set[Any]] = [set() for _ in range(num_engines)]

    def store(self, engine_idx: int, hashes: list[Any]) -> None:
        for h in hashes:
            self._engines_by_hash.setdefault(h, set()).add(engine_idx)
            self._hashes_by_engine[engine_idx].add(h)

    def remove(self, engine_idx: int, hashes: list[Any]) -> None:
        for h in hashes:
            owners = self._engines_by_hash.get(h)
            if owners is not None:
                owners.discard(engine_idx)
                if not owners:
                    del self._engines_by_hash[h]
            self._hashes_by_engine[engine_idx].discard(h)

    def clear(self, engine_idx: int) -> None:
        for h in self._hashes_by_engine[engine_idx]:
            owners = self._engines_by_hash.get(h)
            if owners is not None:
                owners.discard(engine_idx)
                if not owners:
                    del self._engines_by_hash[h]
        self._hashes_by_engine[engine_idx] = set()

    def longest_match(self, hashes: list[Any]) -> tuple[set[int], int]:
        """``(engines, matched_blocks)`` for the deepest cached prefix, or ``(set(), 0)``."""
        for depth in range(len(hashes), 0, -1):
            owners = self._engines_by_hash.get(hashes[depth - 1])
            if owners:
                return set(owners), depth
        return set(), 0

    def __len__(self) -> int:
        return len(self._engines_by_hash)


class RequestBlockHasher:
    """Block hashes of a prompt, computed the way the engines compute theirs.

    Uses vLLM's own ``hash_block_tokens`` with the engines' hash algorithm and
    block size, so the hashes equal the ones the engines publish in their KV
    events. ``hash_fn``/``to_external`` are injectable for tests; the defaults
    import vLLM lazily (the frontend runs in the generation venv).
    """

    def __init__(
        self,
        *,
        block_size: int,
        hash_algo: str = "sha256",
        hash_fn: Optional[Callable[[Any], bytes]] = None,
        to_external: Optional[Callable[[bytes], Any]] = None,
    ) -> None:
        assert block_size >= 1
        self.block_size = block_size
        if hash_fn is None or to_external is None:
            # Deferred: vLLM lives in the generation venv only.
            from vllm.utils import hashing as vllm_hashing
            from vllm.v1.core import kv_cache_utils

            resolved_fn = getattr(vllm_hashing, hash_algo, None)
            if hash_fn is None:
                if resolved_fn is None:
                    raise ValueError(
                        f"unknown prefix_caching_hash_algo {hash_algo!r}; vLLM's "
                        "hashing module has no such function"
                    )
                hash_fn = resolved_fn
            # Same seed derivation as the engines (PYTHONHASHSEED, else the
            # algorithm's default), so the first block hashes equal theirs.
            kv_cache_utils.init_none_hash(hash_fn)
            self._hash_block_tokens = kv_cache_utils.hash_block_tokens
            if to_external is None:
                to_external = kv_cache_utils.maybe_convert_block_hash
        else:
            none_hash = hash_fn("none")

            def _hash_block_tokens(fn, parent, tokens, extra_keys=None):  # type: ignore[no-untyped-def]
                return fn((parent or none_hash, tuple(tokens), extra_keys))

            self._hash_block_tokens = _hash_block_tokens
        self._hash_fn = hash_fn
        self._to_external = to_external

    def hashes(self, token_ids: list[int]) -> list[Any]:
        """One hash per *full* block of ``token_ids``, in the events' representation."""
        result: list[Any] = []
        parent = None
        for start in range(0, len(token_ids) - self.block_size + 1, self.block_size):
            block = self._hash_block_tokens(
                self._hash_fn, parent, token_ids[start : start + self.block_size], None
            )
            result.append(self._to_external(block))
            parent = block
        return result


class KvEventSubscriber:
    """Feeds one engine's KV cache events into a :class:`BlockHashIndex`.

    Subscribes to the ZMQ publisher vLLM runs when ``kv_events_config`` is on
    (``BlockStored`` / ``BlockRemoved`` / ``AllBlocksCleared`` batches, msgspec
    msgpack, one ``(topic, seq, payload)`` multipart message each). Runs in a
    daemon thread; the index is guarded by ``lock``.
    """

    def __init__(
        self,
        *,
        engine_idx: int,
        endpoint: str,
        topic: str,
        index: BlockHashIndex,
        lock: threading.Lock,
    ) -> None:
        self._engine_idx = engine_idx
        self._endpoint = endpoint
        self._topic = topic
        self._index = index
        self._lock = lock
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.batches_seen = 0
        self.last_seq: Optional[int] = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name=f"kv-events-{self._engine_idx}"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def apply(self, batch: Any) -> None:
        """Apply one decoded ``KVEventBatch`` (also the unit-test entry point)."""
        with self._lock:
            for event in batch.events:
                kind = type(event).__name__
                if kind == "BlockStored":
                    self._index.store(self._engine_idx, list(event.block_hashes))
                elif kind == "BlockRemoved":
                    self._index.remove(self._engine_idx, list(event.block_hashes))
                elif kind == "AllBlocksCleared":
                    self._index.clear(self._engine_idx)
        self.batches_seen += 1

    def _run(self) -> None:
        # Deferred: zmq/msgspec and the vLLM event types live in the generation venv.
        import msgspec
        import zmq
        from vllm.distributed.kv_events import KVEventBatch

        decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        sock.connect(self._endpoint)
        sock.setsockopt_string(zmq.SUBSCRIBE, self._topic)
        try:
            while not self._stop.is_set():
                try:
                    _topic, seq_bytes, payload = sock.recv_multipart()
                except zmq.Again:
                    continue
                seq = int.from_bytes(seq_bytes, "big")
                if self.last_seq is not None and seq != self.last_seq + 1:
                    # A gap means blocks stored meanwhile are unknown to us;
                    # routing then degrades to the fallback for them, which is
                    # correct, just less cache-aware.
                    logger.warning(
                        "KV events from engine %d skipped %d -> %d",
                        self._engine_idx,
                        self.last_seq,
                        seq,
                    )
                self.last_seq = seq
                self.apply(decoder.decode(payload))
        finally:
            sock.close(linger=0)


class KvCacheAwareRouter(ConversationRouter):
    """Route by the blocks the prefill engines report holding.

    The request's prompt ids are hashed block by block the way the engines
    hash theirs; the engine holding the deepest prefix wins, least loaded
    among ties. Without a match -- or without prompt ids, which is why this
    router needs ``frontend_tokenize`` -- it behaves as the conversation
    router. The index is fed by one :class:`KvEventSubscriber` per engine,
    attached by the frontend once the engines are up.
    """

    def __init__(
        self,
        num_engines: int,
        *,
        hasher: Optional[RequestBlockHasher] = None,
        table_size: int = 100_000,
    ) -> None:
        super().__init__(num_engines, table_size=table_size)
        self.index = BlockHashIndex(num_engines)
        self.index_lock = threading.Lock()
        self.hasher = hasher
        self.hits = 0
        self.misses = 0

    def pick(self, key: Optional[str], prompt_ids: Optional[list[int]] = None) -> int:
        if self.hasher is None or not prompt_ids:
            return super().pick(key, prompt_ids)
        hashes = self.hasher.hashes(prompt_ids)
        with self.index_lock:
            owners, _depth = self.index.longest_match(hashes)
        if not owners:
            self.misses += 1
            return super().pick(key, prompt_ids)
        self.hits += 1
        idx = min(owners, key=lambda i: (self.inflight[i], i))
        if key is not None:
            self._table[key] = idx
            self._table.move_to_end(key)
        return idx


def make_router(
    kind: str,
    num_engines: int,
    *,
    table_size: int,
    hasher: Optional[RequestBlockHasher] = None,
) -> RoundRobinRouter:
    if kind == "round_robin":
        return RoundRobinRouter(num_engines)
    if kind == "load_balancing":
        return LoadBalancingRouter(num_engines)
    if kind == "conversation":
        return ConversationRouter(num_engines, table_size=table_size)
    if kind == "kv_cache_aware":
        return KvCacheAwareRouter(num_engines, hasher=hasher, table_size=table_size)
    raise ValueError(f"unknown router {kind!r}")


# --------------------------------------------------------------------------- #
#  Bidirectional KV transfer state
# --------------------------------------------------------------------------- #


@dataclass
class _CachedTurn:
    kv_transfer_params: dict[str, Any]
    stored_at: float


class ConversationKVCache:
    """The decode engine's handshake for each conversation's last turn.

    With ``bidirectional_kv_xfer`` the decode engine pins a finished
    conversation's blocks and returns their handshake; attaching it to the
    next turn's prefill leg lets the prefill engine pull that KV instead of
    recomputing the tokens the decode engine generated. Entries expire after
    ``ttl_s`` (keep it below the engine's ``decoder_kv_blocks_ttl``) and the
    whole table is dropped on every weight update: blocks computed under old
    weights must never be pulled back.
    """

    def __init__(self, ttl_s: float) -> None:
        self._ttl_s = ttl_s
        self._store: dict[str, _CachedTurn] = {}
        self.hits = 0
        self.misses = 0
        self.expired = 0

    def get(self, key: str) -> Optional[dict[str, Any]]:
        entry = self._store.pop(key, None)
        if entry is None:
            self.misses += 1
            return None
        if time.monotonic() - entry.stored_at > self._ttl_s:
            self.expired += 1
            return None
        self.hits += 1
        return dict(entry.kv_transfer_params)

    def put(self, key: str, kv_transfer_params: dict[str, Any]) -> None:
        self._store[key] = _CachedTurn(dict(kv_transfer_params), time.monotonic())

    def clear(self) -> int:
        n = len(self._store)
        self._store.clear()
        return n

    def __len__(self) -> int:
        return len(self._store)


# --------------------------------------------------------------------------- #
#  The relay
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EngineEndpoint:
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class RelaySettings:
    """Everything a frontend needs besides the engine addresses.

    A plain record so the driver can build it from the resolved config and ship
    it to the actor; the frontend never reads the generation config itself.
    """

    prefill_router: str
    decode_router: str
    initial_prefill_params: dict[str, Any]
    bidirectional_kv_xfer: bool
    conversation_cache_ttl_s: float
    decode_tokids_b64: bool
    decode_strip_message_history: bool
    request_timeout_s: float
    prefill_router_table_size: int
    # kv_cache_aware: the ZMQ endpoints of the prefill engines' event
    # publishers (one per prefill engine, in prefill-pool order) and the
    # publisher topic. Hashing parameters arrive with mark_engines_ready.
    kv_events_endpoints: tuple[str, ...] = ()
    kv_events_topic: str = ""
    # frontend_tokenize: None renders on the engines.
    render_settings: Optional[Any] = None


class FrontendRelay:
    """The two-leg relay behind one frontend's HTTP app.

    Request path, one turn:

    1. Take the conversation key and the engine each router picks.
    2. Prefill leg (:func:`build_prefill_leg`) to the chosen prefill engine.
    3. Decode leg (:func:`build_decode_leg`) to the chosen decode engine,
       carrying the prefill response's handshake and prompt ids.
    4. Hand the decode response back; with bidirectional transfer, remember
       the decode engine's handshake for the next turn.

    Engine 4xx responses are passed through with status and body intact: Gym
    classifies the ``context length`` rejection by its text, and the
    aggregated path answers it directly.
    """

    def __init__(
        self,
        *,
        prefill: list[EngineEndpoint],
        decode: list[EngineEndpoint],
        settings: RelaySettings,
    ) -> None:
        assert prefill and decode, "a replica needs engines of both roles"
        self.prefill = list(prefill)
        self.decode = list(decode)
        self.settings = settings
        self.prefill_router = make_router(
            settings.prefill_router,
            len(prefill),
            table_size=settings.prefill_router_table_size,
        )
        self.decode_router = make_router(
            settings.decode_router,
            len(decode),
            table_size=settings.prefill_router_table_size,
        )
        self.conversation_cache: Optional[ConversationKVCache] = (
            ConversationKVCache(settings.conversation_cache_ttl_s)
            if settings.bidirectional_kv_xfer
            else None
        )
        self.renderer: Optional[Any] = None
        self._engines_ready = False
        self._event_subscribers: list[KvEventSubscriber] = []
        self._session: Any = None
        self.stats: dict[str, int] = {
            "requests": 0,
            "prefill_errors": 0,
            "decode_errors": 0,
            "frontend_tokenized": 0,
            "frontend_tokenize_fallbacks": 0,
        }

    # -- lifecycle -------------------------------------------------------- #

    @property
    def engines_ready(self) -> bool:
        return self._engines_ready

    def mark_engines_ready(
        self, *, block_size: Optional[int] = None, hash_algo: Optional[str] = None
    ) -> None:
        """Start serving requests; wire the KV-aware router now that engines exist.

        Deferred until here because under ``defer_model_load`` the frontend is
        up, and Gym already holds its URL, while the engines still load.
        """
        router = self.prefill_router
        if isinstance(router, KvCacheAwareRouter):
            if block_size is None or hash_algo is None:
                raise ValueError(
                    "kv_cache_aware routing needs the engines' block_size and "
                    "prefix_caching_hash_algo to hash prompts like they do"
                )
            router.hasher = RequestBlockHasher(
                block_size=block_size, hash_algo=hash_algo
            )
            for engine_idx, endpoint in enumerate(self.settings.kv_events_endpoints):
                subscriber = KvEventSubscriber(
                    engine_idx=engine_idx,
                    endpoint=endpoint,
                    topic=self.settings.kv_events_topic,
                    index=router.index,
                    lock=router.index_lock,
                )
                subscriber.start()
                self._event_subscribers.append(subscriber)
        self._engines_ready = True

    def install_renderer(self, renderer: Any) -> None:
        self.renderer = renderer

    def invalidate_conversation_cache(self) -> int:
        """Drop every remembered decode-side handshake (weight update)."""
        if self.conversation_cache is None:
            return 0
        return self.conversation_cache.clear()

    def shutdown(self) -> None:
        for subscriber in self._event_subscribers:
            subscriber.stop()
        self._event_subscribers = []

    # -- HTTP plumbing ---------------------------------------------------- #

    async def _client(self) -> Any:
        # Deferred: aiohttp lives in the generation venv; created on the
        # serving loop so the connection pool belongs to it.
        import aiohttp

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.settings.request_timeout_s)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=aiohttp.TCPConnector(limit=0),
                json_serialize=json.dumps,
            )
        return self._session

    async def _post_json(
        self, endpoint: EngineEndpoint, path: str, payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any]]:
        session = await self._client()
        async with session.post(endpoint.base_url + path, json=payload) as resp:
            raw = await resp.read()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            body = {
                "error": {"message": raw.decode(errors="replace"), "code": resp.status}
            }
        if not isinstance(body, dict):
            body = {"error": {"message": "non-object JSON from engine", "code": 502}}
        return resp.status, body

    # -- the two legs ----------------------------------------------------- #

    async def _frontend_prompt_ids(self, body: dict[str, Any]) -> Optional[list[int]]:
        if self.renderer is None or "messages" not in body:
            return None
        try:
            ids = await self.renderer.prompt_token_ids(body)
        except Exception:  # noqa: BLE001 - fall back to engine-side rendering
            logger.exception(
                "frontend tokenize failed; rendering on the prefill engine"
            )
            self.stats["frontend_tokenize_fallbacks"] += 1
            return None
        self.stats["frontend_tokenized"] += 1
        return ids

    async def relay(self, path: str, body: dict[str, Any]) -> tuple[int, Any, bool]:
        """Serve one request. Returns ``(status, payload, streaming)``.

        ``payload`` is a dict for JSON responses, or an async byte iterator when
        the client asked to stream and the decode leg is being streamed through.
        """
        self.stats["requests"] += 1
        if not self._engines_ready:
            return (
                503,
                {
                    "error": {
                        "message": "disaggregation frontend up, engines still loading",
                        "type": "service_unavailable",
                        "code": 503,
                    }
                },
                False,
            )

        key = conversation_key(body)
        frontend_ids = (
            await self._frontend_prompt_ids(body) if path == CHAT_PATH else None
        )
        cached = (
            self.conversation_cache.get(key)
            if self.conversation_cache is not None and key is not None
            else None
        )
        prefill_leg = build_prefill_leg(
            body,
            initial_params=self.settings.initial_prefill_params,
            cached_decode_params=cached,
            frontend_prompt_ids=frontend_ids,
            ids_as_b64=self.settings.decode_tokids_b64,
            return_prompt_ids_b64=self.settings.decode_tokids_b64,
        )

        p_idx = self.prefill_router.pick(key, frontend_ids)
        self.prefill_router.acquire(p_idx)
        try:
            p_status, p_body = await self._post_json(
                self.prefill[p_idx], path, prefill_leg
            )
        except Exception as e:  # noqa: BLE001 - any transport failure is a 502 to Gym
            self.stats["prefill_errors"] += 1
            return (
                502,
                _error_body(f"prefill engine {p_idx} unreachable: {e}", 502),
                False,
            )
        finally:
            self.prefill_router.release(p_idx)
        if p_status >= 400:
            self.stats["prefill_errors"] += 1
            return p_status, p_body, False

        try:
            decode_leg = build_decode_leg(
                body,
                prefill_response=p_body,
                prefill_host=self.prefill[p_idx].host,
                ids_as_b64=self.settings.decode_tokids_b64,
                strip_message_history=self.settings.decode_strip_message_history,
            )
        except RelayError as e:
            self.stats["prefill_errors"] += 1
            return e.status, _error_body(e.message, e.status), False

        d_idx = self.decode_router.pick(key)
        self.decode_router.acquire(d_idx)
        if body.get("stream"):
            # Streamed through as-is; nothing is read off the stream, so
            # bidirectional state is not updated for streamed turns (Gym does
            # not stream).
            return 200, self._stream_decode(d_idx, path, decode_leg), True
        try:
            d_status, d_body = await self._post_json(
                self.decode[d_idx], path, decode_leg
            )
        except Exception as e:  # noqa: BLE001 - any transport failure is a 502 to Gym
            self.stats["decode_errors"] += 1
            return (
                502,
                _error_body(f"decode engine {d_idx} unreachable: {e}", 502),
                False,
            )
        finally:
            self.decode_router.release(d_idx)
        if d_status >= 400:
            self.stats["decode_errors"] += 1
            return d_status, d_body, False

        # The decode engine's handshake is the frontend's business, not Gym's.
        decode_params = d_body.pop("kv_transfer_params", None)
        if self.conversation_cache is not None and key is not None and decode_params:
            self.conversation_cache.put(key, decode_params)
        return 200, d_body, False

    async def _stream_decode(self, d_idx: int, path: str, leg: dict[str, Any]) -> Any:
        session = await self._client()
        try:
            async with session.post(
                self.decode[d_idx].base_url + path, json=leg
            ) as resp:
                async for chunk in resp.content.iter_any():
                    yield chunk
        finally:
            self.decode_router.release(d_idx)


def _error_body(message: str, code: int) -> dict[str, Any]:
    return {"error": {"message": message, "type": "server_error", "code": code}}


# --------------------------------------------------------------------------- #
#  HTTP app and server
# --------------------------------------------------------------------------- #


def create_frontend_app(relay: FrontendRelay) -> Any:
    """The FastAPI app over ``relay``: the OpenAI routes Gym uses plus /health."""
    # Deferred: fastapi lives in the generation venv.
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI()

    @app.get("/health")
    async def health() -> JSONResponse:
        # 200 as soon as the frontend serves: Gym needs the URL answering
        # before the engines finish loading. Requests meanwhile get a 503.
        return JSONResponse(
            {
                "status": "ok",
                "engines_ready": relay.engines_ready,
                "prefill_engines": len(relay.prefill),
                "decode_engines": len(relay.decode),
            }
        )

    @app.get("/stats")
    async def stats() -> JSONResponse:
        payload: dict[str, Any] = dict(relay.stats)
        payload["prefill_inflight"] = list(relay.prefill_router.inflight)
        payload["decode_inflight"] = list(relay.decode_router.inflight)
        if relay.conversation_cache is not None:
            cache = relay.conversation_cache
            payload["conversation_cache"] = {
                "size": len(cache),
                "hits": cache.hits,
                "misses": cache.misses,
                "expired": cache.expired,
            }
        router = relay.prefill_router
        if isinstance(router, KvCacheAwareRouter):
            payload["kv_cache_aware"] = {
                "hits": router.hits,
                "misses": router.misses,
                "indexed_blocks": len(router.index),
            }
        return JSONResponse(payload)

    async def _relay_route(path: str, request: Request) -> Any:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse(
                _error_body("request body is not JSON", 400), status_code=400
            )
        if not isinstance(body, dict):
            return JSONResponse(_error_body("request body must be an object", 400), 400)
        status, payload, streaming = await relay.relay(path, body)
        if streaming:
            return StreamingResponse(payload, media_type="text/event-stream")
        return JSONResponse(content=payload, status_code=status)

    @app.post(CHAT_PATH)
    async def chat_completions(request: Request) -> Any:
        return await _relay_route(CHAT_PATH, request)

    @app.post(COMPLETIONS_PATH)
    async def completions(request: Request) -> Any:
        return await _relay_route(COMPLETIONS_PATH, request)

    return app


def start_server(
    app: Any, *, host: str = "0.0.0.0", port: int = 0
) -> tuple[threading.Thread, str, Any]:
    """Serve ``app`` in a daemon thread; returns ``(thread, base_url, server)``."""
    # Deferred: uvicorn lives in the generation venv.
    import uvicorn

    from nemo_rl.distributed.virtual_cluster import (
        _get_free_port_local,
        _get_node_ip_local,
    )

    if port == 0:
        port = _get_free_port_local()
    node_ip = _get_node_ip_local()
    base_url = f"http://{node_ip}:{port}/v1"

    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning", timeout_keep_alive=120
    )
    server = uvicorn.Server(config)

    def _run() -> None:
        try:
            import uvloop

            asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        except ImportError:
            pass
        asyncio.run(server.serve())

    thread = threading.Thread(target=_run, daemon=True, name="disagg-frontend")
    thread.start()
    logger.info("vLLM disagg frontend starting on %s", base_url)
    return thread, base_url, server


def wait_ready(base_url: str, timeout_s: float = 300.0) -> None:
    """Block until the frontend answers /health."""
    import requests

    health = base_url.rsplit("/v1", 1)[0] + "/health"
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            if requests.get(health, timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"disagg frontend at {base_url} did not become ready within {timeout_s}s"
            )
        time.sleep(0.5)


# --------------------------------------------------------------------------- #
#  The actor
# --------------------------------------------------------------------------- #


@dataclass
class FrontendServeArgs:
    """Constructor payload of one frontend: addresses plus :class:`RelaySettings`."""

    prefill_addrs: list[tuple[str, int]]
    decode_addrs: list[tuple[str, int]]
    settings: RelaySettings
    engines_ready: bool = False
    kv_hashing: dict[str, Any] = field(default_factory=dict)


class DisaggFrontendImpl:
    """Hosts one replica's frontend.

    Its own process rather than an engine worker's: this is the request hot
    path for the whole replica, and sharing a process with an engine would
    couple the replica's routing latency to that one engine's load. It holds
    no GPUs, so the isolation is cheap.

    Serving starts in the constructor: Ray re-runs ``__init__`` on actor
    restart but never replays method calls, so with a hard node pin and a
    deterministic port a restarted frontend comes back at the SAME URL and
    Gym's sticky clients recover after their 5xx retries. The routing tables
    start empty, which is a one-time cache miss.

    Held separately from the ``@ray.remote``-wrapped :class:`DisaggFrontendActor`
    so it can be exercised without Ray.
    """

    def __init__(
        self,
        replica_idx: int,
        frontend_idx: int = 0,
        num_frontends: int = 1,
        port: int = 0,
        serve_args: Optional[FrontendServeArgs] = None,
    ) -> None:
        self._replica_idx = replica_idx
        self._frontend_idx = frontend_idx
        self._num_frontends = num_frontends
        self._port = port
        self._thread: Optional[threading.Thread] = None
        self._server: Any = None
        self._base_url: Optional[str] = None
        self._relay: Optional[FrontendRelay] = None
        if serve_args is not None:
            self._start_serving(serve_args)

    def _start_serving(self, serve_args: FrontendServeArgs) -> None:
        self._relay = FrontendRelay(
            prefill=[EngineEndpoint(h, p) for h, p in serve_args.prefill_addrs],
            decode=[EngineEndpoint(h, p) for h, p in serve_args.decode_addrs],
            settings=serve_args.settings,
        )
        if serve_args.settings.render_settings is not None:
            # Built once; loading the tokenizer and HF config takes seconds.
            from nemo_rl.models.generation.vllm.chat_render import FrontendChatRenderer

            self._relay.install_renderer(
                FrontendChatRenderer(serve_args.settings.render_settings)
            )
        if serve_args.engines_ready:
            self._relay.mark_engines_ready(**serve_args.kv_hashing)
        self._thread, self._base_url, self._server = start_server(
            create_frontend_app(self._relay), port=self._port
        )

    def start(self) -> str:
        """Wait for the server to answer and return the URL NeMo-Gym will talk to."""
        assert self._base_url is not None and self._relay is not None, (
            "DisaggFrontendImpl must be constructed with serve_args; the "
            "constructor is what starts serving so Ray actor restart replays it."
        )
        wait_ready(self._base_url)
        # wait_ready cannot tell "my server is up" from "a server is up on this
        # host:port". uvicorn's bind failure only unwinds the daemon thread, so
        # without this a port collision is answered 200 by the winner and this
        # replica advertises a URL it does not own.
        assert self._thread is not None and self._thread.is_alive(), (
            f"disagg frontend {self._frontend_idx} for replica {self._replica_idx} "
            f"died while coming up at {self._base_url} (most likely its port was "
            "already bound on this node)"
        )
        logger.info(
            "disagg frontend %d/%d for replica %d ready at %s",
            self._frontend_idx,
            self._num_frontends,
            self._replica_idx,
            self._base_url,
        )
        return self._base_url

    def base_url(self) -> Optional[str]:
        return self._base_url

    def mark_engines_ready(self, **kv_hashing: Any) -> bool:
        assert self._relay is not None
        self._relay.mark_engines_ready(**kv_hashing)
        return True

    def invalidate_conversation_cache(self) -> int:
        assert self._relay is not None
        return self._relay.invalidate_conversation_cache()

    def stats(self) -> dict[str, Any]:
        assert self._relay is not None
        return dict(self._relay.stats)

    def shutdown(self) -> bool:
        if self._relay is not None:
            self._relay.shutdown()
        if self._server is not None:
            self._server.should_exit = True
        self._server = None
        self._thread = None
        self._base_url = None
        return True


@ray.remote(num_cpus=1, num_gpus=0)  # pragma: no cover
class DisaggFrontendActor(DisaggFrontendImpl):
    """Ray actor wrapper around :class:`DisaggFrontendImpl`."""

    pass
