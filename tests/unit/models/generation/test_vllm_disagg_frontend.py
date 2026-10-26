"""The vLLM disaggregation frontend: legs, routers, bidirectional state, relay."""

import asyncio
import hashlib
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.openai_server_utils import (
    decode_token_ids_b64,
    encode_token_ids_b64,
)
from nemo_rl.models.generation.vllm.kv_connector_profiles import NIXL_PROFILE
from nemo_rl.models.generation.vllm.vllm_disagg_frontend import (
    CHAT_PATH,
    COMPLETIONS_PATH,
    BlockHashIndex,
    ConversationKVCache,
    ConversationRouter,
    DisaggFrontendImpl,
    EngineEndpoint,
    FrontendRelay,
    KvCacheAwareRouter,
    KvEventSubscriber,
    LoadBalancingRouter,
    RelayError,
    RelaySettings,
    RequestBlockHasher,
    RoundRobinRouter,
    build_decode_leg,
    build_prefill_leg,
    conversation_key,
    make_router,
)

_INITIAL = NIXL_PROFILE.initial_prefill_params


def _gym_body(**extra):
    body = {
        "model": "m",
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "hello",
                "prompt_token_ids": [1, 2, 3],
                "generation_token_ids": [4, 5],
                "generation_log_probs": [-0.1, -0.2],
            },
            {"role": "user", "content": "more"},
        ],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 64,
        "logprobs": True,
        "top_logprobs": 0,
        "return_tokens_as_token_ids": True,
        "ng_capture": {"rollout_id": "r"},
        "required_prefix_token_ids": [1, 2, 3, 4, 5],
        "conversation_params": {"conversation_id": "sess-1"},
        "stream": False,
    }
    body.update(extra)
    return body


# -------------------------------------------------------------------------- #
#  Wire helpers
# -------------------------------------------------------------------------- #


def test_b64_token_ids_round_trip():
    ids = [0, 1, 70000, 151643]
    assert decode_token_ids_b64(encode_token_ids_b64(ids)) == ids


def test_conversation_key_prefers_the_gym_session_id():
    assert conversation_key(_gym_body()) == "sess-1"


def test_conversation_key_falls_back_to_the_opening_message_and_is_turn_stable():
    turn1 = {"messages": [{"role": "user", "content": "q"}]}
    turn2 = {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a", "generation_token_ids": [1]},
            {"role": "user", "content": "q2"},
        ]
    }
    assert conversation_key(turn1) == conversation_key(turn2)
    assert conversation_key(turn1) != conversation_key(
        {"messages": [{"role": "user", "content": "z"}]}
    )
    assert conversation_key({"prompt": "x"}) is None


# -------------------------------------------------------------------------- #
#  Legs
# -------------------------------------------------------------------------- #


def test_prefill_leg_is_one_token_non_streaming_and_carries_the_cold_marker():
    body = _gym_body()
    leg = build_prefill_leg(body, initial_params=_INITIAL)
    assert leg["max_tokens"] == 1
    assert leg["stream"] is False
    assert leg["return_token_ids"] is True
    assert leg["kv_transfer_params"] == _INITIAL
    for dropped in (
        "logprobs",
        "top_logprobs",
        "return_tokens_as_token_ids",
        "ng_capture",
        "conversation_params",
    ):
        assert dropped not in leg
    # The engine still splices the prefix itself when the frontend did not render.
    assert leg["required_prefix_token_ids"] == [1, 2, 3, 4, 5]
    assert leg["messages"][2]["generation_token_ids"] == [4, 5]
    # The caller's body is untouched.
    assert body["max_tokens"] == 64 and "ng_capture" in body


def test_prefill_leg_reflags_a_cached_decode_handshake_for_bidirectional_pull():
    cached = {
        "remote_block_ids": [[7, 8]],
        "remote_engine_id": "d0",
        "do_remote_decode": True,
        "do_remote_prefill": False,
    }
    leg = build_prefill_leg(
        _gym_body(), initial_params=_INITIAL, cached_decode_params=cached
    )
    assert leg["kv_transfer_params"]["remote_block_ids"] == [[7, 8]]
    assert leg["kv_transfer_params"]["do_remote_decode"] is True
    assert leg["kv_transfer_params"]["do_remote_prefill"] is False


@pytest.mark.parametrize("as_b64", [False, True])
def test_prefill_leg_attaches_frontend_ids_and_drops_the_prefix_but_keeps_message_fields(
    as_b64,
):
    leg = build_prefill_leg(
        _gym_body(),
        initial_params=_INITIAL,
        frontend_prompt_ids=[9, 9, 9],
        ids_as_b64=as_b64,
    )
    if as_b64:
        assert decode_token_ids_b64(leg["prompt_token_ids_b64"]) == [9, 9, 9]
        assert "prompt_token_ids" not in leg["kv_transfer_params"]
    else:
        assert leg["kv_transfer_params"]["prompt_token_ids"] == [9, 9, 9]
        assert leg["kv_transfer_params"]["do_remote_decode"] is True
    assert "required_prefix_token_ids" not in leg
    # Kept: the prefill engine's shadow validation recomputes from them.
    assert leg["messages"][2]["prompt_token_ids"] == [1, 2, 3]


def _prefill_response(**extra):
    resp = {
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "x"}}],
        "prompt_token_ids": [1, 2, 3, 4, 5, 6],
        "kv_transfer_params": {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_block_ids": [[1, 2]],
            "remote_engine_id": "replica0-prefill0",
            "remote_request_id": "req",
            "remote_host": "localhost",
            "remote_port": 8556,
            "tp_size": 2,
            "some_future_key": {"opaque": 1},
        },
    }
    resp.update(extra)
    return resp


def test_decode_leg_relays_the_handshake_opaquely_and_fixes_a_default_host():
    leg = build_decode_leg(
        _gym_body(), prefill_response=_prefill_response(), prefill_host="10.0.0.7"
    )
    params = leg["kv_transfer_params"]
    assert params["some_future_key"] == {"opaque": 1}
    assert params["remote_host"] == "10.0.0.7"
    assert params["do_remote_prefill"] is True
    assert params["prompt_token_ids"] == [1, 2, 3, 4, 5, 6]
    # Gym-only fields the decode engine must see are kept...
    assert leg["logprobs"] is True and leg["ng_capture"] == {"rollout_id": "r"}
    # ...while the prefix machinery is disarmed: ids are final.
    assert "required_prefix_token_ids" not in leg
    assert "generation_token_ids" not in leg["messages"][2]
    assert "conversation_params" not in leg
    assert len(leg["messages"]) == 4


def test_decode_leg_keeps_a_real_remote_host():
    resp = _prefill_response()
    resp["kv_transfer_params"]["remote_host"] = "10.0.0.9"
    leg = build_decode_leg(_gym_body(), prefill_response=resp, prefill_host="10.0.0.7")
    assert leg["kv_transfer_params"]["remote_host"] == "10.0.0.9"


def test_decode_leg_b64_and_history_stripping():
    leg = build_decode_leg(
        _gym_body(),
        prefill_response=_prefill_response(
            prompt_token_ids=None, prompt_token_ids_b64=encode_token_ids_b64([5, 6])
        ),
        prefill_host="h",
        ids_as_b64=True,
        strip_message_history=True,
    )
    assert decode_token_ids_b64(leg["prompt_token_ids_b64"]) == [5, 6]
    assert "prompt_token_ids" not in leg["kv_transfer_params"]
    assert leg["messages"] == [{"role": "user", "content": "more"}]


def test_decode_leg_requires_a_handshake_and_prompt_ids():
    with pytest.raises(RelayError) as e:
        build_decode_leg(
            _gym_body(), prefill_response={"choices": []}, prefill_host="h"
        )
    assert e.value.status == 502
    with pytest.raises(RelayError, match="prompt_token_ids"):
        build_decode_leg(
            _gym_body(),
            prefill_response=_prefill_response(prompt_token_ids=None),
            prefill_host="h",
        )


def test_completions_decode_leg_hands_over_a_token_prompt():
    leg = build_decode_leg(
        {"model": "m", "prompt": "hello", "max_tokens": 3},
        prefill_response=_prefill_response(),
        prefill_host="h",
    )
    assert leg["prompt"] == [1, 2, 3, 4, 5, 6]
    assert "prompt_token_ids" not in leg["kv_transfer_params"]


# -------------------------------------------------------------------------- #
#  Routers
# -------------------------------------------------------------------------- #


def test_round_robin_cycles():
    r = RoundRobinRouter(3)
    assert [r.pick(None) for _ in range(5)] == [0, 1, 2, 0, 1]


def test_load_balancing_picks_the_least_loaded_engine():
    r = LoadBalancingRouter(3)
    r.acquire(0)
    r.acquire(0)
    r.acquire(1)
    assert r.pick(None) == 2
    r.acquire(2)
    r.acquire(2)
    assert r.pick(None) == 1


def test_conversation_router_is_sticky_and_bounded():
    r = ConversationRouter(2, table_size=2)
    first = r.pick("a")
    r.acquire(first)
    assert r.pick("a") == first
    other = r.pick("b")
    assert other != first
    r.pick("c")  # evicts "a", the least recently used
    r.release(first)
    assert "a" not in r._table and "c" in r._table


def _fake_hasher(block_size=2):
    def hash_fn(obj):
        return hashlib.sha256(repr(obj).encode()).digest()

    return RequestBlockHasher(
        block_size=block_size, hash_fn=hash_fn, to_external=lambda b: b
    )


def test_request_block_hasher_chains_full_blocks_only():
    hasher = _fake_hasher()
    short = hasher.hashes([1, 2, 3])
    longer = hasher.hashes([1, 2, 3, 4, 5])
    assert len(short) == 1 and len(longer) == 2
    assert longer[0] == short[0]
    assert hasher.hashes([9, 2, 3, 4]) != longer


def test_block_hash_index_returns_the_deepest_owner():
    index = BlockHashIndex(2)
    index.store(0, ["h1", "h2"])
    index.store(1, ["h1"])
    assert index.longest_match(["h1", "h2", "h3"]) == ({0}, 2)
    assert index.longest_match(["h1"]) == ({0, 1}, 1)
    assert index.longest_match(["zzz"]) == (set(), 0)
    index.remove(0, ["h2"])
    assert index.longest_match(["h1", "h2"]) == ({0, 1}, 1)
    index.clear(0)
    assert index.longest_match(["h1", "h2"]) == ({1}, 1)
    assert len(index) == 1


def test_kv_cache_aware_router_prefers_the_engine_holding_the_prefix():
    hasher = _fake_hasher()
    router = KvCacheAwareRouter(2, hasher=hasher)
    prompt = [1, 2, 3, 4, 5, 6]
    router.index.store(1, hasher.hashes(prompt[:4]))
    router.acquire(0)
    assert router.pick("conv", prompt) == 1
    assert router.hits == 1
    # No match: falls back to least loaded (engine 1 now has one in flight too).
    router.acquire(1)
    router.acquire(1)
    assert router.pick("other", [8, 8, 8, 8]) == 0
    assert router.misses == 1
    # Sticky afterwards.
    assert router.pick("conv") == 1


def test_kv_event_subscriber_applies_batches_to_the_index():
    router = KvCacheAwareRouter(1, hasher=_fake_hasher())
    sub = KvEventSubscriber(
        engine_idx=0,
        endpoint="tcp://x",
        topic="",
        index=router.index,
        lock=router.index_lock,
    )
    stored = type("BlockStored", (), {})()
    stored.block_hashes = ["a", "b"]
    removed = type("BlockRemoved", (), {})()
    removed.block_hashes = ["b"]
    sub.apply(SimpleNamespace(events=[stored]))
    assert router.index.longest_match(["a", "b"]) == ({0}, 2)
    sub.apply(SimpleNamespace(events=[removed]))
    assert router.index.longest_match(["a", "b"]) == ({0}, 1)
    sub.apply(SimpleNamespace(events=[type("AllBlocksCleared", (), {})()]))
    assert len(router.index) == 0
    assert sub.batches_seen == 3


def test_make_router_rejects_unknown_kinds():
    assert isinstance(make_router("round_robin", 2, table_size=10), RoundRobinRouter)
    with pytest.raises(ValueError):
        make_router("sticky", 2, table_size=10)


# -------------------------------------------------------------------------- #
#  Bidirectional state
# -------------------------------------------------------------------------- #


def test_conversation_cache_pops_on_read_expires_and_clears(monkeypatch):
    cache = ConversationKVCache(ttl_s=10)
    cache.put("c", {"remote_block_ids": [[1]]})
    assert cache.get("c") == {"remote_block_ids": [[1]]}
    assert cache.get("c") is None  # consumed by the turn that read it
    cache.put("c", {"remote_block_ids": [[2]]})
    clock = {"t": 1000.0}
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_disagg_frontend.time.monotonic",
        lambda: clock["t"],
    )
    cache.put("d", {"x": 1})
    clock["t"] += 11
    assert cache.get("d") is None
    assert cache.expired == 1
    cache.put("e", {"x": 1})
    assert cache.clear() == 2 and len(cache) == 0


# -------------------------------------------------------------------------- #
#  The relay
# -------------------------------------------------------------------------- #


def _settings(**overrides):
    base = dict(
        prefill_router="conversation",
        decode_router="load_balancing",
        initial_prefill_params=dict(_INITIAL),
        bidirectional_kv_xfer=False,
        conversation_cache_ttl_s=450.0,
        decode_tokids_b64=False,
        decode_strip_message_history=False,
        request_timeout_s=30.0,
        prefill_router_table_size=100,
    )
    base.update(overrides)
    return RelaySettings(**base)


class _FakeEngines:
    """Scripted engines behind ``FrontendRelay._post_json``."""

    def __init__(self, prefill_status=200, decode_status=200, decode_kv=None):
        self.calls = []
        self.prefill_status = prefill_status
        self.decode_status = decode_status
        self.decode_kv = decode_kv

    async def post(self, endpoint, path, payload):
        self.calls.append((endpoint.port, path, payload))
        if endpoint.port < 9000:  # prefill pool
            if self.prefill_status != 200:
                return self.prefill_status, {
                    "error": {
                        "message": "maximum context length",
                        "code": self.prefill_status,
                    }
                }
            return 200, _prefill_response()
        if self.decode_status != 200:
            return self.decode_status, {
                "error": {"message": "boom", "code": self.decode_status}
            }
        body = {
            "id": "chatcmpl-d",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "ok",
                        "generation_token_ids": [7],
                    },
                }
            ],
            "prompt_token_ids": [1, 2, 3, 4, 5, 6],
        }
        if self.decode_kv is not None:
            body["kv_transfer_params"] = dict(self.decode_kv)
        return 200, body


def _relay(engines, **settings_overrides):
    relay = FrontendRelay(
        prefill=[EngineEndpoint("10.0.0.1", 8001), EngineEndpoint("10.0.0.1", 8002)],
        decode=[EngineEndpoint("10.0.0.2", 9001)],
        settings=_settings(**settings_overrides),
    )
    relay._post_json = engines.post  # type: ignore[method-assign]
    relay.mark_engines_ready()
    return relay


def test_relay_runs_both_legs_and_hides_the_decode_handshake_from_gym():
    engines = _FakeEngines(decode_kv={"remote_block_ids": [[3]]})
    relay = _relay(engines)
    status, payload, streaming = asyncio.run(relay.relay(CHAT_PATH, _gym_body()))
    assert status == 200 and not streaming
    assert payload["choices"][0]["message"]["generation_token_ids"] == [7]
    assert "kv_transfer_params" not in payload
    (p_port, p_path, p_leg), (d_port, d_path, d_leg) = engines.calls
    assert p_port in (8001, 8002) and d_port == 9001
    assert p_path == d_path == CHAT_PATH
    assert p_leg["max_tokens"] == 1 and p_leg["kv_transfer_params"] == _INITIAL
    assert d_leg["kv_transfer_params"]["remote_host"] == "10.0.0.1"
    assert d_leg["kv_transfer_params"]["prompt_token_ids"] == [1, 2, 3, 4, 5, 6]
    assert relay.prefill_router.inflight == [0, 0] and relay.decode_router.inflight == [
        0
    ]


def test_relay_keeps_a_conversation_on_one_prefill_engine():
    engines = _FakeEngines()
    relay = _relay(engines)
    for _ in range(3):
        asyncio.run(relay.relay(CHAT_PATH, _gym_body()))
    prefill_ports = {
        port for port, _, leg in engines.calls if leg.get("max_tokens") == 1
    }
    assert len(prefill_ports) == 1


def test_relay_passes_a_prefill_4xx_through_untouched():
    engines = _FakeEngines(prefill_status=400)
    status, payload, _ = asyncio.run(_relay(engines).relay(CHAT_PATH, _gym_body()))
    assert status == 400
    assert "maximum context length" in payload["error"]["message"]
    assert len(engines.calls) == 1  # no decode leg


def test_relay_reports_a_decode_failure():
    engines = _FakeEngines(decode_status=500)
    relay = _relay(engines)
    status, payload, _ = asyncio.run(relay.relay(CHAT_PATH, _gym_body()))
    assert status == 500 and relay.stats["decode_errors"] == 1


def test_relay_answers_503_until_the_engines_are_ready():
    relay = FrontendRelay(
        prefill=[EngineEndpoint("h", 8001)],
        decode=[EngineEndpoint("h", 9001)],
        settings=_settings(),
    )
    status, payload, _ = asyncio.run(relay.relay(CHAT_PATH, _gym_body()))
    assert status == 503 and "still loading" in payload["error"]["message"]


def test_relay_bidirectional_round_trip_feeds_the_next_prefill_leg():
    engines = _FakeEngines(
        decode_kv={
            "remote_block_ids": [[3, 4]],
            "remote_engine_id": "d0",
            "do_remote_decode": True,
        }
    )
    relay = _relay(engines, bidirectional_kv_xfer=True)
    asyncio.run(relay.relay(CHAT_PATH, _gym_body()))
    assert len(relay.conversation_cache) == 1
    asyncio.run(relay.relay(CHAT_PATH, _gym_body()))
    second_prefill_leg = engines.calls[2][2]
    params = second_prefill_leg["kv_transfer_params"]
    assert params["remote_block_ids"] == [[3, 4]]
    assert params["do_remote_decode"] is True and params["do_remote_prefill"] is False
    # Consumed by the turn; re-populated by its decode leg.
    assert len(relay.conversation_cache) == 1
    assert relay.invalidate_conversation_cache() == 1
    assert len(relay.conversation_cache) == 0


def test_relay_uses_the_frontend_renderer_and_falls_back_on_failure():
    class _Renderer:
        def __init__(self, fail=False):
            self.fail = fail

        async def prompt_token_ids(self, body):
            if self.fail:
                raise RuntimeError("no tokenizer")
            return [4, 2]

    engines = _FakeEngines()
    relay = _relay(engines)
    relay.install_renderer(_Renderer())
    asyncio.run(relay.relay(CHAT_PATH, _gym_body()))
    assert engines.calls[0][2]["kv_transfer_params"]["prompt_token_ids"] == [4, 2]
    assert "required_prefix_token_ids" not in engines.calls[0][2]
    assert relay.stats["frontend_tokenized"] == 1

    relay.install_renderer(_Renderer(fail=True))
    asyncio.run(relay.relay(CHAT_PATH, _gym_body()))
    assert "prompt_token_ids" not in engines.calls[2][2]["kv_transfer_params"]
    assert engines.calls[2][2]["required_prefix_token_ids"] == [1, 2, 3, 4, 5]
    assert relay.stats["frontend_tokenize_fallbacks"] == 1


def test_relay_completions_path_is_relayed_too():
    engines = _FakeEngines()
    status, _, _ = asyncio.run(
        _relay(engines).relay(
            COMPLETIONS_PATH, {"model": "m", "prompt": "hi", "max_tokens": 4}
        )
    )
    assert status == 200
    assert engines.calls[1][2]["prompt"] == [1, 2, 3, 4, 5, 6]


def test_frontend_impl_requires_serve_args_to_start():
    impl = DisaggFrontendImpl(replica_idx=0)
    with pytest.raises(AssertionError, match="serve_args"):
        impl.start()
    assert impl.base_url() is None and impl.shutdown() is True


def test_frontend_app_serves_health_and_relays_chat():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from nemo_rl.models.generation.vllm.vllm_disagg_frontend import create_frontend_app

    engines = _FakeEngines()
    relay = _relay(engines)
    client = TestClient(create_frontend_app(relay))
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["prefill_engines"] == 2
    resp = client.post(CHAT_PATH, json=_gym_body())
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["generation_token_ids"] == [7]
    assert client.get("/stats").json()["requests"] == 1
    assert (
        client.post(
            CHAT_PATH, content=b"not json", headers={"content-type": "application/json"}
        ).status_code
        == 400
    )
