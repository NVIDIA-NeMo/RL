"""Worker-side PD disaggregation: engine identity, connector kwargs, shadow check."""

import asyncio
import copy
import logging
import os
from types import SimpleNamespace

import pytest

from nemo_rl.models.generation.vllm import vllm_worker
from nemo_rl.models.generation.vllm.config import resolve_vllm_role_overrides
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)

_WORKER_LOGGER = "nemo_rl.models.generation.vllm.vllm_worker_async"


def _disagg_config(**overrides):
    vllm_cfg = {
        "tensor_parallel_size": 2,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
        "gpu_memory_utilization": 0.6,
        "max_model_len": 4096,
        "precision": "bfloat16",
        "async_engine": True,
        "expose_http_server": True,
        "skip_tokenizer_init": False,
        "kv_cache_dtype": "auto",
    }
    vllm_cfg.update(overrides)
    return resolve_vllm_role_overrides(
        {
            "backend": "vllm",
            "model_name": "test/model",
            "colocated": {
                "enabled": False,
                "resources": {"gpus_per_node": None, "num_nodes": None},
            },
            "disaggregation": {"enabled": True, "frontend_tokenize": True},
            "vllm_cfg": vllm_cfg,
            "vllm_kwargs": {},
        }
    )


# -------------------------------------------------------------------------- #
#  Engine identity and connector kwargs
# -------------------------------------------------------------------------- #


def test_apply_disagg_engine_identity_exports_the_drivers_ports_and_host(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(vllm_worker, "_get_node_ip_local", lambda: "10.1.2.3")
    for name in (
        "VLLM_PORT",
        "VLLM_NIXL_SIDE_CHANNEL_PORT",
        "VLLM_NIXL_SIDE_CHANNEL_HOST",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "vllm-cache"))

    worker = BaseVllmGenerationWorker.__new__(BaseVllmGenerationWorker)
    worker._disagg_engine = {
        "engine_index": 4,
        "seed": 4,
        "env": {"VLLM_PORT": 7100, "VLLM_NIXL_SIDE_CHANNEL_PORT": 8557},
        "side_channel_host_env": "VLLM_NIXL_SIDE_CHANNEL_HOST",
    }

    # The driver's seed replaces configure_worker's bundle-derived one.
    assert worker._apply_disagg_engine_identity(seed=99) == 4
    assert os.environ["VLLM_PORT"] == "7100"
    assert os.environ["VLLM_NIXL_SIDE_CHANNEL_PORT"] == "8557"
    # The connector advertises this host; "localhost" would send the peer to itself.
    assert os.environ["VLLM_NIXL_SIDE_CHANNEL_HOST"] == "10.1.2.3"
    # Engines on one node must not share a torch.compile cache directory.
    assert os.environ["VLLM_CACHE_ROOT"] == str(tmp_path / "vllm-cache_pd4")


def test_apply_disagg_engine_identity_without_a_side_channel_sets_no_host(
    monkeypatch,
):
    monkeypatch.setattr(vllm_worker, "_get_node_ip_local", lambda: "10.1.2.3")
    monkeypatch.delenv("VLLM_NIXL_SIDE_CHANNEL_HOST", raising=False)
    monkeypatch.delenv("VLLM_CACHE_ROOT", raising=False)
    worker = BaseVllmGenerationWorker.__new__(BaseVllmGenerationWorker)
    worker._disagg_engine = {
        "engine_index": 0,
        "seed": 0,
        "env": {"VLLM_PORT": 7000},
        "side_channel_host_env": None,
    }

    assert worker._apply_disagg_engine_identity(seed=None) == 0
    assert "VLLM_NIXL_SIDE_CHANNEL_HOST" not in os.environ
    assert os.environ["VLLM_CACHE_ROOT"].endswith("/.cache/vllm_pd0")


@pytest.mark.vllm
def test_add_disagg_engine_kwargs_builds_the_connector_and_events_configs():
    from vllm.config import KVEventsConfig, KVTransferConfig

    worker = BaseVllmGenerationWorker.__new__(BaseVllmGenerationWorker)
    worker.cfg = _disagg_config(
        disagg_frontend={
            "prefill_router": "kv_cache_aware",
            "bidirectional_kv_xfer": True,
        },
        kv_transfer={"kv_connector_extra_config": {"backends": ["UCX"]}},
    )
    worker._disagg_role = "prefill"
    worker._disagg_engine = {
        "engine_id": "replica0-prefill0",
        "kv_events_port": 8572,
        "kv_events_topic": "nemo-rl-kv",
    }
    llm_kwargs: dict = {}

    worker._add_disagg_engine_kwargs(llm_kwargs)

    kv = llm_kwargs["kv_transfer_config"]
    assert isinstance(kv, KVTransferConfig)
    assert (kv.kv_connector, kv.kv_role, kv.engine_id) == (
        "NixlConnector",
        "kv_producer",
        "replica0-prefill0",
    )
    assert kv.kv_connector_extra_config == {
        "backends": ["UCX"],
        "bidirectional_kv_xfer": True,
    }
    assert (kv.kv_load_failure_policy, kv.kv_buffer_device) == ("fail", "cuda")
    events = llm_kwargs["kv_events_config"]
    assert isinstance(events, KVEventsConfig)
    assert (
        events.enable_kv_cache_events,
        events.publisher,
        events.endpoint,
        events.topic,
    ) == (True, "zmq", "tcp://*:8572", "nemo-rl-kv")

    # A decode engine consumes KV and publishes no events.
    worker._disagg_role = "decode"
    worker._disagg_engine = {"engine_id": "replica0-decode0"}
    decode_kwargs: dict = {}
    worker._add_disagg_engine_kwargs(decode_kwargs)
    assert decode_kwargs["kv_transfer_config"].kv_role == "kv_consumer"
    assert "kv_events_config" not in decode_kwargs


# -------------------------------------------------------------------------- #
#  Shadow validation of frontend-supplied prompt ids
# -------------------------------------------------------------------------- #

_MESSAGES = [
    {"role": "user", "content": "q"},
    {
        "role": "assistant",
        "content": "a",
        "prompt_token_ids": [1, 2],
        "generation_token_ids": [3],
    },
    {"role": "user", "content": "q2"},
]


def _leg_request(supplied):
    request = SimpleNamespace(
        kv_transfer_params={
            "do_remote_decode": True,
            "prompt_token_ids": list(supplied),
        }
    )
    request.model_copy = lambda update: SimpleNamespace(**update)
    return request


class _ShadowRenderer:
    def __init__(self, recomputed=None, error=None):
        self.recomputed = recomputed
        self.error = error
        self.calls = []

    async def preprocess_chat(self, request, messages, **render_kwargs):
        self.calls.append((request, messages, render_kwargs))
        if self.error is not None:
            raise self.error
        return None, [{"prompt_token_ids": list(self.recomputed)}]


def test_shadow_check_recomputes_from_the_messages_and_logs_only_a_mismatch(
    caplog,
):
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    caplog.set_level(logging.ERROR, logger=_WORKER_LOGGER)
    render_kwargs = {"default_template": None, "parser": None}

    renderer = _ShadowRenderer(recomputed=[1, 2, 3, 4])
    asyncio.run(
        worker._shadow_check_supplied_prompt_ids(
            renderer,
            _leg_request([1, 2, 3, 4]),
            _MESSAGES,
            [1, 2, 3, 4],
            render_kwargs,
        )
    )

    [(shadow, messages, kwargs)] = renderer.calls
    # The supplied ids are withheld so the shadow render really renders, and
    # it splices from the Gym fields the way an unassisted request would.
    assert shadow.kv_transfer_params == {"do_remote_decode": True}
    assert shadow.required_prefix_token_ids == [1, 2, 3]
    assert (
        shadow.prompt_token_ids,
        shadow.prompt_token_ids_b64,
        shadow.ng_capture,
    ) == (None, None, None)
    assert messages == _MESSAGES and messages is not _MESSAGES
    assert kwargs == render_kwargs
    assert caplog.records == []

    renderer = _ShadowRenderer(recomputed=[1, 2, 9, 4])
    asyncio.run(
        worker._shadow_check_supplied_prompt_ids(
            renderer,
            _leg_request([1, 2, 3, 4]),
            copy.deepcopy(_MESSAGES),
            [1, 2, 3, 4],
            {},
        )
    )
    assert "SHADOW MISMATCH" in caplog.text
    assert "first divergence at 2" in caplog.text


def test_shadow_check_never_fails_serving(caplog):
    worker = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    caplog.set_level(logging.ERROR, logger=_WORKER_LOGGER)

    asyncio.run(
        worker._shadow_check_supplied_prompt_ids(
            _ShadowRenderer(error=RuntimeError("no template")),
            _leg_request([1]),
            _MESSAGES,
            [1],
            {},
        )
    )

    assert "shadow recompute failed" in caplog.text
