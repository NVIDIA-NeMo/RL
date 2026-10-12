# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")
# megatron_worker imports megatron.core at module level; skip when it is absent.
pytest.importorskip("megatron.core")

from nemo_rl.models.generation.megatron.megatron_generation import (  # noqa: E402
    MegatronGeneration,
)
from nemo_rl.models.generation.megatron.megatron_worker import (  # noqa: E402
    MegatronGenerationMixin,
)

pytestmark = pytest.mark.nemo_gym


@pytest.fixture
def inference_loop():
    """Run an asyncio loop on a background thread, mirroring the worker's setup."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    yield loop, thread
    loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=5)
    loop.close()


class _WorkerGroup:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def run_all_workers_single_data(self, method_name: str, **kwargs):
        self.calls.append((method_name, kwargs))
        return [True, False]


def test_generation_setup_token_capture_fans_tq_config_to_workers(monkeypatch):
    generation = object.__new__(MegatronGeneration)
    generation.cfg = {"mcore_generation_config": {"expose_http_server": True}}
    worker_group = _WorkerGroup()
    generation._policy = SimpleNamespace(worker_group=worker_group)
    monkeypatch.setattr(
        "nemo_rl.models.generation.megatron.megatron_generation.ray.get",
        lambda value: value,
    )

    dp_cfg = {"backend": "simple"}
    generation.setup_token_capture(dp_cfg, "rollout_staging")

    assert worker_group.calls == [
        (
            "setup_token_capture",
            {"dp_cfg": dp_cfg, "staging_partition": "rollout_staging"},
        )
    ]


def test_generation_setup_token_capture_requires_exposed_http_server() -> None:
    generation = object.__new__(MegatronGeneration)
    generation.cfg = {"mcore_generation_config": {"expose_http_server": False}}
    worker_group = _WorkerGroup()
    generation._policy = SimpleNamespace(worker_group=worker_group)

    with pytest.raises(ValueError, match="expose_http_server=true"):
        generation.setup_token_capture({"backend": "simple"}, "rollout_staging")

    # The driver-side guard fires before any worker is asked to install hooks.
    assert worker_group.calls == []


@pytest.mark.parametrize("version", [-1, 1.0, "7", True], ids=repr)
def test_worker_rejects_invalid_rollout_weight_versions(monkeypatch, version) -> None:
    monkeypatch.setattr(
        "nemo_rl.models.generation.megatron.megatron_worker.torch.distributed.get_rank",
        lambda: 0,
    )
    worker = object.__new__(MegatronGenerationMixin)
    worker._token_capture_enabled = True
    epochs = []
    worker.inference_client = SimpleNamespace(
        set_generation_epoch=lambda version: epochs.append(version)
    )

    # The check is `type(version) is not int`, so bool (an int subclass) is
    # rejected alongside negative ints, floats, and numeric strings.
    with pytest.raises(
        ValueError, match="rollout weight version must be a non-negative int"
    ):
        worker.set_rollout_weight_version(version)

    assert epochs == []


def test_worker_installs_prompt_preparer_and_stager_only_on_mp_coordinator(
    monkeypatch, inference_loop
):
    installed_sinks = []
    installed_sources = []

    class _Sink:
        def __init__(self, client, *, staging_partition):
            installed_sinks.append((client, staging_partition))

    class _Source:
        def __init__(self, client, *, staging_partition):
            installed_sources.append((client, staging_partition))

    class _Preparer:
        def __init__(self, source):
            self.source = source

    class _Stager:
        def __init__(self, sink):
            self.sink = sink

    monkeypatch.setattr(
        "nemo_rl.data_plane.build_data_plane_client", lambda *_a, **_k: "dp"
    )
    monkeypatch.setattr("nemo_rl.data_plane.tq_token_sink.TQTokenSink", _Sink)
    monkeypatch.setattr("nemo_rl.data_plane.tq_token_sink.TQTokenSource", _Source)
    monkeypatch.setattr(
        "nemo_rl.models.generation.megatron.token_capture.TQMegatronPromptPreparer",
        _Preparer,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.megatron.token_capture.TQMegatronTokenStager",
        _Stager,
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.megatron.megatron_worker.torch.distributed.get_rank",
        lambda: 0,
    )

    worker = object.__new__(MegatronGenerationMixin)
    worker.dynamic_inference_engine = SimpleNamespace(
        payload_stager=None,
        prompt_preparer=None,
        is_mp_coordinator=True,
    )
    loop, loop_thread = inference_loop
    worker._inference_loop = loop
    epochs = []
    worker.inference_client = SimpleNamespace(
        set_generation_epoch=lambda version: epochs.append(
            (version, threading.current_thread())
        )
    )
    worker._token_capture_enabled = False
    worker._request_payload_stager = None
    worker._request_prompt_preparer = None

    assert worker.setup_token_capture({}, "rollout_staging")
    assert (
        worker.dynamic_inference_engine.payload_stager is worker._request_payload_stager
    )
    assert (
        worker.dynamic_inference_engine.prompt_preparer
        is worker._request_prompt_preparer
    )
    assert installed_sinks == [("dp", "rollout_staging")]
    assert installed_sources == [("dp", "rollout_staging")]

    worker.set_rollout_weight_version(7)
    # The client's ZMQ socket is not thread safe and its listener task runs on
    # the inference loop thread, so the epoch send must happen on that thread.
    assert epochs == [(7, loop_thread)]

    follower = object.__new__(MegatronGenerationMixin)
    follower.dynamic_inference_engine = SimpleNamespace(
        payload_stager=None,
        prompt_preparer=None,
        is_mp_coordinator=False,
    )
    follower._token_capture_enabled = False
    follower._request_payload_stager = None
    follower._request_prompt_preparer = None
    assert not follower.setup_token_capture({}, "rollout_staging")
    # Followers accept weight-version stamps even though they host no hooks.
    assert follower._token_capture_enabled is True
    assert follower.dynamic_inference_engine.payload_stager is None
    assert follower.dynamic_inference_engine.prompt_preparer is None
    assert installed_sinks == [("dp", "rollout_staging")]
    assert installed_sources == [("dp", "rollout_staging")]


def test_worker_requires_minf_payload_stager_protocol() -> None:
    worker = object.__new__(MegatronGenerationMixin)
    worker.dynamic_inference_engine = SimpleNamespace(is_mp_coordinator=True)

    with pytest.raises(RuntimeError, match="RequestPayloadStager"):
        worker.setup_token_capture({}, "rollout_staging")
