# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

nemo_gym = pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_rl.models.generation.megatron.megatron_generation import (  # noqa: E402
    MegatronGeneration,
)
from nemo_rl.models.generation.megatron.megatron_worker import (  # noqa: E402
    MegatronGenerationMixin,
)

pytestmark = pytest.mark.nemo_gym


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


def test_worker_installs_prompt_preparer_and_stager_only_on_mp_coordinator(
    monkeypatch,
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
        "nemo_rl.data_plane.tq_token_sink.TQMegatronPromptPreparer", _Preparer
    )
    monkeypatch.setattr(
        "nemo_rl.data_plane.tq_token_sink.TQMegatronTokenStager", _Stager
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
    epochs = []
    worker.inference_client = SimpleNamespace(
        set_generation_epoch=lambda version: epochs.append(version)
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
    assert epochs == [7]

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
    assert follower.dynamic_inference_engine.payload_stager is None
    assert follower.dynamic_inference_engine.prompt_preparer is None
    assert installed_sinks == [("dp", "rollout_staging")]
    assert installed_sources == [("dp", "rollout_staging")]


def test_worker_requires_minf_payload_stager_protocol() -> None:
    worker = object.__new__(MegatronGenerationMixin)
    worker.dynamic_inference_engine = SimpleNamespace(is_mp_coordinator=True)

    with pytest.raises(RuntimeError, match="RequestPayloadStager"):
        worker.setup_token_capture({}, "rollout_staging")
