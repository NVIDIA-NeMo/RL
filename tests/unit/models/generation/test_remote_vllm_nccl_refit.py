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

import json
import threading
import urllib.request
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from nemo_rl.models.generation.remote_vllm import (
    RemoteVllmClient,
    RemoteVllmGeneration,
    RemoteVllmServiceConfig,
)
from nemo_rl.weight_sync import remote_vllm_nccl_weight_synchronizer as sync_mod
from nemo_rl.weight_sync.factory import create_weight_synchronizer


class _Response:
    def __init__(self, body: dict[str, Any] | None = None) -> None:
        self._payload = b"" if body is None else json.dumps(body).encode()

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def _config_dict(**refit: Any) -> dict[str, Any]:
    return {
        "base_url": "http://rollout.test:9210/v1",
        "served_model_name": "policy",
        "max_model_len": 4096,
        "refit": refit,
    }


def _master_config(refit_transport: str | None, **refit: Any) -> SimpleNamespace:
    return SimpleNamespace(
        env={"should_use_nemo_gym": True},
        policy={
            "megatron_cfg": {"enabled": True},
            "generation": {
                "colocated": {"enabled": False},
                "refit_transport": refit_transport,
                "remote_vllm_cfg": _config_dict(**refit),
            },
        },
        token_capture=SimpleNamespace(enabled=False),
    )


def test_external_transport_is_nccl_only() -> None:
    config = RemoteVllmServiceConfig.model_validate(_config_dict())
    assert config.refit.transport == "nccl"
    assert config.refit.nccl_num_engines == 1
    RemoteVllmGeneration.validate_settings(_master_config(None))
    with pytest.raises(ValueError, match="must be unset"):
        RemoteVllmGeneration.validate_settings(_master_config("hf_checkpoint"))
    for refit in ({"transport": "hf_checkpoint"}, {"nccl_num_engines": 2}):
        with pytest.raises(ValueError):
            RemoteVllmServiceConfig.model_validate(_config_dict(**refit))


def test_weight_transfer_requests_match_vllm_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[tuple[str, dict[str, Any], float]] = []

    def urlopen(request: urllib.request.Request, timeout: float) -> _Response:
        requests.append((request.full_url, json.loads(request.data or b""), timeout))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    client = RemoteVllmClient(
        RemoteVllmServiceConfig.model_validate(_config_dict(timeout_s=60))
    )
    init_info = {"rank_offset": 16, "world_size": 18, "packed": True}
    client.init_weight_transfer_engine(init_info)
    client.start_weight_update()
    client.update_weights(
        {"names": ["w"], "dtype_names": ["bfloat16"], "shapes": [[1]]}
    )
    client.finish_weight_update("3")

    origin = "http://rollout.test:9210"
    assert requests == [
        (
            f"{origin}/init_weight_transfer_engine",
            {"init_info": init_info},
            60,
        ),
        (f"{origin}/start_weight_update", {}, 60),
        (
            f"{origin}/update_weights",
            {
                "update_info": {
                    "names": ["w"],
                    "dtype_names": ["bfloat16"],
                    "shapes": [[1]],
                }
            },
            60,
        ),
        (f"{origin}/finish_weight_update", {"weight_version": "3"}, 60),
    ]


class _Ref:
    def __init__(self, done: bool = True) -> None:
        self.done = threading.Event()
        if done:
            self.done.set()


class _FakeRay:
    @staticmethod
    def wait(refs, num_returns, timeout):  # noqa: ANN001, ANN205
        for ref in refs:
            ref.done.wait(timeout=0.01)
        return [r for r in refs if r.done.is_set()], [
            r for r in refs if not r.done.is_set()
        ]

    @staticmethod
    def get(refs):  # noqa: ANN001, ANN205
        return [True for _ in refs]


class _FakeClient:
    def __init__(self, calls: list, health: dict | None, fail_update: bool) -> None:
        self.calls = calls
        self._health = health
        self._fail_update = fail_update

    def health(self, *, timeout_s: float | None = None) -> dict | None:
        return self._health

    def init_weight_transfer_engine(self, init_info):  # noqa: ANN001, ANN201
        self.calls.append(("init", init_info))

    def start_weight_update(self) -> None:
        self.calls.append(("start",))

    def update_weights(self, update_info: dict) -> None:
        self.calls.append(("update", update_info))
        if self._fail_update:
            raise RuntimeError("External vLLM returned HTTP 400")

    def finish_weight_update(self, weight_version: str) -> None:
        self.calls.append(("finish", weight_version))


def _setup(
    monkeypatch: pytest.MonkeyPatch,
    *,
    health: dict | None,
    fail_update: bool = False,
    broadcast_done: bool = True,
    **refit: Any,
):
    monkeypatch.setattr(sync_mod, "ray", _FakeRay)
    calls: list = []
    generation = RemoteVllmGeneration(
        {"remote_vllm_cfg": _config_dict(**refit)}  # type: ignore[typeddict-item]
    )
    generation.client = _FakeClient(calls, health, fail_update)  # type: ignore[assignment]
    generation.pause_generation_for_refit = (  # type: ignore[method-assign]
        lambda clear_cache: calls.append(("pause", clear_cache)) or True
    )
    generation.resume_generation_after_refit = (  # type: ignore[method-assign]
        lambda: calls.append(("resume",)) or True
    )
    info = {
        "backbone.layers.1.mixer.experts.0.up_proj.weight": (
            torch.Size([3, 2]),
            torch.bfloat16,
        ),
        "backbone.layers.0.mixer.A_log": (torch.Size([4]), torch.float32),
    }

    def broadcast(**kwargs: Any) -> list[_Ref]:
        calls.append(("broadcast", kwargs))
        return [_Ref(done=broadcast_done)]

    policy = SimpleNamespace(
        cfg={"megatron_cfg": {"enabled": True}, "offload_policy_before_refit": False},
        prepare_refit_info=lambda: info,
        init_collective=lambda ip, port, world_size, train_world_size, nccl_peer: (
            calls.append(("policy_init", world_size, train_world_size, nccl_peer))
            or [_Ref()]
        ),
        broadcast_weights_for_collective=broadcast,
    )
    cluster = SimpleNamespace(
        world_size=lambda: 16, get_master_address_and_port=lambda: ("10.0.0.1", 4242)
    )
    synchronizer = create_weight_synchronizer(
        policy, generation, "remote_vllm", False, train_cluster=cluster
    )
    return calls, synchronizer


def test_init_contains_one_deployments_gpu_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, synchronizer = _setup(
        monkeypatch,
        health={"total_backends": 1, "healthy_backends": 1},
        nccl_engine_world_size=2,
    )
    assert isinstance(synchronizer, sync_mod.RemoteVllmNcclWeightSynchronizer)
    synchronizer.init_communicator()
    assert ("policy_init", 18, 16, "vllm") in calls
    init = next(c for c in calls if c[0] == "init")
    assert init[1] == {
        "master_address": "10.0.0.1",
        "master_port": 4242,
        "rank_offset": 16,
        "world_size": 18,
        "packed": True,
        "packed_buffer_size_bytes": 1024**3,
        "packed_num_buffers": 2,
    }
    synchronizer.shutdown()


def test_single_server_without_lb_health_is_one_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, synchronizer = _setup(monkeypatch, health=None)
    synchronizer.init_communicator()
    assert ("policy_init", 17, 16, "vllm") in calls


@pytest.mark.parametrize("control_fanout", [False, True])
def test_init_rejects_multiple_backends_before_any_policy_collective(
    monkeypatch: pytest.MonkeyPatch, control_fanout: bool
) -> None:
    calls, synchronizer = _setup(
        monkeypatch,
        health={
            "total_backends": 2,
            "healthy_backends": 2,
            "control_fanout": control_fanout,
        },
    )
    with pytest.raises(RuntimeError, match="exactly one engine deployment"):
        synchronizer.init_communicator()
    assert calls == []
    synchronizer.shutdown()


def test_refit_sequence_and_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, synchronizer = _setup(monkeypatch, health=None)
    synchronizer.init_communicator()
    calls.clear()

    synchronizer.sync_weights()
    synchronizer.sync_weights()

    kinds = [c[0] for c in calls]
    assert kinds[:6] == ["pause", "start", "broadcast", "update", "finish", "resume"]
    assert calls[4] == ("finish", "1") and ("finish", "2") in calls
    assert calls[2][1] == {
        "refit_timeout_s": 1800.0,
        "buffer_size_bytes": 1024**3,
        "num_buffers": 2,
        "use_legacy_packing": True,
    }
    # NCCLWeightTransferUpdateInfo has no "packed" field.
    assert calls[3][1] == {
        "names": [
            "backbone.layers.1.mixer.experts.0.up_proj.weight",
            "backbone.layers.0.mixer.A_log",
        ],
        "dtype_names": ["bfloat16", "float32"],
        "shapes": [[3, 2], [4]],
    }
    assert not synchronizer.is_stale


def test_failed_update_request_fails_refit_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, synchronizer = _setup(
        monkeypatch, health=None, fail_update=True, broadcast_done=False
    )
    synchronizer.init_communicator()
    with pytest.raises(RuntimeError, match="update_weights failed"):
        synchronizer.sync_weights()
    assert ("resume",) not in calls and ("finish", "1") not in calls
    assert synchronizer.is_stale


def test_failed_update_cannot_be_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, synchronizer = _setup(monkeypatch, health=None, fail_update=True)
    synchronizer.init_communicator()
    with pytest.raises(RuntimeError):
        synchronizer.sync_weights()
    calls.clear()
    with pytest.raises(RuntimeError, match="restart the training job"):
        synchronizer.sync_weights()
    assert calls == []
    synchronizer.shutdown()


def test_shutdown_is_idempotent_and_does_not_resume_serving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, synchronizer = _setup(monkeypatch, health=None)
    synchronizer.init_communicator()
    calls.clear()
    synchronizer.shutdown()
    synchronizer.shutdown()
    assert calls == []
    with pytest.raises(RuntimeError, match="shut down"):
        synchronizer.sync_weights()


def test_missing_policy_receiver_has_a_controller_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, synchronizer = _setup(
        monkeypatch, health=None, broadcast_done=False, timeout_s=0.02
    )
    synchronizer.init_communicator()
    with pytest.raises(TimeoutError):
        synchronizer.sync_weights()
    assert ("resume",) not in calls
    synchronizer.shutdown()


def test_kv_scales_rejected_before_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, synchronizer = _setup(monkeypatch, health=None)
    synchronizer.init_communicator()
    calls.clear()
    with pytest.raises(NotImplementedError, match="KV scales"):
        synchronizer.sync_weights(kv_scales={"layer": 1.0})
    assert calls == []
    synchronizer.shutdown()
