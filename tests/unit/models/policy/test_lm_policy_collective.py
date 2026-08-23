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

from types import SimpleNamespace

import torch

from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.weight_sync.interfaces import WeightSyncSelection


def test_policy_forwards_nccl_peer_to_workers():
    calls = []

    class WorkerGroup:
        def run_all_workers_single_data(self, method_name, **kwargs):
            calls.append((method_name, kwargs))
            return ["future"]

        def shutdown(self, **_kwargs):
            pass

    policy = Policy.__new__(Policy)
    policy.worker_group = WorkerGroup()

    futures = policy.init_collective(
        "127.0.0.1",
        1234,
        4,
        train_world_size=2,
        nccl_peer="vllm",
    )

    assert futures == ["future"]
    assert calls == [
        (
            "init_collective",
            {
                "ip": "127.0.0.1",
                "port": 1234,
                "world_size": 4,
                "train_world_size": 2,
                "nccl_peer": "vllm",
            },
        )
    ]


def test_policy_worker_initializes_requested_nccl_peer(monkeypatch):
    calls = []

    class ProcessGroup:
        def __init__(self, **kwargs):
            calls.append(("create", kwargs))

        def init_nccl_communicator(self, **kwargs):
            calls.append(("init", kwargs))

    monkeypatch.setattr(
        "nemo_rl.distributed.stateless_process_group.StatelessProcessGroup",
        ProcessGroup,
    )
    monkeypatch.setattr(
        "nemo_rl.models.policy.workers.base_policy_worker.torch.cuda.current_device",
        lambda: 3,
    )

    worker = AbstractPolicyWorker.__new__(AbstractPolicyWorker)
    worker.rank = 1
    worker.init_collective(
        "127.0.0.1",
        1234,
        4,
        train_world_size=2,
        nccl_peer="vllm",
    )

    assert calls == [
        (
            "create",
            {
                "master_address": "127.0.0.1",
                "port": 1234,
                "rank": 1,
                "world_size": 4,
            },
        ),
        ("init", {"device": 3, "peer": "vllm"}),
    ]


def test_policy_forwards_packed_collective_options_to_workers():
    calls = []

    class WorkerGroup:
        def run_all_workers_single_data(self, method_name, **kwargs):
            calls.append((method_name, kwargs))
            return ["future"]

        def shutdown(self, **_kwargs):
            pass

    policy = Policy.__new__(Policy)
    policy.worker_group = WorkerGroup()

    futures = policy.broadcast_weights_for_collective(
        kv_scales={"k_scale": 1.25},
        buffer_size_bytes=1024**3,
        num_buffers=2,
    )

    assert futures == ["future"]
    assert calls == [
        (
            "broadcast_weights_for_collective",
            {
                "kv_scales": {"k_scale": 1.25},
                "buffer_size_bytes": 1024**3,
                "num_buffers": 2,
            },
        )
    ]


def test_policy_forwards_target_only_selection_to_collective_worker():
    calls = []

    class WorkerGroup:
        def run_all_workers_single_data(self, method_name, **kwargs):
            calls.append((method_name, kwargs))
            return ["future"]

    policy = Policy.__new__(Policy)
    policy.worker_group = WorkerGroup()

    policy.broadcast_weights_for_collective(selection=WeightSyncSelection(draft=False))

    assert calls == [
        (
            "broadcast_weights_for_collective",
            {
                "kv_scales": None,
                "buffer_size_bytes": None,
                "num_buffers": None,
                "selection": WeightSyncSelection(draft=False),
            },
        )
    ]


def test_policy_forwards_target_only_selection_to_ipc_worker():
    calls = []

    class WorkerGroup:
        def run_all_workers_single_data(self, method_name, **kwargs):
            calls.append((method_name, kwargs))
            return ["future"]

    policy = Policy.__new__(Policy)
    policy.worker_group = WorkerGroup()

    policy.stream_weights_via_ipc_zmq(
        buffer_size_bytes=1024,
        selection=WeightSyncSelection(draft=False),
    )

    assert calls == [
        (
            "stream_weights_via_ipc_zmq",
            {
                "buffer_size_bytes": 1024,
                "kv_scales": None,
                "selection": WeightSyncSelection(draft=False),
            },
        )
    ]


def test_megatron_worker_target_only_skips_draft_preflight_and_payload(
    monkeypatch,
):
    from nemo_rl.models.policy import utils as policy_utils
    from nemo_rl.models.policy.workers import megatron_policy_worker as worker_module

    worker_cls = worker_module.MegatronPolicyWorkerImpl
    worker = object.__new__(worker_cls)
    worker.rank = 0
    worker.model = object()
    worker.refit_conversion_tasks = []
    worker.cfg = {"generation": {"backend": "vllm", "vllm_cfg": {}}}
    worker.megatron_bridge = SimpleNamespace(
        export_hf_weights=lambda *_args, **_kwargs: iter(
            [("target.weight", torch.ones(2, dtype=torch.float32))]
        ),
        transformer_config=SimpleNamespace(num_layers=0),
    )

    draft_preflight_calls = []

    def preflight():
        draft_preflight_calls.append("draft_pp_collective")
        return (("draft.weight", torch.ones(3, dtype=torch.float32)),), None

    worker._preflight_draft_weights_for_refit = preflight
    worker.maybe_init_zmq = lambda: None
    worker.zmq_socket = object()
    payloads = []
    monkeypatch.setattr(
        policy_utils,
        "stream_weights_via_ipc_zmq_impl",
        lambda *, params_generator, **_kwargs: payloads.append(list(params_generator)),
    )

    for selection in (
        WeightSyncSelection(),
        WeightSyncSelection(draft=False),
        WeightSyncSelection(),
    ):
        worker.stream_weights_via_ipc_zmq(selection=selection)

    assert draft_preflight_calls == ["draft_pp_collective", "draft_pp_collective"]
    assert [[name for name, _ in payload] for payload in payloads] == [
        ["target.weight", "draft.weight"],
        ["target.weight"],
        ["target.weight", "draft.weight"],
    ]
    assert [
        sum(tensor.numel() * tensor.element_size() for _, tensor in payload)
        for payload in payloads
    ] == [20, 8, 20]


def test_megatron_collective_target_only_skips_draft_preflight_and_payload(
    monkeypatch,
):
    from nemo_rl.models.policy.workers import megatron_policy_worker as worker_module

    worker_cls = worker_module.MegatronPolicyWorkerImpl
    worker = object.__new__(worker_cls)
    worker.model = object()
    worker.refit_conversion_tasks = []
    worker.cfg = {"generation": {"backend": "vllm", "vllm_cfg": {}}}
    worker.megatron_bridge = SimpleNamespace(
        export_hf_weights=lambda *_args, **_kwargs: iter(
            [("target.weight", torch.ones(2, dtype=torch.float32))]
        ),
        transformer_config=SimpleNamespace(num_layers=0),
    )
    worker.model_update_group = object()
    draft_preflight_calls = []

    def preflight():
        draft_preflight_calls.append("draft_pp_collective")
        return (("draft.weight", torch.ones(3, dtype=torch.float32)),), None

    worker._preflight_draft_weights_for_refit = preflight
    payloads = []
    monkeypatch.setattr(
        worker_module,
        "packed_broadcast_producer",
        lambda *, iterator, **_kwargs: payloads.append(list(iterator)),
    )

    for selection in (
        WeightSyncSelection(),
        WeightSyncSelection(draft=False),
        WeightSyncSelection(),
    ):
        worker.broadcast_weights_for_collective(selection=selection)

    assert draft_preflight_calls == ["draft_pp_collective", "draft_pp_collective"]
    assert [[name for name, _ in payload] for payload in payloads] == [
        ["target.weight", "draft.weight"],
        ["target.weight"],
        ["target.weight", "draft.weight"],
    ]
    assert [
        sum(tensor.numel() * tensor.element_size() for _, tensor in payload)
        for payload in payloads
    ] == [20, 8, 20]
