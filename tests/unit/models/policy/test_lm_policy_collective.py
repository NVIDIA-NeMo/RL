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

from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker


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
