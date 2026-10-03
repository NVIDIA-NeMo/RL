# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU contracts for the bounded xToken transport (no simulated S1 claim)."""

from copy import deepcopy
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict

from nemo_rl.data_plane.xtoken import (
    XTOKEN_LOGITS_FIELD,
    XTokenTQReceiveResult,
    XTokenTQReference,
    XTokenTQTransport,
    XTokenTransportConfig,
    check_payload_size,
    fetch_logits,
    publish_logits,
    select_tq_nodes,
    validate_tq_support,
)


def test_default_and_positive_bounds():
    cfg = XTokenTransportConfig()
    assert cfg.backend == "ipc"
    assert cfg.max_payload_bytes == 64 * 1024 * 1024
    assert cfg.timeout_s == 120
    for override in ({"backend": "other"}, {"timeout_s": 0}, {"max_payload_bytes": 0}):
        with pytest.raises(ValidationError):
            XTokenTransportConfig(**override)


def support_args():
    policy = {
        "dtensor_cfg": {
            "enabled": True,
            "_v2": True,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
        },
        "train_global_batch_size": 1,
        "train_micro_batch_size": 1,
        "sequence_packing": {"enabled": False},
        "dynamic_batching": {"enabled": False},
    }
    return {
        "data_plane": {"enabled": True, "impl": "transfer_queue", "backend": "simple"},
        "policies": [deepcopy(policy), deepcopy(policy)],
        "num_nodes": 2,
        "gpus_per_node": 1,
        "batch_size": 1,
    }


def test_supported_layout():
    validate_tq_support(**support_args())


@pytest.mark.parametrize(
    "key,value",
    [
        ("num_nodes", 1),
        ("num_nodes", 3),
        ("gpus_per_node", 2),
        ("batch_size", 2),
        ("data_plane", None),
        ("policies", []),
    ],
)
def test_unsupported_layout(key, value):
    args = support_args()
    args[key] = value
    with pytest.raises(ValueError):
        validate_tq_support(**args)


@pytest.mark.parametrize(
    "key,value", [("enabled", False), ("backend", "mooncake_cpu"), ("impl", "local")]
)
def test_unsupported_plane(key, value):
    args = support_args()
    args["data_plane"][key] = value
    with pytest.raises(ValueError):
        validate_tq_support(**args)


@pytest.mark.parametrize(
    "path,value",
    [
        (("dtensor_cfg", "enabled"), False),
        (("dtensor_cfg", "_v2"), False),
        (("dtensor_cfg", "tensor_parallel_size"), 2),
        (("dtensor_cfg", "context_parallel_size"), 2),
        (("sequence_packing", "enabled"), True),
        (("dynamic_batching", "enabled"), True),
        (("train_global_batch_size",), 2),
        (("train_micro_batch_size",), 2),
    ],
)
@pytest.mark.parametrize("index", [0, 1])
def test_unsupported_policy(index, path, value):
    args = support_args()
    target = args["policies"][index]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_tq_support(**args)


def node(node_id, gpu=1, alive=True):
    return {
        "NodeID": node_id,
        "Alive": alive,
        "NodeManagerAddress": node_id,
        "Resources": {"GPU": gpu, f"node:{node_id}": 1},
    }


def test_deterministic_distinct_nodes():
    nodes = [node("z"), node("a"), node("dead", alive=False), node("cpu", gpu=0)]
    assert select_tq_nodes(nodes) == ({"node:a": 0.001}, {"node:z": 0.001})
    assert select_tq_nodes(list(reversed(nodes))) == select_tq_nodes(nodes)


def test_no_local_fallback():
    with pytest.raises(ValueError, match="two live"):
        select_tq_nodes([node("a"), node("dead", alive=False)])
    broken = node("b")
    broken["Resources"].pop("node:b")
    with pytest.raises(ValueError, match="advertise"):
        select_tq_nodes([node("a"), broken])


def test_padded_capacity_boundary():
    check_payload_size(seq_len=64, vocab_size=151936, max_bytes=64 * 1024 * 1024)
    check_payload_size(seq_len=4, vocab_size=8, max_bytes=128)
    with pytest.raises(ValueError, match="exceeds"):
        check_payload_size(seq_len=4, vocab_size=8, max_bytes=127)
    with pytest.raises(ValueError, match="positive"):
        check_payload_size(seq_len=0, vocab_size=8, max_bytes=128)


def reference():
    return XTokenTQReference("run", "step", (1, 4, 8), "teacher", 0.01)


def test_publish_contiguous_cpu_and_roundtrip():
    client = MagicMock()
    payload = torch.arange(32, dtype=torch.float32).reshape(1, 8, 4).transpose(1, 2)
    ref = publish_logits(
        client,
        payload,
        partition_id="run",
        sample_id="step",
        producer_node_id="teacher",
        max_payload_bytes=128,
    )
    fields = client.put_samples.call_args.kwargs["fields"]
    assert fields[XTOKEN_LOGITS_FIELD].device.type == "cpu"
    assert fields[XTOKEN_LOGITS_FIELD].is_contiguous()
    assert ref.nbytes == 128
    client.get_samples.return_value = fields
    received = fetch_logits(
        client, ref, consumer_node_id="student", max_payload_bytes=128
    )
    torch.testing.assert_close(received, payload, rtol=0, atol=0)
    client.get_samples.assert_called_once_with(
        sample_ids=["step"], partition_id="run", select_fields=[XTOKEN_LOGITS_FIELD]
    )


@pytest.mark.parametrize(
    "payload",
    [
        torch.zeros(1, 4, 8, dtype=torch.float16),
        torch.zeros(2, 4, 8),
        torch.zeros(4, 8),
    ],
)
def test_publish_rejects_before_put(payload):
    client = MagicMock()
    with pytest.raises(ValueError, match="dense FP32"):
        publish_logits(
            client,
            payload,
            partition_id="run",
            sample_id="step",
            producer_node_id="teacher",
            max_payload_bytes=128,
        )
    client.put_samples.assert_not_called()


@pytest.mark.parametrize(
    "payload", [torch.zeros(1, 4, 7), torch.zeros(1, 4, 8, dtype=torch.float64)]
)
def test_fetch_rejects_malformed_payload(payload):
    client = MagicMock()
    client.get_samples.return_value = TensorDict(
        {XTOKEN_LOGITS_FIELD: payload}, batch_size=[1]
    )
    with pytest.raises(ValueError, match="shape/dtype"):
        fetch_logits(
            client, reference(), consumer_node_id="student", max_payload_bytes=128
        )


def test_fetch_missing_key_and_same_node():
    client = MagicMock()
    with pytest.raises(ValueError, match="different nodes"):
        fetch_logits(
            client, reference(), consumer_node_id="teacher", max_payload_bytes=128
        )
    client.get_samples.assert_not_called()
    client.get_samples.side_effect = KeyError("step")
    with pytest.raises(KeyError):
        fetch_logits(
            client, reference(), consumer_node_id="student", max_payload_bytes=128
        )


def transport():
    result = XTokenTQTransport(
        config=XTokenTransportConfig(backend="tq"),
        data_plane=support_args()["data_plane"],
        teacher=MagicMock(),
        student=MagicMock(),
    )
    result.client = MagicMock()
    result.client.list_sample_ids.return_value = []
    result._stop_workers = MagicMock()
    return result


def test_unique_run_and_step_keys_and_explicit_cleanup():
    t = transport()
    assert t.partition_id != transport().partition_id
    keys = []
    for _ in range(10):
        with t.step():
            keys.append(t._steps[-1])
            if len(keys) == 1:
                t.client.clear_samples.assert_not_called()
    assert len(set(keys)) == 10
    assert t._steps == []
    assert [call.args for call in t.client.clear_samples.call_args_list] == [
        ([key], t.partition_id) for key in keys
    ]


@pytest.mark.parametrize("stage", ["put", "get", "train"])
def test_failure_stops_before_clearing_and_does_not_retry(stage):
    t = transport()
    events = []
    t._stop_workers.side_effect = lambda: events.append("stop")
    t.client.clear_samples.side_effect = lambda *_: events.append("clear")
    t.teacher.get_full_logits_tq.side_effect = lambda *_, **kw: XTokenTQReference(
        kw["partition_id"], kw["sample_id"], (1, 4, 8), "teacher", 0.1
    )
    t.student.materialize_full_logits_tq.return_value = XTokenTQReceiveResult(
        [], "student", 128, 0.2, 128
    )
    train = MagicMock(side_effect=TimeoutError("train"))
    if stage == "put":
        t.teacher.get_full_logits_tq.side_effect = TimeoutError("put")
    elif stage == "get":
        t.student.materialize_full_logits_tq.side_effect = TimeoutError("get")
    with pytest.raises(TimeoutError, match=stage):
        with t.step():
            t.transfer(MagicMock())
            train()
    assert events == ["stop", "clear"]
    assert t._steps == []
    t.teacher.get_full_logits_tq.assert_called_once()
    assert train.call_count == (1 if stage == "train" else 0)
    assert t.student.materialize_full_logits_tq.call_count == (
        0 if stage == "put" else 1
    )


def test_cleanup_preserves_both_errors():
    t = transport()
    t.client.clear_samples.side_effect = RuntimeError("cleanup")
    with pytest.raises(ExceptionGroup) as caught:
        with t.step():
            raise ValueError("training")
    assert [str(e) for e in caught.value.exceptions] == ["training", "cleanup"]


@pytest.mark.parametrize("terminated", [True, False])
def test_cleanup_waits_for_actor_death(terminated):
    import ray

    t = transport()
    del t._stop_workers  # Exercise the production termination barrier.
    actors = [MagicMock(), MagicMock()]
    t.teacher.worker_group.workers = actors[:1]
    t.student.worker_group.workers = actors[1:]
    error = ray.exceptions.ActorDiedError() if terminated else TimeoutError("stop")
    with patch.object(ray, "kill") as kill, patch.object(ray, "get", side_effect=error):
        with pytest.raises(ValueError if terminated else ExceptionGroup):
            with t.step():
                raise ValueError("training")
    assert kill.call_count == 2
    if terminated:
        t.client.clear_samples.assert_called_once()
    else:
        t.client.clear_samples.assert_not_called()


def test_transfer_is_metadata_only_and_checks_key():
    t = transport()
    with t.step():
        t.teacher.get_full_logits_tq.return_value = XTokenTQReference(
            t.partition_id, t._steps[-1], (1, 4, 8), "teacher", 0.1
        )
        t.student.materialize_full_logits_tq.return_value = XTokenTQReceiveResult(
            [{"teacher_shards": [{"payload_ipc": "student-local"}]}],
            "student",
            128,
            0.2,
            128,
        )
        assert (
            t.transfer(MagicMock())[0]["teacher_shards"][0]["payload_ipc"]
            == "student-local"
        )
        assert t.metrics["put_payload_bytes"] == t.metrics["get_payload_bytes"] == 128
    t = transport()
    t.teacher.get_full_logits_tq.return_value = reference()
    with pytest.raises(ValueError, match="stale or foreign"):
        with t.step():
            t.transfer(MagicMock())
    t.student.materialize_full_logits_tq.assert_not_called()
