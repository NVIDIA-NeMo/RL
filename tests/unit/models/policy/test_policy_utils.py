# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import sys
import types
from unittest.mock import MagicMock

import pytest
import requests
import torch

from nemo_rl.models.policy import utils as policy_utils

pytestmark = pytest.mark.sglang


def test_setup_ipc_gather_group_returns_none_when_dist_uninit(monkeypatch):
    monkeypatch.setattr(policy_utils.dist, "is_initialized", lambda: False)

    group, src, ranks = policy_utils._setup_ipc_gather_group(
        rank=0,
        current_device_uuid="uuid0",
        sglang_gpu_uuids=["uuid0"],
        sglang_url_to_gpu_uuids={"http://sglang": ["uuid0"]},
    )

    assert group is None
    assert src is None
    assert ranks is None


def test_setup_ipc_gather_group_selects_matching_ranks(monkeypatch):
    all_ranks = ["uuid0", "uuid1", "uuid2", "uuid3"]

    monkeypatch.setattr(policy_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(policy_utils.dist, "get_world_size", lambda: 4)
    monkeypatch.setattr(policy_utils.dist, "get_rank", lambda: 1)

    def fake_all_gather_object(output_list, _value):
        for idx, item in enumerate(all_ranks):
            output_list[idx] = item

    monkeypatch.setattr(policy_utils.dist, "all_gather_object", fake_all_gather_object)

    group, src, ranks = policy_utils._setup_ipc_gather_group(
        rank=1,
        current_device_uuid="uuid1",
        sglang_gpu_uuids=["uuid1", "uuid3"],
        sglang_url_to_gpu_uuids={"http://sglang": ["uuid1", "uuid3"]},
    )

    assert group is None
    assert src == 1
    assert ranks == [1, 3]


def test_gather_ipc_handlers_returns_filtered_on_src(monkeypatch):
    handlers = ["h0", "h1", "h2", "h3"]
    monkeypatch.setattr(policy_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(policy_utils.dist, "get_world_size", lambda: 4)

    def fake_all_gather_object(output_list, _value):
        for idx, item in enumerate(handlers):
            output_list[idx] = item

    monkeypatch.setattr(policy_utils.dist, "all_gather_object", fake_all_gather_object)

    gathered = policy_utils._gather_ipc_handlers(
        serialized_handler="h1",
        gather_group=None,
        gather_src=0,
        rank=0,
        matching_ranks=[0, 2],
    )

    assert gathered == ["h0", "h2"]


def test_gather_ipc_handlers_non_src_returns_none(monkeypatch):
    monkeypatch.setattr(policy_utils.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(policy_utils.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(policy_utils.dist, "all_gather_object", lambda *_args: None)

    gathered = policy_utils._gather_ipc_handlers(
        serialized_handler="h1",
        gather_group=None,
        gather_src=0,
        rank=1,
        matching_ranks=[0, 1],
    )

    assert gathered is None


def test_send_tensor_to_sglang_http_error(monkeypatch):
    response = MagicMock()
    response.raise_for_status.side_effect = requests.exceptions.HTTPError("boom")
    response.status_code = 500
    response.text = "error"
    monkeypatch.setattr(
        policy_utils.requests, "post", lambda *_args, **_kwargs: response
    )

    with pytest.raises(RuntimeError, match="Failed to send tensor 'w'"):
        policy_utils._send_tensor_to_sglang(
            url="http://sglang/update",
            tensor_name="w",
            gathered_handlers=["h0"],
            shape=torch.Size([1]),
            dtype="torch.float32",
        )


def test_send_tensor_to_sglang_generic_error(monkeypatch):
    def raise_error(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(policy_utils.requests, "post", raise_error)

    with pytest.raises(RuntimeError, match="Failed to send tensor 'w'"):
        policy_utils._send_tensor_to_sglang(
            url="http://sglang/update",
            tensor_name="w",
            gathered_handlers=["h0"],
            shape=torch.Size([1]),
            dtype="torch.float32",
        )


def test_stream_weights_via_http_impl_no_matching_url(monkeypatch):
    monkeypatch.setattr(policy_utils.torch.cuda, "empty_cache", lambda: None)

    with pytest.raises(RuntimeError, match="No matching SGLang server"):
        policy_utils.stream_weights_via_http_impl(
            params_generator=iter([]),
            sglang_url_to_gpu_uuids={"http://sglang": ["uuid0"]},
            rank=0,
            worker_name="worker",
            current_device_uuid="uuid1",
        )


def test_stream_weights_via_http_impl_sends_tensors(monkeypatch):
    def params_generator():
        yield "w1", torch.tensor([1.0])
        yield "w2", torch.tensor([2.0])

    dummy_module = types.ModuleType(
        "nemo_rl.models.generation.sglang.sglang_copied_utils"
    )

    class DummySerializer:
        @staticmethod
        def serialize(*_args, **_kwargs):
            return "handler"

    dummy_module.MultiprocessingSerializer = DummySerializer
    monkeypatch.setitem(
        sys.modules,
        "nemo_rl.models.generation.sglang.sglang_copied_utils",
        dummy_module,
    )
    monkeypatch.setattr(policy_utils.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        policy_utils.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(
        policy_utils.torch.Tensor, "cuda", lambda self: self, raising=False
    )

    send_calls = []

    def fake_send_tensor_to_sglang(
        url, name, gathered_handlers, shape, dtype, flush_cache=False
    ):
        send_calls.append(
            {
                "url": url,
                "name": name,
                "handlers": gathered_handlers,
                "shape": shape,
                "dtype": dtype,
                "flush_cache": flush_cache,
            }
        )

    monkeypatch.setattr(
        policy_utils,
        "_setup_ipc_gather_group",
        lambda *_args, **_kwargs: (None, 0, [0]),
    )
    monkeypatch.setattr(
        policy_utils, "_gather_ipc_handlers", lambda *_args, **_kwargs: ["handler"]
    )
    monkeypatch.setattr(
        policy_utils, "_send_tensor_to_sglang", fake_send_tensor_to_sglang
    )

    policy_utils.stream_weights_via_http_impl(
        params_generator=params_generator(),
        sglang_url_to_gpu_uuids={
            "http://sglang-a": ["uuid0"],
            "http://sglang-b": ["uuid0"],
        },
        rank=0,
        worker_name="worker",
        current_device_uuid="uuid0",
    )

    assert [call["name"] for call in send_calls] == ["w1", "w2"]
    assert all(call["handlers"] == ["handler"] for call in send_calls)
