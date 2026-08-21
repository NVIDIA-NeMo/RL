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
"""Draft weights ride every supported refit transport with preserved dtypes."""

import torch
from torch import nn

import nemo_rl.models.policy.workers.dtensor_policy_worker_v2 as worker_mod
from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
    DTensorPolicyWorkerV2Impl,
)


class _TinyDraft(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.register_buffer("d2t", torch.arange(8, dtype=torch.int64))
        self.register_buffer("t2d", torch.zeros(16, dtype=torch.bool))


def _make_worker(with_draft: bool = True) -> DTensorPolicyWorkerV2Impl:
    worker = DTensorPolicyWorkerV2Impl.__new__(DTensorPolicyWorkerV2Impl)
    worker.model = nn.Linear(4, 4, bias=False)
    worker.draft_model = _TinyDraft() if with_draft else None
    worker.dtype = torch.bfloat16
    worker.cpu_offload = False
    worker.rank = 0
    worker.zmq_socket = object()
    worker.maybe_init_zmq = lambda: None
    worker.model_update_group = object()
    return worker


def _collect(params_generator):
    return {name: tensor for name, tensor in params_generator}


def _assert_draft_stream(params: dict[str, torch.Tensor]):
    assert "weight" in params  # policy weight
    assert params["weight"].dtype == torch.bfloat16
    assert "draft.proj.weight" in params
    assert params["draft.proj.weight"].dtype == torch.bfloat16
    # Integer/bool buffers must keep their dtype: token ids and vocab masks
    # are not representable in bf16.
    assert params["draft.d2t"].dtype == torch.int64
    assert params["draft.t2d"].dtype == torch.bool


def test_refit_params_generator_appends_typed_draft_keys():
    params = _collect(_make_worker()._refit_params_generator())
    _assert_draft_stream(params)


def test_refit_params_generator_without_draft_has_no_draft_keys():
    params = _collect(_make_worker(with_draft=False)._refit_params_generator())
    assert not any(name.startswith("draft.") for name in params)


def test_ipc_zmq_transport_carries_draft_stream(monkeypatch):
    captured = {}

    def fake_impl(*, params_generator, **kwargs):
        captured.update(_collect(params_generator))

    monkeypatch.setattr(
        "nemo_rl.models.policy.utils.stream_weights_via_ipc_zmq_impl", fake_impl
    )
    _make_worker().stream_weights_via_ipc_zmq()
    _assert_draft_stream(captured)


def test_collective_transport_carries_draft_stream(monkeypatch):
    captured = {}

    def fake_producer(*, iterator, **kwargs):
        captured.update(_collect(iterator))

    monkeypatch.setattr(worker_mod, "packed_broadcast_producer", fake_producer)
    _make_worker().broadcast_weights_for_collective()
    _assert_draft_stream(captured)


def test_checkpoint_engine_params_carry_draft_stream():
    params = _collect(_make_worker()._checkpoint_engine_params())
    _assert_draft_stream(params)


def test_prepare_refit_info_advertises_typed_draft_entries():
    manifest = _make_worker().prepare_refit_info()
    assert manifest["weight"] == (torch.Size([4, 4]), torch.bfloat16)
    assert manifest["draft.proj.weight"] == (torch.Size([4, 4]), torch.bfloat16)
    assert manifest["draft.d2t"] == (torch.Size([8]), torch.int64)
    assert manifest["draft.t2d"] == (torch.Size([16]), torch.bool)
