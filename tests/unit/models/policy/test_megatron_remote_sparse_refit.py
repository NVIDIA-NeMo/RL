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

import torch


def test_remote_sparse_stream_drains_cuda_before_return(monkeypatch):
    from nemo_rl.models.policy.workers.megatron_remote_sparse_refit import (
        MegatronRemoteSparseRefit,
    )

    class Worker:
        @staticmethod
        def _iter_params_with_optional_kv_scales():
            return iter(())

    worker = Worker()
    remote_refit = object.__new__(MegatronRemoteSparseRefit)
    remote_refit._worker = worker
    remote_refit._tracker = object()
    result = {"payloads": 1, "changed_elements": 2, "total_elements": 3}
    events = []

    def stream(*_args, **_kwargs):
        events.append("stream")
        return result

    monkeypatch.setattr(
        "nemo_rl.models.policy.workers.megatron_remote_sparse_refit."
        "stream_sparse_delta_payloads_via_zmq",
        stream,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", lambda _name: None)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", lambda: None)

    actual = remote_refit.stream(
        "zmq",
        ["tcp://receiver:5555"],
        transfer_id="transfer",
        api_key_env_var=None,
        timeout_s=1.0,
        shard_rank=0,
        shard_count=1,
    )

    assert actual is result
    assert events == ["stream", "sync"]
