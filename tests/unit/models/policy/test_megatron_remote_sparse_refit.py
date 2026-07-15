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

from nemo_rl.models.generation.vllm.config import VllmDeltaCompressionConfig
from nemo_rl.models.policy.workers.megatron_remote_sparse_refit import (
    MegatronRemoteSparseRefit,
)

_DELTA_CONFIG = VllmDeltaCompressionConfig(
    encoding="overwrite", sparse_bucket_size_bytes=1024
)


def _worker(weights=()):
    def export():
        return iter(weights)

    return SimpleNamespace(_iter_params_with_optional_kv_scales=export)


def test_remote_sparse_initializes_canonical_hf_baseline(monkeypatch) -> None:
    monkeypatch.setenv("NRL_REFIT_BASELINE_IN_MEMORY", "1")
    weights = [
        ("embedding.weight", torch.ones(2, 3)),
        ("linear.weight", torch.ones(4, 3)),
    ]
    remote_refit = MegatronRemoteSparseRefit(_worker(weights), _DELTA_CONFIG)

    info = remote_refit.initialize_baseline(
        shard_rank=0, shard_count=1, transport="zmq"
    )

    assert info == {
        name: (tuple(tensor.shape), tensor.dtype) for name, tensor in weights
    }
    assert set(vars(remote_refit)) == {"_worker", "_tracker"}


def test_remote_sparse_preserves_xor_config() -> None:
    remote_refit = MegatronRemoteSparseRefit(
        _worker(), _DELTA_CONFIG.model_copy(update={"encoding": "xor"})
    )

    assert remote_refit._tracker.encoding == "xor"


def test_remote_sparse_streams_one_canonical_path_and_drains_cuda(monkeypatch) -> None:
    weights = [("model.weight", torch.ones(2))]
    remote_refit = MegatronRemoteSparseRefit(_worker(weights), _DELTA_CONFIG)
    expected = {"payloads": 1, "changed_elements": 2, "total_elements": 2}
    events = []

    def stream(iterator, **kwargs):
        assert list(iterator) == weights
        assert kwargs == {
            "delta_tracker": remote_refit._tracker,
            "transfer_id": "transfer",
            "refit_targets": ["tcp://receiver:5555"],
            "api_key_env_var": None,
            "timeout_s": 1.0,
            "shard_rank": 0,
            "shard_count": 1,
        }
        events.append("stream")
        return expected

    monkeypatch.setattr(
        "nemo_rl.models.policy.workers.megatron_remote_sparse_refit."
        "stream_sparse_delta_payloads_via_zmq",
        stream,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: events.append("sync"))

    result = remote_refit.stream(
        "zmq",
        ["tcp://receiver:5555"],
        transfer_id="transfer",
        api_key_env_var=None,
        timeout_s=1.0,
        shard_rank=0,
        shard_count=1,
        overwrite_names=["model.weight"],
    )

    assert result is expected
    assert remote_refit._tracker.overwrite_names == frozenset({"model.weight"})
    assert events == ["stream", "sync"]


def test_remote_sparse_finishes_single_tracker(monkeypatch) -> None:
    remote_refit = MegatronRemoteSparseRefit(_worker(), _DELTA_CONFIG)
    events = []
    monkeypatch.setattr(
        remote_refit._tracker,
        "on_sync_succeeded",
        lambda: events.append("succeeded"),
    )
    monkeypatch.setattr(
        remote_refit._tracker,
        "on_sync_failed",
        lambda: events.append("failed"),
    )

    remote_refit.finish(True)
    remote_refit.finish(False)

    assert events == ["succeeded", "failed"]
