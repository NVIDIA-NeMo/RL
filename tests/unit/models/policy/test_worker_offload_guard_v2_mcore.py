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
"""Lightweight unit tests for the v2 + Megatron worker offload guards.

These mirror the dtensor v1 guard tests in
``test_dtensor_worker_offload_guard.py`` and lock in the same #1141
contract on the v2 and Megatron paths. They build a bare worker
instance via ``__new__`` and patch the flag, so they run at the L0
unit-test tier without Ray, distributed, or a real model.

The Megatron worker test is marked ``@pytest.mark.mcore`` because
``MegatronPolicyWorkerImpl`` itself imports megatron-bridge at module
load time.
"""

from __future__ import annotations

import pytest


def _make_v2_worker(*, weights_offloaded: bool):
    from nemo_rl.models.policy.workers.dtensor_policy_worker_v2 import (
        DTensorPolicyWorkerV2Impl,
    )

    worker = DTensorPolicyWorkerV2Impl.__new__(DTensorPolicyWorkerV2Impl)
    worker._weights_offloaded = weights_offloaded
    return worker


def test_v2_passes_when_on_gpu() -> None:
    worker = _make_v2_worker(weights_offloaded=False)
    worker._assert_weights_on_device("train")
    worker._assert_weights_on_device("get_logprobs")
    worker._assert_weights_on_device("score")
    worker._assert_weights_on_device("get_topk_logits")


def test_v2_train_message_points_at_prepare_for_training() -> None:
    worker = _make_v2_worker(weights_offloaded=True)
    with pytest.raises(RuntimeError) as excinfo:
        worker._assert_weights_on_device("train")
    msg = str(excinfo.value)
    assert "train()" in msg
    assert "prepare_for_training()" in msg
    assert "offload_after_refit()" in msg
    assert "#1141" in msg


@pytest.mark.parametrize("method_name", ["get_logprobs", "score", "get_topk_logits"])
def test_v2_inference_message_points_at_prepare_for_lp_inference(
    method_name: str,
) -> None:
    worker = _make_v2_worker(weights_offloaded=True)
    with pytest.raises(RuntimeError) as excinfo:
        worker._assert_weights_on_device(method_name)
    msg = str(excinfo.value)
    assert f"{method_name}()" in msg
    assert "prepare_for_lp_inference()" in msg


@pytest.mark.mcore
class TestMegatronOffloadGuard:
    """Megatron worker imports megatron-bridge at module load, so these
    tests are marked ``mcore`` and only run under ``--mcore-only``.

    The guard logic is identical to v1/v2 — the implementation in
    ``megatron_policy_worker.py`` uses ``getattr(self, '_weights_offloaded',
    False)`` so the flag does not need to be set before the first prepare
    call.
    """

    def _make_worker(self, *, weights_offloaded: bool | None):
        from nemo_rl.models.policy.workers.megatron_policy_worker import (
            MegatronPolicyWorkerImpl,
        )

        worker = MegatronPolicyWorkerImpl.__new__(MegatronPolicyWorkerImpl)
        if weights_offloaded is not None:
            worker._weights_offloaded = weights_offloaded
        return worker

    def test_passes_when_on_gpu(self) -> None:
        worker = self._make_worker(weights_offloaded=False)
        for method in ("train", "get_logprobs", "get_topk_logits", "generate"):
            worker._assert_weights_on_device(method)

    def test_passes_when_flag_unset(self) -> None:
        """Before the first offload, the flag may not exist on the
        instance yet. The guard must treat that as 'on GPU'."""
        worker = self._make_worker(weights_offloaded=None)
        for method in ("train", "get_logprobs", "get_topk_logits", "generate"):
            worker._assert_weights_on_device(method)

    def test_train_message(self) -> None:
        worker = self._make_worker(weights_offloaded=True)
        with pytest.raises(RuntimeError) as excinfo:
            worker._assert_weights_on_device("train")
        msg = str(excinfo.value)
        assert "train()" in msg
        assert "prepare_for_training()" in msg
        assert "offload_after_refit()" in msg
        assert "#1141" in msg

    @pytest.mark.parametrize(
        "method_name", ["get_logprobs", "get_topk_logits", "generate"]
    )
    def test_inference_message(self, method_name: str) -> None:
        worker = self._make_worker(weights_offloaded=True)
        with pytest.raises(RuntimeError) as excinfo:
            worker._assert_weights_on_device(method_name)
        msg = str(excinfo.value)
        assert f"{method_name}()" in msg
        assert "prepare_for_lp_inference()" in msg
