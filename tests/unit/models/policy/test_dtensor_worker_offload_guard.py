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
"""Lightweight unit tests for the DTensor worker's offload guard.

These tests focus on the behavior of ``_assert_weights_on_device`` and do
not spin up a real Ray worker, model, or GPU — the goal is to lock in the
contract from issue #1141 (custom training loops that skip
``prepare_for_*`` get a clear ``RuntimeError`` instead of a CUDA
``illegal memory access``) at the L0 unit-test tier.
"""

from __future__ import annotations

import pytest

from nemo_rl.models.policy.workers.dtensor_policy_worker import (
    DTensorPolicyWorkerImpl,
)


def _make_worker_under_test(*, weights_offloaded: bool) -> DTensorPolicyWorkerImpl:
    """Build a bare ``DTensorPolicyWorkerImpl`` instance without running
    ``__init__``.

    The full ``__init__`` requires Ray, distributed, and a model. We only
    need to exercise the ``_assert_weights_on_device`` branch, which
    depends solely on ``self._weights_offloaded``.
    """
    worker = DTensorPolicyWorkerImpl.__new__(DTensorPolicyWorkerImpl)
    worker._weights_offloaded = weights_offloaded
    return worker


def test_assert_weights_on_device_passes_when_weights_on_gpu() -> None:
    """When weights are on GPU, the guard is a no-op for any caller."""
    worker = _make_worker_under_test(weights_offloaded=False)

    # All three known callers must pass without raising.
    worker._assert_weights_on_device("train")
    worker._assert_weights_on_device("get_logprobs")
    worker._assert_weights_on_device("score")


def test_assert_weights_on_device_train_raises_with_helpful_message() -> None:
    """train() should point users at prepare_for_training()."""
    worker = _make_worker_under_test(weights_offloaded=True)

    with pytest.raises(RuntimeError) as excinfo:
        worker._assert_weights_on_device("train")

    msg = str(excinfo.value)
    # The message must name the failing method, the prepare step that
    # was skipped, and reference the underlying CUDA failure mode so
    # users searching their logs land here.
    assert "train()" in msg
    assert "prepare_for_training()" in msg
    assert "offload_after_refit()" in msg
    assert "illegal memory access" in msg
    assert "#1141" in msg


@pytest.mark.parametrize("method_name", ["get_logprobs", "score"])
def test_assert_weights_on_device_inference_raises_with_helpful_message(
    method_name: str,
) -> None:
    """Inference-side methods should point at prepare_for_lp_inference()."""
    worker = _make_worker_under_test(weights_offloaded=True)

    with pytest.raises(RuntimeError) as excinfo:
        worker._assert_weights_on_device(method_name)

    msg = str(excinfo.value)
    assert f"{method_name}()" in msg
    assert "prepare_for_lp_inference()" in msg
    assert "offload_after_refit()" in msg
