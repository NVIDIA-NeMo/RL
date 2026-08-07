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

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("megatron.bridge")

from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker  # noqa: E402
from nemo_rl.models.policy.workers.megatron_policy_worker import (  # noqa: E402
    MegatronPolicyWorkerImpl,
    _profile_policy_step,
)

pytestmark = pytest.mark.mcore


def _wrapped_train(body: MagicMock) -> Callable[..., dict[str, Any]]:
    @_profile_policy_step
    def train(
        self: Any,
        data: Any,
        loss_fn: Any,
        eval_mode: bool = False,
        gbs: int | None = None,
        mbs: int | None = None,
        check_dim_skip_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        return body(
            data,
            loss_fn,
            eval_mode=eval_mode,
            gbs=gbs,
            mbs=mbs,
            check_dim_skip_keys=check_dim_skip_keys,
        )

    return train


def test_monolithic_train_profiles_complete_update():
    profiler = MagicMock()
    worker = SimpleNamespace(_policy_profiler=profiler)
    body = MagicMock(return_value={"loss": 1.0})

    result = _wrapped_train(body)(worker, "data", "loss", gbs=8, mbs=2)

    assert result == {"loss": 1.0}
    profiler.begin_train_step.assert_called_once_with()
    profiler.finish_train_step.assert_called_once_with()
    profiler.abort_train_step.assert_not_called()


def test_monolithic_eval_does_not_profile():
    profiler = MagicMock()
    worker = SimpleNamespace(_policy_profiler=profiler)
    body = MagicMock(return_value={})

    _wrapped_train(body)(worker, "data", "loss", eval_mode=True)

    profiler.begin_train_step.assert_not_called()
    profiler.finish_train_step.assert_not_called()
    profiler.abort_train_step.assert_not_called()


def test_monolithic_train_aborts_profiler_on_error():
    profiler = MagicMock()
    worker = SimpleNamespace(_policy_profiler=profiler)
    body = MagicMock(side_effect=ValueError("bad batch"))

    with pytest.raises(ValueError, match="bad batch"):
        _wrapped_train(body)(worker, "data", "loss")

    profiler.abort_train_step.assert_called_once_with(reason="policy_train_error")
    profiler.finish_train_step.assert_not_called()


def test_profiler_abort_error_does_not_mask_training_error():
    profiler = MagicMock()
    profiler.abort_train_step.side_effect = RuntimeError("profiler cleanup failed")
    worker = SimpleNamespace(_policy_profiler=profiler)
    body = MagicMock(side_effect=ValueError("bad batch"))

    with pytest.raises(ValueError, match="bad batch"):
        _wrapped_train(body)(worker, "data", "loss")


def test_shutdown_closes_profiler_then_cleans_up():
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker._policy_profiler = MagicMock()

    with patch.object(AbstractPolicyWorker, "shutdown", return_value=True) as cleanup:
        assert worker.shutdown() is True

    worker._policy_profiler.close.assert_called_once_with()
    cleanup.assert_called_once_with()


def test_shutdown_cleans_up_when_profiler_close_fails():
    worker = object.__new__(MegatronPolicyWorkerImpl)
    worker._policy_profiler = MagicMock()
    worker._policy_profiler.close.side_effect = RuntimeError("incomplete")

    with (
        patch.object(AbstractPolicyWorker, "shutdown", return_value=True) as cleanup,
        pytest.raises(RuntimeError, match="incomplete"),
    ):
        worker.shutdown()

    cleanup.assert_called_once_with()
