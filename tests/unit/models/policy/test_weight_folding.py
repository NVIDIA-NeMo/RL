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

import contextlib
import copy
from collections.abc import Iterator
from contextlib import AbstractContextManager

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.mcore

mtq = pytest.importorskip(
    "modelopt.torch.quantization",
    reason="Requires the nvidia-modelopt package",
)

from nemo_rl.modelopt.models.policy.workers.weight_folding import (  # noqa: E402
    temporarily_fold_weights,
)


def _quantized_model() -> tuple[nn.Module, torch.Tensor]:
    """Build a small INT8 fake-quantized model and the input it was calibrated on."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(16, 16, bias=False),
        nn.Linear(16, 16, bias=False),
    )
    inputs = torch.randn(4, 16)
    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, lambda m: m(inputs))
    return model, inputs


def test_folding_does_not_change_forward_output() -> None:
    """The whole point: folded forwards must produce identical logits."""
    model, inputs = _quantized_model()
    with torch.no_grad():
        expected = model(inputs)

        with temporarily_fold_weights(model):
            folded = model(inputs)

        restored = model(inputs)

    assert torch.equal(expected, folded), "folding changed the forward output"
    assert torch.equal(expected, restored), "restore changed the forward output"


def test_folds_inside_and_restores_weights_and_quantizers() -> None:
    model, _ = _quantized_model()
    linear = model[0]
    original_weight = linear.weight.detach().clone()
    original_storage = linear.weight.data_ptr()
    original_amax = linear.weight_quantizer.amax.detach().clone()

    with temporarily_fold_weights(model):
        # Inside: weight carries the snapped value and the quantizer steps aside.
        assert not torch.equal(original_weight, linear.weight)
        assert not linear.weight_quantizer.is_enabled

    assert torch.equal(original_weight, linear.weight)
    assert linear.weight.data_ptr() == original_storage, (
        "restore must write through existing parameter storage"
    )
    assert linear.weight_quantizer.is_enabled
    # keep_attrs=True must have preserved calibration state.
    assert linear.weight_quantizer.amax is not None
    assert torch.equal(original_amax, linear.weight_quantizer.amax)


def test_restores_after_exception_in_body() -> None:
    model, _ = _quantized_model()
    linear = model[0]
    original_weight = linear.weight.detach().clone()

    with pytest.raises(RuntimeError, match="boom"):
        with temporarily_fold_weights(model):
            raise RuntimeError("boom")

    assert torch.equal(original_weight, linear.weight)
    assert linear.weight_quantizer.is_enabled


def test_restores_fused_weight_quantizers() -> None:
    """Regression: MoE/fused modules expose e.g. ``w13_weight_quantizer``.

    ``fold_weight`` folds any ``*_weight_quantizer``, so restoring only a plain
    ``module.weight_quantizer`` would leave these folded and disabled for the rest of
    training.
    """
    model, _ = _quantized_model()
    linear = model[0]
    linear.register_parameter("w13_weight", nn.Parameter(torch.randn(16, 16)))
    linear.add_module("w13_weight_quantizer", copy.deepcopy(linear.weight_quantizer))

    original_fused = linear.w13_weight.detach().clone()

    with temporarily_fold_weights(model):
        assert not torch.equal(original_fused, linear.w13_weight)
        assert not linear.w13_weight_quantizer.is_enabled

    assert torch.equal(original_fused, linear.w13_weight), (
        "fused weight was folded but never restored"
    )
    assert linear.w13_weight_quantizer.is_enabled


def test_quantizer_disabled_beforehand_stays_disabled() -> None:
    model, _ = _quantized_model()
    model[0].weight_quantizer.disable()

    with temporarily_fold_weights(model):
        pass

    assert not model[0].weight_quantizer.is_enabled
    assert model[1].weight_quantizer.is_enabled


def test_nothing_to_fold_is_a_noop() -> None:
    """An unquantized model has no weight quantizers; the context must still work."""
    model = nn.Sequential(nn.Linear(8, 8))
    original = model[0].weight.detach().clone()

    with temporarily_fold_weights(model):
        pass

    assert torch.equal(original, model[0].weight)


@pytest.mark.parametrize(
    "config, expects_fold",
    [
        ({}, False),
        ({"quant_fold_frozen_weight_snap": False}, False),
        ({"quant_fold_frozen_weight_snap": True}, True),
    ],
)
def test_quant_worker_routes_logprobs_through_fold(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, bool],
    expects_fold: bool,
) -> None:
    worker_module = pytest.importorskip(
        "nemo_rl.modelopt.models.policy.workers.megatron_quant_policy_worker",
        reason="Requires Megatron and Ray",
    )

    events: list[str] = []

    def recording_context(
        *args: object, **kwargs: object
    ) -> AbstractContextManager[None]:
        @contextlib.contextmanager
        def manager() -> Iterator[None]:
            events.append("fold_enter")
            try:
                yield
            finally:
                events.append("fold_exit")

        return manager()

    def base_get_logprobs(self: object, *args: object, **kwargs: object) -> str:
        events.append("base")
        return "result"

    monkeypatch.setattr(worker_module, "temporarily_fold_weights", recording_context)
    monkeypatch.setattr(
        worker_module.MegatronPolicyWorkerImpl, "get_logprobs", base_get_logprobs
    )

    worker_class = (
        worker_module.MegatronQuantPolicyWorker.__ray_metadata__.modified_class
    )
    worker = object.__new__(worker_class)
    worker.cfg = config
    worker.model = object()
    worker.rank = 0

    assert worker.get_logprobs() == "result"
    assert events == (["fold_enter", "base", "fold_exit"] if expects_fold else ["base"])
