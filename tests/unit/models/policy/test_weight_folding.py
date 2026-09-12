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

"""Coverage for the frozen-weight logprob fold.

The fold itself lives in ModelOpt (``mtq.temporarily_fold_weights``). The tests here
are deliberately split in two:

* a few contract tests pinning the upstream behaviour the QAT correctness argument
  depends on -- identical forward output, restore through the original parameter
  storage, calibration state kept, restore on exception. These are what would catch a
  ModelOpt pin bump that regressed the semantics NeMo RL relies on;
* an integration test that the quant worker routes ``get_logprobs`` through the context
  manager only when ``policy.quant_fold_frozen_weight_snap`` is set.

Everything else about the fold (fused/MoE weights, tied-embedding modules carrying
``weight = None``, disabled quantizers) is upstream's own to test.
"""

import contextlib
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

temporarily_fold_weights = getattr(mtq, "temporarily_fold_weights", None)

requires_fold_api = pytest.mark.skipif(
    temporarily_fold_weights is None,
    reason="Requires a nvidia-modelopt build with mtq.temporarily_fold_weights",
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


@requires_fold_api
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


@requires_fold_api
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
    # Calibration state must survive the fold.
    assert linear.weight_quantizer.amax is not None
    assert torch.equal(original_amax, linear.weight_quantizer.amax)


@requires_fold_api
def test_restores_after_exception_in_body() -> None:
    model, _ = _quantized_model()
    linear = model[0]
    original_weight = linear.weight.detach().clone()

    with pytest.raises(RuntimeError, match="boom"):
        with temporarily_fold_weights(model):
            raise RuntimeError("boom")

    assert torch.equal(original_weight, linear.weight)
    assert linear.weight_quantizer.is_enabled


@requires_fold_api
def test_snapshot_device_cpu_keeps_snapshots_off_the_parameter_device() -> None:
    """``snapshot_device`` is threaded straight through to ModelOpt."""
    model, inputs = _quantized_model()
    linear = model[0]
    original_weight = linear.weight.detach().clone()

    with torch.no_grad():
        expected = model(inputs)
        with temporarily_fold_weights(model, snapshot_device="cpu"):
            folded = model(inputs)

    assert torch.equal(expected, folded)
    assert torch.equal(original_weight, linear.weight)
    assert linear.weight_quantizer.is_enabled


@pytest.mark.parametrize(
    "config, expects_fold",
    [
        ({}, False),
        ({"quant_fold_frozen_weight_snap": False}, False),
        ({"quant_fold_frozen_weight_snap": True}, True),
        (
            {
                "quant_fold_frozen_weight_snap": True,
                "quant_fold_snapshot_device": "cpu",
            },
            True,
        ),
    ],
)
def test_quant_worker_routes_logprobs_through_fold(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, object],
    expects_fold: bool,
) -> None:
    worker_module = pytest.importorskip(
        "nemo_rl.modelopt.models.policy.workers.megatron_quant_policy_worker",
        reason="Requires Megatron and Ray",
    )

    events: list[str] = []
    recorded: list[dict[str, object]] = []

    def recording_context(
        *args: object, **kwargs: object
    ) -> AbstractContextManager[None]:
        recorded.append(kwargs)

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
    if expects_fold:
        # An absent key must reach ModelOpt as None, its own default -- not as a
        # value invented at the call site.
        assert recorded == [
            {"snapshot_device": config.get("quant_fold_snapshot_device")}
        ]
