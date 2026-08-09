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
    temporarily_cache_weight_quantization,
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


def test_quantizer_disabled_beforehand_is_skipped_and_stays_disabled() -> None:
    """Disabled quantizers are identity at forward time, so folding them is a no-op.

    They must be skipped entirely: no wasted restore clone, weight untouched, and the
    disabled state preserved through the block.
    """
    model, _ = _quantized_model()
    model[0].weight_quantizer.disable()
    disabled_weight = model[0].weight.detach().clone()

    with temporarily_fold_weights(model):
        # Skipped: the disabled quantizer's weight is never folded.
        assert torch.equal(disabled_weight, model[0].weight)
        assert not model[0].weight_quantizer.is_enabled
        # The enabled sibling still folds.
        assert not model[1].weight_quantizer.is_enabled

    assert not model[0].weight_quantizer.is_enabled
    assert model[1].weight_quantizer.is_enabled


def test_none_weight_with_quantizer_is_skipped() -> None:
    """Regression: Megatron tied-embedding ``output_layer`` has ``weight = None``.

    ModelOpt attaches a ``weight_quantizer`` to it anyway, and upstream
    ``mtq.fold_weight`` crashes with ``AttributeError: 'NoneType' object has no
    attribute 'data'`` on such modules (observed on Qwen3-0.6B, which ties word
    embeddings). The fold must skip the pair and still fold everything else.
    """
    model, inputs = _quantized_model()

    class TiedOutputLayer(nn.Module):
        def __init__(self, template: nn.Module) -> None:
            super().__init__()
            self.weight = None  # borrowed from the embedding at forward time
            self.add_module(
                "weight_quantizer", copy.deepcopy(template.weight_quantizer)
            )

    model.tied_head = TiedOutputLayer(model[0])
    assert model.tied_head.weight_quantizer.is_enabled

    with torch.no_grad():
        expected = model[0](inputs)

    with temporarily_fold_weights(model):  # must not raise
        assert not model[0].weight_quantizer.is_enabled
        with torch.no_grad():
            folded = model[0](inputs)

    assert torch.equal(expected, folded)
    assert model.tied_head.weight_quantizer.is_enabled


def test_nothing_to_fold_is_a_noop() -> None:
    """An unquantized model has no weight quantizers; the context must still work."""
    model = nn.Sequential(nn.Linear(8, 8))
    original = model[0].weight.detach().clone()

    with temporarily_fold_weights(model):
        pass

    assert torch.equal(original, model[0].weight)


# ---------------------------------------------------------------------------
# temporarily_cache_weight_quantization (training-stage cache)
# ---------------------------------------------------------------------------


def _grads(model: nn.Module, inputs: torch.Tensor) -> list[torch.Tensor]:
    """One fwd/bwd pass; returns detached copies of all parameter gradients."""
    model.zero_grad()
    model(inputs).square().sum().backward()
    return [p.grad.detach().clone() for p in model.parameters()]


def test_cache_forward_is_bit_identical_and_actually_served_from_cache() -> None:
    model, inputs = _quantized_model()
    with torch.no_grad():
        expected = model(inputs)

    original_amax = model[0].weight_quantizer.amax.detach().clone()
    with temporarily_cache_weight_quantization(model):
        assert "forward" in vars(model[0].weight_quantizer)
        with torch.no_grad():
            cached_out = model(inputs)
            # Corrupt amax in-place: the *real* quantizer would now produce a
            # different value, so an unchanged output proves the cache is served.
            model[0].weight_quantizer.amax.copy_(original_amax * 100)
            still_cached = model(inputs)
            model[0].weight_quantizer.amax.copy_(original_amax)

    assert "forward" not in vars(model[0].weight_quantizer)
    with torch.no_grad():
        restored = model(inputs)

    assert torch.equal(expected, cached_out), "caching changed the forward output"
    assert torch.equal(expected, still_cached), "forward bypassed the cache"
    assert torch.equal(expected, restored), "exit did not restore the real quantizer"


def test_cache_gradients_bit_identical_pass_through() -> None:
    """Default ModelOpt configs use pass-through STE; grads must match bitwise."""
    model, inputs = _quantized_model()
    expected = _grads(model, inputs)

    with temporarily_cache_weight_quantization(model):
        got = _grads(model, inputs)
    after = _grads(model, inputs)

    for e, g, a in zip(expected, got, after):
        assert torch.equal(e, g), "cached backward diverged from the real quantizer"
        assert torch.equal(e, a), "gradients changed after cache exit"


def test_cache_gradients_bit_identical_with_active_clip_mask() -> None:
    """With ``pass_through_bwd=False`` ModelOpt clips grads at amax; replicate it."""
    model, inputs = _quantized_model()
    for layer in model:
        wq = layer.weight_quantizer
        wq._pass_through_bwd = False
        with torch.no_grad():
            wq.amax.copy_(wq.amax * 0.5)  # force a non-trivial clip mask

    clipped = (model[0].weight.abs() > model[0].weight_quantizer.amax).sum()
    assert clipped > 0, "test setup failed to activate the clip mask"

    expected = _grads(model, inputs)
    assert (expected[0] == 0).sum() >= clipped, "upstream clip mask not in effect"

    with temporarily_cache_weight_quantization(model):
        got = _grads(model, inputs)

    for e, g in zip(expected, got):
        assert torch.equal(e, g), "cached clip-mask backward diverged"


def test_cache_falls_back_for_non_weight_tensors() -> None:
    """Refit paths call weight quantizers on ``.float()`` copies; those must not
    be served the cached (bf16-shaped) value."""
    model, inputs = _quantized_model()
    wq = model[0].weight_quantizer
    float_weight = model[0].weight.detach().float() * 0.25
    with torch.no_grad():
        expected = wq(float_weight)

    with temporarily_cache_weight_quantization(model):
        with torch.no_grad():
            got = wq(float_weight)

    assert torch.equal(expected, got), "cache served a stale value for a foreign tensor"


def test_cache_restores_forward_on_exception() -> None:
    model, inputs = _quantized_model()

    with pytest.raises(RuntimeError, match="boom"):
        with temporarily_cache_weight_quantization(model):
            raise RuntimeError("boom")

    assert "forward" not in vars(model[0].weight_quantizer)
    assert "forward" not in vars(model[1].weight_quantizer)


def test_cache_rebuilds_from_fresh_weights_per_window() -> None:
    """Simulates the per-global-batch usage: weight update between windows."""
    model, inputs = _quantized_model()

    with temporarily_cache_weight_quantization(model):
        pass

    with torch.no_grad():  # "optimizer step"
        model[0].weight.mul_(1.5)
        expected = model(inputs)

    with temporarily_cache_weight_quantization(model):
        with torch.no_grad():
            got = model(inputs)

    assert torch.equal(expected, got), "cache went stale across a weight update"


def test_cache_skips_ineligible_and_disabled_quantizers() -> None:
    model, inputs = _quantized_model()
    # pre_quant_scale adds a term to the forward chain the replica cannot
    # reproduce; such quantizers must keep their original forward. (Scale of
    # ones keeps the reference output unchanged.)
    model[0].weight_quantizer._enable_pre_quant_scale = True
    model[0].weight_quantizer.pre_quant_scale = torch.ones_like(model[0].weight[0])
    model[1].weight_quantizer.disable()
    with torch.no_grad():
        expected = model(inputs)

    with temporarily_cache_weight_quantization(model):
        assert "forward" not in vars(model[0].weight_quantizer)
        assert "forward" not in vars(model[1].weight_quantizer)
        with torch.no_grad():
            got = model(inputs)

    assert torch.equal(expected, got)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVFP4 needs CUDA")
@pytest.mark.parametrize("pass_through_bwd", [True, False])
def test_cache_gradients_bit_identical_nvfp4(pass_through_bwd: bool) -> None:
    """The shipped recipes use NVFP4 dynamic block quantization on GPU."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(32, 32, bias=False),
        nn.Linear(32, 32, bias=False),
    ).cuda()
    inputs = torch.randn(4, 32, device="cuda")
    mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, lambda m: m(inputs))
    for layer in model:
        layer.weight_quantizer._pass_through_bwd = pass_through_bwd

    with torch.no_grad():
        expected_out = model(inputs)
    expected = _grads(model, inputs)

    with temporarily_cache_weight_quantization(model):
        assert "forward" in vars(model[0].weight_quantizer)
        with torch.no_grad():
            got_out = model(inputs)
        got = _grads(model, inputs)

    assert torch.equal(expected_out, got_out)
    for e, g in zip(expected, got):
        assert torch.equal(e, g), "NVFP4 cached backward diverged"


@pytest.mark.parametrize(
    "config, expects_cache",
    [
        ({}, False),
        ({"quant_cache_train_weight_snap": False}, False),
        ({"quant_cache_train_weight_snap": True}, True),
    ],
)
def test_quant_worker_routes_train_through_cache(
    monkeypatch: pytest.MonkeyPatch,
    config: dict[str, bool],
    expects_cache: bool,
) -> None:
    worker_module = pytest.importorskip(
        "nemo_rl.modelopt.models.policy.workers.megatron_quant_policy_worker",
        reason="Requires Megatron and Ray",
    )
    base_module = worker_module.megatron_policy_worker

    events: list[str] = []

    def recording_context(
        *args: object, **kwargs: object
    ) -> AbstractContextManager[None]:
        @contextlib.contextmanager
        def manager() -> Iterator[None]:
            events.append("cache_enter")
            try:
                yield
            finally:
                events.append("cache_exit")

        return manager()

    def fake_forward_backward(*args: object, **kwargs: object) -> str:
        events.append("fwd_bwd")
        return "losses"

    def base_train(self: object, *args: object, **kwargs: object) -> str:
        # The real train() resolves megatron_forward_backward from the base
        # module's globals once per global batch.
        assert base_module.megatron_forward_backward(...) == "losses"
        assert base_module.megatron_forward_backward(...) == "losses"
        events.append("train_done")
        return "result"

    monkeypatch.setattr(
        worker_module, "temporarily_cache_weight_quantization", recording_context
    )
    monkeypatch.setattr(base_module, "megatron_forward_backward", fake_forward_backward)
    monkeypatch.setattr(worker_module.MegatronPolicyWorkerImpl, "train", base_train)

    worker_class = (
        worker_module.MegatronQuantPolicyWorker.__ray_metadata__.modified_class
    )
    worker = object.__new__(worker_class)
    worker.cfg = config
    worker.model = object()
    worker.rank = 0

    assert worker.train() == "result"
    if expects_cache:
        # One cache window per forward-backward call, none spanning both.
        assert events == [
            "cache_enter",
            "fwd_bwd",
            "cache_exit",
            "cache_enter",
            "fwd_bwd",
            "cache_exit",
            "train_done",
        ]
    else:
        assert events == ["fwd_bwd", "fwd_bwd", "train_done"]
    # The scoped patch must be unwound after train() returns.
    assert base_module.megatron_forward_backward is fake_forward_backward


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
