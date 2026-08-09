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

"""Reuse fake-quantized weights across frozen-weight QAT stages.

In ModelOpt QAT the weight quantizer sits inside the linear, so every forward recomputes
``weight_quantizer(weight)`` — an elementwise pass over the weight shard, uncached.
That result only changes when the weight changes, i.e. at an optimizer step, so any
stretch of forwards between weight updates recomputes the identical tensor:

* the ``get_logprobs`` re-scoring stage (``torch.no_grad()``, no optimizer step at all)
  — served by :func:`temporarily_fold_weights`;
* the gradient-accumulation microbatches of one training global batch (the optimizer
  steps once, *after* all of them) — served by
  :func:`temporarily_cache_weight_quantization`, which must keep the quantizer in the
  autograd graph and therefore caches instead of folding.

Folding writes the fake-quantized value into the weight and disables the weight
quantizer — exactly the frozen-weight steady state. ModelOpt ships this as
:func:`modelopt.torch.quantization.fold_weight`, and this module applies the same
per-weight formula (``quant_module.py::QuantModule.fold_weight``), but folds each
discovered pair directly instead of delegating to the utility, for two reasons:

* ``fold_weight`` selects on ``fake_quant`` alone and dereferences ``weight.data``
  unconditionally, so it crashes with ``AttributeError`` on Megatron models with tied
  embeddings, where the ``output_layer`` exposes a ``weight_quantizer`` but its
  ``weight`` is ``None`` (the embedding weight is borrowed at forward time).
* Folding a *disabled* quantizer is an identity no-op (a disabled quantizer returns
  its input unchanged), so cloning those weights for restore would only waste memory —
  on standard QARL recipes the disabled ``lm_head``/embedding quantizers account for
  ~40% of the quantized-weight bytes.

The fold is reversible: original weights are cloned before folding and written back
through their existing parameter storage on exit, so it can wrap a single stage of an
otherwise-continuing QAT run.
"""

import contextlib
from collections.abc import Callable, Iterator
from typing import Any, cast

import torch
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

_QUANTIZER_SUFFIX = "weight_quantizer"


def _foldable_weight_quantizers(
    model: torch.nn.Module,
) -> Iterator[tuple[torch.Tensor, TensorQuantizer]]:
    """Yield the enabled ``(weight, quantizer)`` pairs whose fold changes the forward.

    Mirrors ``QuantModule.fold_weight``'s discovery — any attribute whose name ends in
    ``weight_quantizer`` holding a ``fake_quant`` ``TensorQuantizer``, paired with the
    weight named by dropping the ``_quantizer`` suffix. Matching the suffix scan matters
    for fused and MoE modules, which expose names like ``w13_weight_quantizer`` and
    ``gate_up_proj_weight_quantizer``: a plain ``module.weight_quantizer`` lookup would
    miss those, leaving them folded and disabled after the stage.

    Beyond the upstream scan, pairs are skipped when:

    * the quantizer is disabled — its forward is the identity, so folding it is a no-op
      that would only cost a wasted restore clone (recipes routinely disable
      ``lm_head``/embedding quantizers, which hold ~40% of the quantized-weight bytes);
    * the weight is not a tensor — Megatron tied-embedding ``output_layer`` modules
      carry ``weight = None`` and borrow the embedding weight at forward time
      (upstream ``fold_weight`` crashes on these);
    * the quantizer is a ``SequentialQuantizer`` (W4A4 double-quant) — it subclasses
      ``nn.Sequential``, not ``TensorQuantizer``, and upstream skips it too.
    """
    for module in model.modules():
        for name in dir(module):
            if not name.endswith(_QUANTIZER_SUFFIX):
                continue
            quantizer = getattr(module, name, None)
            if (
                not isinstance(quantizer, TensorQuantizer)
                or not quantizer.fake_quant
                or not quantizer.is_enabled
            ):
                continue
            weight = getattr(module, name[: -len("_quantizer")], None)
            if isinstance(weight, torch.Tensor):
                yield weight, quantizer


@contextlib.contextmanager
def temporarily_fold_weights(
    model: torch.nn.Module,
    *,
    verbose: bool = False,
    rank: int = 0,
) -> Iterator[None]:
    """Fold fake-quantized weights into the parameters for the duration of the block.

    Applies ModelOpt's fold formula (``quantizer(weight.float()).to(weight.dtype)``,
    from ``QuantModule.fold_weight``) to each enabled weight quantizer, snapping the
    quantized value into the existing parameter storage and disabling the quantizer,
    then restores the original weights and re-enables the quantizers on exit. Forwards
    inside the block read an already-snapped weight and short-circuit out of the
    disabled quantizer, so the snap happens once per stage instead of once per
    microbatch. Calibration state (``_amax`` / ``_pre_quant_scale``) is never touched.

    Only valid while the weights are frozen — ``no_grad`` with no optimizer step —
    which is exactly the ``get_logprobs`` re-scoring stage. Folding across a live
    training step would corrupt the weights, since the fold is written into the
    parameter itself.

    Activation quantization is unaffected: only *weight* quantizers are disabled, so a
    recipe's ``input_quantizer`` / ``output_quantizer`` keep running and W4A4 logprobs
    are unchanged.

    Costs one temporary copy of each folded weight shard, held only for the block.
    """
    folded = list(_foldable_weight_quantizers(model))
    original_weights = [(weight, weight.detach().clone()) for weight, _ in folded]

    try:
        with torch.no_grad():
            for weight, quantizer in folded:
                # Exact upstream fold formula (quant_module.py::fold_weight).
                weight.data.copy_(quantizer(weight.float()).to(weight.dtype))
                quantizer.disable()
        yield
    finally:
        with torch.no_grad():
            for weight, original in original_weights:
                weight.data.copy_(original)
        for _, quantizer in folded:
            quantizer.enable()
    if verbose and rank == 0:
        print(
            f"[weight_folding] frozen-weight stage: folded and restored "
            f"{len(folded)} weight quantizer(s)."
        )


def _cache_eligible(quantizer: TensorQuantizer) -> bool:
    """Whether ``quantizer``'s steady-state forward/backward can be replicated exactly.

    The cached forward replays a precomputed output and replicates the backward of
    ``TensorQuantizer._fake_quantize`` (pass-through STE, or the amax clip mask when
    ``pass_through_bwd`` is disabled). Any quantizer feature that adds other terms to
    the forward chain — smoothquant ``pre_quant_scale``, input rotation, the
    static-block reshape, bias, calibration collection — would make that replica
    wrong, so such quantizers are left on their original forward (correct, just not
    accelerated).
    """
    return (
        type(quantizer) is TensorQuantizer
        and quantizer._if_quant
        and not quantizer._if_calib
        and getattr(quantizer, "pre_quant_scale", None) is None
        and not getattr(quantizer, "rotate_is_enabled", False)
        and not getattr(quantizer, "is_static_block_quant", True)
        and getattr(quantizer, "bias_calibrator", None) is None
        and hasattr(quantizer, "_get_amax")
    )


def _make_cached_forward(
    quantizer: TensorQuantizer,
    weight: torch.Tensor,
    cached: torch.Tensor,
    amax: torch.Tensor | None,
    original_forward: Callable[[torch.Tensor], torch.Tensor],
    stats: dict[str, int],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a replacement ``forward`` that replays ``cached`` for ``weight``.

    Exactness contract, mirroring ``TensorQuantizer._fake_quantize``:

    * forward: the module's quantized-weight tensor is ``cached``, computed once via
      the quantizer's own forward — bit-identical by construction.
    * backward: upstream saves ``(inputs, amax)`` only when ``pass_through_bwd`` is
      off and amax exists, and then applies ``where(|inputs| <= amax, grad, 0)``
      (``_fake_quant_backward_function``); otherwise the gradient passes through
      unchanged. ``amax`` here is pre-resolved to ``None`` for the pass-through case.

    Any call that is not exactly "the same weight storage, quantization enabled" falls
    back to the original forward — refit paths call weight quantizers on ``.float()``
    copies, and ``disable_quantization`` can flip state mid-window.
    """
    w_ptr = weight.data_ptr()
    w_meta = (weight.shape, weight.stride(), weight.dtype, weight.device)

    class _CachedWeightFakeQuant(torch.autograd.Function):
        @staticmethod
        def forward(  # pyrefly: ignore[bad-override]  Always ignore torch.autograd.Function.forward's type since it's always more specific than the base class
            ctx: Any,
            inputs: torch.Tensor,
        ) -> torch.Tensor:
            if amax is not None:
                ctx.save_for_backward(inputs)
            return cached.view_as(cached)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: torch.Tensor) -> torch.Tensor:
            grad = grad_outputs[0]
            if not ctx.saved_tensors:
                return grad
            (inputs,) = ctx.saved_tensors
            # Exact upstream clip-mask STE (tensor_quant.py::_fake_tensor_quant_backward).
            zero = grad.new_zeros(1)
            return torch.where(inputs.abs() <= amax, grad, zero)

    def cached_forward(inputs):
        if (
            isinstance(inputs, torch.Tensor)
            and inputs.data_ptr() == w_ptr
            and (inputs.shape, inputs.stride(), inputs.dtype, inputs.device) == w_meta
            and quantizer._if_quant
            and not quantizer._if_calib
            and quantizer.is_enabled
        ):
            stats["hits"] += 1
            return _CachedWeightFakeQuant.apply(inputs)
        stats["misses"] += 1
        return original_forward(inputs)

    return cached_forward


@contextlib.contextmanager
def temporarily_cache_weight_quantization(
    model: torch.nn.Module,
    *,
    verbose: bool = False,
    rank: int = 0,
) -> Iterator[None]:
    """Serve ``weight_quantizer(weight)`` from a per-stage cache, exact in both passes.

    For each enabled fake-quant weight quantizer, computes the quantized weight once
    (via the quantizer's own forward) and patches the quantizer to replay it, with a
    backward that replicates ModelOpt's exactly: pass-through STE by default, or the
    ``where(|w| <= amax, grad, 0)`` clip mask when the quantizer sets
    ``pass_through_bwd=False``. Both directions are bit-identical to the unpatched
    quantizer, so this is safe around *training* forward-backward passes — unlike
    :func:`temporarily_fold_weights`, which disables the quantizer and thereby drops
    the clip mask from the gradient.

    Only valid while the weights are frozen: the cache must be rebuilt after every
    optimizer step, so wrap exactly one gradient-accumulation window (one
    ``megatron_forward_backward`` call), never a loop that steps the optimizer.

    Never mutates parameters or quantizer state; the patch is an instance-level
    ``forward`` override removed on exit (exception-safe). Calls with any other
    tensor, or after the quantizer is disabled mid-window, fall back to the original
    forward. Costs one cached copy of each quantized weight shard for the window.
    """
    stats = {"hits": 0, "misses": 0}
    patched: list[TensorQuantizer] = []
    try:
        with torch.no_grad():
            for weight, quantizer in _foldable_weight_quantizers(model):
                if not _cache_eligible(quantizer) or "forward" in vars(quantizer):
                    continue
                original_forward = quantizer.forward
                try:
                    cached = original_forward(weight).detach()
                    amax: torch.Tensor | None = (
                        None
                        if quantizer.is_mx_format
                        or getattr(quantizer, "_pass_through_bwd", True)
                        else cast("torch.Tensor | None", quantizer._get_amax(weight))
                    )
                except Exception:
                    continue  # unexpected quantizer flavor: leave it unpatched
                object.__setattr__(
                    quantizer,
                    "forward",
                    _make_cached_forward(
                        quantizer, weight, cached, amax, original_forward, stats
                    ),
                )
                patched.append(quantizer)
        yield
    finally:
        for quantizer in patched:
            object.__delattr__(quantizer, "forward")
    if verbose and rank == 0:
        print(
            f"[weight_folding] cached {len(patched)} weight quantizer(s) for the "
            f"frozen-weight window: {stats['hits']} cache hits, "
            f"{stats['misses']} fallback calls."
        )
