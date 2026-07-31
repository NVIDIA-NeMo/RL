# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Opt-in cache for fake-quantized ("snapped") weights during frozen-weight stages.

In ModelOpt QAT the weight quantizer sits inside the linear, so every forward
recomputes ``W_snapped = weight_quantizer(weight)`` — an elementwise pass over the
weight shard, uncached. During the ``get_logprobs`` re-scoring stage the weights are
frozen (``torch.no_grad()``, no optimizer step between microbatches), so that snap is
byte-identical across all N microbatches and is pure wasted work.

This module monkeypatches ``TensorQuantizer.forward`` with a caching wrapper that, only
while :func:`weight_snap_cache` is active, stores the snapped weight on its first
computation and reuses it. Three gates keep it correct:

1. ``not torch.is_grad_enabled()`` — the cache engages only in no-grad stages. The
   training forward (grad on) always recomputes, so the STE gradient path is untouched.
2. Only *weight* quantizers are tagged (``_nrl_cache_ok``); activation/output quantizers
   are never cached (their inputs change every microbatch anyway).
3. The cache is keyed on ``(weight.data_ptr(), weight._version)``, so an optimizer step
   (which bumps ``_version`` or moves storage) can never serve a stale snap. The cached
   copies are also freed on context exit.

Covers both the dense TE path (``te_quantized_linear_fn``) and the MoE grouped path
(``te_grouped_quantized_linear_fn``); both route through ``TensorQuantizer.forward``.

:func:`materialized_weight_snap` is a stronger variant for weight-only recipes. Caching
removes the snap *arithmetic* but leaves ModelOpt's per-forward machinery in place, and
measurement on 8xH100 at 32B showed that machinery -- not the snap -- is most of what
remains of the QAT logprobs tax. Since weight-only fake-quant is elementwise,
``GEMM(x, snap(W)) == GEMM(x, W')`` for ``W' = snap(W)``, so the snap can be materialized
into the parameter once and the wrapper dropped for the whole stage. See that function
for the constraint that makes it weight-only.
"""

import contextlib
import os
import threading

import torch

from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

_state = threading.local()


def _verify_budget() -> int:
    """Read NRL_SNAP_CACHE_VERIFY at call time.

    If set to ``N``, the first ``N`` cache hits assert the cached snap is bit-identical
    to a fresh recompute (the snap is deterministic, so it must be) — a correctness
    self-check for the first landing. Leave unset in production.
    """
    return int(os.environ.get("NRL_SNAP_CACHE_VERIFY", "0") or "0")


def _enabled() -> bool:
    return getattr(_state, "enabled", False)


def _install_patch() -> None:
    """Idempotently wrap ``TensorQuantizer.forward`` with the caching shim."""
    if getattr(TensorQuantizer, "_nrl_snap_patched", False):
        return
    _orig_forward = TensorQuantizer.forward

    def _forward(self, inputs):
        if (
            _enabled()
            and not torch.is_grad_enabled()
            and getattr(self, "_nrl_cache_ok", False)
        ):
            try:
                key = (inputs.data_ptr(), inputs._version)
            except (RuntimeError, AttributeError):
                # e.g. DTensor has no data_ptr(); fall back to recompute.
                return _orig_forward(self, inputs)
            cached = getattr(self, "_nrl_snap_cache", None)
            if cached is not None and cached[0] == key:
                _state.hits += 1
                if _state.verify_left > 0:
                    _state.verify_left -= 1
                    fresh = _orig_forward(self, inputs)
                    if not torch.equal(fresh, cached[1]):
                        raise AssertionError(
                            "snap_cache: cached snapped weight differs from a fresh "
                            "recompute — the cache is unsafe here."
                        )
                return cached[1]
            out = _orig_forward(self, inputs)
            self._nrl_snap_cache = (key, out)
            _state.misses += 1
            return out
        return _orig_forward(self, inputs)

    TensorQuantizer.forward = _forward
    TensorQuantizer._nrl_snap_patched = True


def _weight_tensor_quantizers(model):
    """Yield every TensorQuantizer reachable through a module's ``weight_quantizer``.

    Handles both a bare ``TensorQuantizer`` and a ``SequentialQuantizer`` (w4a4
    double-quant) container by walking submodules — a TensorQuantizer is itself an
    ``nn.Module``, so ``.modules()`` yields it and any nested quantizers.
    """
    for module in model.modules():
        wq = getattr(module, "weight_quantizer", None)
        if isinstance(wq, torch.nn.Module):
            for sub in wq.modules():
                if isinstance(sub, TensorQuantizer):
                    yield sub


def _quantized_weight_modules(model):
    """Yield ``(module, weight_quantizer)`` for every module that fake-quantizes a weight."""
    for module in model.modules():
        wq = getattr(module, "weight_quantizer", None)
        weight = getattr(module, "weight", None)
        if isinstance(wq, torch.nn.Module) and isinstance(weight, torch.Tensor):
            yield module, wq


@contextlib.contextmanager
def materialized_weight_snap(model, verbose: bool = False, rank: int = 0):
    """Snap every weight once, then bypass ModelOpt's per-forward wrapper for the stage.

    :func:`weight_snap_cache` removes the *arithmetic* of re-snapping but leaves
    ModelOpt's machinery running on every forward: ``_QuantFunctionalMixin.forward``
    swaps Megatron's linear functional in and out (``replace_function`` ->
    ``getattr`` + 2x ``setattr`` on entry, ``setattr`` + ``delattr`` on exit, per
    functional) and calls three quantizers, ~263k times per rank per logprobs stage at
    32B. Measured on 8xH100, that machinery -- not the snap -- is the bulk of what is
    left of the QAT logprobs tax.

    Weight-only fake-quant is an elementwise transform of the weight, so::

        GEMM(x, snap(W))  ==  GEMM(x, W')     with  W' = snap(W)

    which means the wrapper is unnecessary once the snap has been materialized. This
    context manager therefore snaps each weight once into ``weight.data``, makes
    ``functionals_to_replace`` yield nothing so ``_QuantFunctionalMixin.forward``
    degenerates to the original linear, and restores both on exit.

    Only valid while the weights are frozen (``no_grad``, no optimizer step), which is
    exactly the ``get_logprobs`` re-scoring stage. Memory cost matches
    :func:`weight_snap_cache`: one extra copy of the weight shard, freed on exit.
    """
    from modelopt.torch.quantization.plugins.custom import (
        _ParallelLinear,
        _QuantFunctionalMixin,
    )

    # Bypassing the wrapper also bypasses the linear's input/output quantizers, so this
    # is only equivalent for WEIGHT-ONLY recipes (W4A16). Under W4A4 the activation
    # quantizers do real work and skipping them would silently change the numerics --
    # refuse rather than produce wrong logprobs. (Attention q/k/v bmm quantizers are
    # called directly by the attention module, not through this wrapper, so they are
    # unaffected either way.)
    for module, _ in _quantized_weight_modules(model):
        for attr in ("input_quantizer", "output_quantizer"):
            q = getattr(module, attr, None)
            if q is not None and getattr(q, "is_enabled", False):
                raise ValueError(
                    "quant_materialize_frozen_weight_snap requires a weight-only "
                    f"quantization recipe, but {attr} is enabled on a linear "
                    "(e.g. W4A4). Bypassing the ModelOpt wrapper would skip activation "
                    "quantization and change the logprobs. Use "
                    "quant_cache_frozen_weight_snap instead."
                )

    saved: list[tuple[torch.nn.Module, torch.Tensor]] = []
    skipped = 0
    with torch.no_grad():
        for module, wq in _quantized_weight_modules(model):
            snapped = wq(module.weight)
            if snapped is module.weight:
                # Quantizer disabled (e.g. output_layer) -- nothing to materialize.
                skipped += 1
                continue
            saved.append((module, module.weight.data))
            module.weight.data = snapped

    # NOTE: `_ParallelLinear` OVERRIDES `_QuantFunctionalMixin.functionals_to_replace`,
    # so patching only the mixin is a silent no-op for every linear in the model.
    empty = property(lambda self: iter(()))
    patched = [
        (klass, klass.__dict__.get("functionals_to_replace"))
        for klass in (_ParallelLinear, _QuantFunctionalMixin)
    ]
    for klass, _ in patched:
        klass.functionals_to_replace = empty
    try:
        yield
    finally:
        for klass, original in patched:
            if original is None:
                del klass.functionals_to_replace
            else:
                klass.functionals_to_replace = original
        for module, original_weight in saved:
            module.weight.data = original_weight
        if verbose and rank == 0:
            print(
                f"[snap_cache] frozen-weight stage: {len(saved)} weight snaps "
                f"materialized once, wrapper bypassed ({skipped} quantizers disabled)."
            )


@contextlib.contextmanager
def weight_snap_cache(model, verbose: bool = False, rank: int = 0):
    """Cache snapped weights across the forwards run inside this context.

    Intended to wrap a single frozen-weight stage (e.g. ``get_logprobs``). Tags the
    model's weight quantizers, enables caching for the duration, then frees the cached
    tensors on exit so the memory (a second bf16 copy of the weight shard) is only held
    during the stage — when vLLM is offloaded and the optimizer is on CPU.
    """
    _install_patch()
    for q in _weight_tensor_quantizers(model):
        q._nrl_cache_ok = True

    prev_enabled = _enabled()
    _state.enabled = True
    _state.hits = 0
    _state.misses = 0
    _state.verify_left = _verify_budget()
    try:
        yield
    finally:
        _state.enabled = prev_enabled
        hits, misses = getattr(_state, "hits", 0), getattr(_state, "misses", 0)
        for q in _weight_tensor_quantizers(model):
            q.__dict__.pop("_nrl_snap_cache", None)
        if verbose and rank == 0:
            print(
                f"[snap_cache] frozen-weight stage: {misses} weight snaps computed, "
                f"{hits} reused (=snaps avoided)."
            )
