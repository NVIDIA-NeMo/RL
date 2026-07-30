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
