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
2. Only *weight* quantizers are enrolled in the active context; activation/output
   quantizers are never cached (their inputs change every microbatch anyway).
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
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

_state = threading.local()
_patch_lock = threading.Lock()
_patch_installed = False

_SnapKey = tuple[int, int]
_CachedSnap = tuple[_SnapKey, torch.Tensor]


@dataclass
class _SnapCacheContext:
    """State scoped to one active frozen-weight cache context."""

    eligible_quantizers: set[TensorQuantizer]
    cached_snaps: dict[TensorQuantizer, _CachedSnap]
    hits: int
    misses: int
    verify_left: int


def _verify_budget() -> int:
    """Read NRL_SNAP_CACHE_VERIFY at call time.

    If set to ``N``, the first ``N`` cache hits assert the cached snap is bit-identical
    to a fresh recompute (the snap is deterministic, so it must be) — a correctness
    self-check for the first landing. Leave unset in production.
    """
    return int(os.environ.get("NRL_SNAP_CACHE_VERIFY", "0") or "0")


def _active_context() -> _SnapCacheContext | None:
    """Return the cache context active on this thread, if any."""
    return getattr(_state, "context", None)


def _install_patch() -> None:
    """Idempotently wrap ``TensorQuantizer.forward`` with the caching shim."""
    global _patch_installed

    with _patch_lock:
        if _patch_installed:
            return
        original_forward = TensorQuantizer.forward

        def _forward(self: TensorQuantizer, inputs: torch.Tensor) -> torch.Tensor:
            context = _active_context()
            if (
                context is None
                or torch.is_grad_enabled()
                or self not in context.eligible_quantizers
            ):
                return original_forward(self, inputs)

            try:
                key = (inputs.data_ptr(), inputs._version)
            except (RuntimeError, AttributeError):
                # e.g. DTensor has no data_ptr(); fall back to recompute.
                return original_forward(self, inputs)
            cached = context.cached_snaps.get(self)
            if cached is not None and cached[0] == key:
                context.hits += 1
                if context.verify_left > 0:
                    context.verify_left -= 1
                    fresh = original_forward(self, inputs)
                    if not torch.equal(fresh, cached[1]):
                        raise AssertionError(
                            "snap_cache: cached snapped weight differs from a fresh "
                            "recompute — the cache is unsafe here."
                        )
                return cached[1]
            out = original_forward(self, inputs)
            context.cached_snaps[self] = (key, out)
            context.misses += 1
            return out

        TensorQuantizer.forward = _forward
        _patch_installed = True


def _weight_tensor_quantizers(
    model: torch.nn.Module,
) -> Iterator[TensorQuantizer]:
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


def _quantized_weight_modules(
    model: torch.nn.Module,
) -> Iterator[tuple[torch.nn.Module, torch.nn.Module, torch.Tensor]]:
    """Yield each module, weight quantizer, and weight participating in fake quantization."""
    for module in model.modules():
        wq = getattr(module, "weight_quantizer", None)
        weight = getattr(module, "weight", None)
        if isinstance(wq, torch.nn.Module) and isinstance(weight, torch.Tensor):
            yield module, wq, weight


def _restore_class_attribute(
    klass: type,
    name: str,
    previous: Any,
    had_own: bool,
) -> None:
    """Restore one class attribute to its exact pre-context state."""
    if had_own:
        setattr(klass, name, previous)
    else:
        delattr(klass, name)


def _shadow_forward_after(
    klass: type,
    cutoff: type,
    cleanup: contextlib.ExitStack,
) -> bool:
    """Point ``klass.forward`` at the first ``forward`` defined *after* ``cutoff`` in the MRO.

    Used to remove a ModelOpt forward wrapper for the duration of a stage without touching
    any instance. Registers restoration before mutating the class and returns whether a
    wrapper was removed.
    """
    mro = klass.__mro__
    if cutoff not in mro:
        return False
    target: Callable[..., Any] | None = None
    for base in mro[mro.index(cutoff) + 1 :]:
        if "forward" in base.__dict__:
            target = base.__dict__["forward"]
            break
    if target is None:
        return False
    had_own = "forward" in klass.__dict__
    previous = klass.__dict__.get("forward")
    cleanup.callback(
        _restore_class_attribute,
        klass,
        "forward",
        previous,
        had_own,
    )
    klass.forward = target
    return True


@contextlib.contextmanager
def plain_module_attr_lookup(
    model: torch.nn.Module,
    *,
    verbose: bool = False,
    rank: int = 0,
) -> Iterator[None]:
    """Restore plain ``nn.Module`` attribute lookup on ModelOpt modules for the stage.

    On ``nn.Module`` parameters, buffers and submodules live in ``_parameters`` /
    ``_buffers`` / ``_modules``, not in ``__dict__`` — so ``self.weight`` always misses
    normal lookup and lands in ``__getattr__``. ModelOpt's modules are ``DynamicModule``s,
    whose override does, per access: fetch the attribute manager, check ``hp_keys()``,
    check ``da_keys()``, then enter a ``_dict_with_special()`` context manager which
    fetches the manager and rebuilds *both* key sets again — before finally delegating to
    ``nn.Module.__getattr__``.

    Megatron's linear forward reads ``self.weight`` and ``self.bias`` on every call, so at
    257 linears x 1024 microbatches this runs ~526k times per rank per logprobs stage.
    None of it is removed by :func:`weight_snap_cache` or :func:`materialized_weight_snap`
    — it is inherent to the module being a ``DynamicModule``.

    When a module has registered no hparams and no dynamic attributes, that whole path
    provably reduces to ``nn.Module.__getattr__``: ``_dict_with_special`` only does work
    when a special key is itself an hparam or dynamic attribute. This asserts exactly that
    and otherwise **refuses** — ``QuantLinearConvBase`` registers ``weight`` with a
    ``_get_quantized_weight`` callback, and bypassing there would silently drop weight
    quantization. (Megatron's ``_ParallelLinear`` extends ``QuantModule``, not
    ``QuantLinearConvBase``, so it registers neither.)

    Measured at 32B/TP8 on 8xH100: 2.0 s (step 1) and 7.3 s (step 2) off the logprobs
    stage. Safe for any frozen-weight stage; it changes lookup cost, never values.
    """
    from modelopt.torch.opt.dynamic import DynamicModule

    classes: set[type] = set()
    for module in model.modules():
        if not isinstance(module, DynamicModule):
            continue
        manager = module._get_dm_attribute_manager(use_default=True)
        dynamic_attrs, hparams = list(manager.da_keys()), list(manager.hp_keys())
        if dynamic_attrs or hparams:
            raise ValueError(
                f"plain_module_attr_lookup refused: {type(module).__name__} has dynamic "
                f"attributes {dynamic_attrs} / hparams {hparams}, so "
                "DynamicModule.__getattr__ is load-bearing there and cannot be bypassed."
            )
        classes.add(type(module))

    with contextlib.ExitStack() as cleanup:
        for klass in classes:
            had_own = "__getattr__" in klass.__dict__
            previous = klass.__dict__.get("__getattr__")
            cleanup.callback(
                _restore_class_attribute,
                klass,
                "__getattr__",
                previous,
                had_own,
            )
            klass.__getattr__ = torch.nn.Module.__getattr__
        yield
    if verbose and rank == 0:
        print(
            f"[snap_cache] plain attribute lookup restored on {len(classes)} "
            f"DynamicModule class(es) for the frozen-weight stage."
        )


def _restore_weight(weight: torch.Tensor, original_weight: torch.Tensor) -> None:
    """Restore a temporarily materialized parameter without replacing its storage."""
    with torch.no_grad():
        weight.data.copy_(original_weight)


@contextlib.contextmanager
def materialized_weight_snap(
    model: torch.nn.Module,
    *,
    verbose: bool = False,
    rank: int = 0,
) -> Iterator[None]:
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
    context manager therefore snaps each weight once *into the existing parameter
    storage*, removes ModelOpt's forward wrappers by shadowing them with the underlying
    original ``forward``, and restores both on exit.

    Only valid while the weights are frozen (``no_grad``, no optimizer step), which is
    exactly the ``get_logprobs`` re-scoring stage. Memory cost matches
    :func:`weight_snap_cache`: one extra copy of the weight shard, freed on exit.
    """
    from modelopt.torch.quantization.plugins.custom import _QuantFunctionalMixin

    # Removing a linear's wrapper also removes the input/output quantizers it invokes,
    # and materializing the snap while that wrapper still ran would double-quantize. So
    # for linears there is no safe partial mode: under W4A4 this optimization simply does
    # not apply, and refusing is better than silently changing the logprobs.
    for module, _, _ in _quantized_weight_modules(model):
        for attr in ("input_quantizer", "output_quantizer"):
            q = getattr(module, attr, None)
            if q is not None and getattr(q, "is_enabled", False):
                raise ValueError(
                    "quant_materialize_frozen_weight_snap requires weight-only "
                    f"quantization on linears, but {attr} is enabled (e.g. W4A4). "
                    "Removing the ModelOpt linear wrapper would skip activation "
                    "quantization and change the logprobs. Use "
                    "quant_cache_frozen_weight_snap instead."
                )

    # Materialize IN PLACE. Assigning `weight.data = snapped` would repoint every
    # parameter at a freshly allocated tensor instead of Megatron's original (bucketed)
    # parameter storage; copying into the existing storage keeps the GEMM reading exactly
    # the memory the unquantized model reads. Setup measured 0.33s for 7.27GiB at 32B.
    saved = 0
    skipped = 0
    wrappers_removed = 0
    skipped_attention: list[str] = []
    with contextlib.ExitStack() as cleanup:
        with torch.no_grad():
            for _, wq, weight in _quantized_weight_modules(model):
                snapped = wq(weight)
                if snapped is weight:
                    # Quantizer disabled (e.g. output_layer) -- nothing to materialize.
                    skipped += 1
                    continue
                original_weight = weight.detach().clone()
                cleanup.callback(_restore_weight, weight, original_weight)
                weight.data.copy_(snapped)
                saved += 1
                del snapped

        # Remove the ModelOpt forward wrappers outright rather than just emptying
        # `functionals_to_replace`: that left `_QuantFunctionalMixin.forward` still
        # running a generator and an ExitStack per linear, and left the attention wrapper
        # calling three (disabled) quantizers per layer. Shadowing `forward` with the
        # underlying original makes the converted model execute the unquantized code path.
        for klass in {
            type(m) for m in model.modules() if isinstance(m, _QuantFunctionalMixin)
        }:
            wrappers_removed += _shadow_forward_after(
                klass,
                _QuantFunctionalMixin,
                cleanup,
            )

        # The attention wrapper is a SEPARATE, OPTIONAL win (it only adds three quantizer
        # calls per attention per microbatch). Removing it would skip the q/k/v bmm
        # quantizers, so under KV-cache quantization we simply leave it in place for the
        # affected classes rather than refusing the whole optimization -- the recipe still
        # gets the materialized snap and the linear-wrapper removal, which are the bulk of
        # it.
        attention_classes: dict[type, type] = {}
        kv_quantized: set[type] = set()
        for module in model.modules():
            for base in type(module).__mro__:
                if base.__name__ != "_QuantTEDotProductAttention":
                    continue
                attention_classes[type(module)] = base
                if any(
                    getattr(getattr(module, attr, None), "is_enabled", False)
                    for attr in (
                        "q_bmm_quantizer",
                        "k_bmm_quantizer",
                        "v_bmm_quantizer",
                    )
                ):
                    kv_quantized.add(type(module))
                break
        skipped_attention = sorted(k.__name__ for k in kv_quantized)
        for klass, base in attention_classes.items():
            if klass not in kv_quantized:
                wrappers_removed += _shadow_forward_after(klass, base, cleanup)

        yield
    if verbose and rank == 0:
        print(
            f"[snap_cache] frozen-weight stage: {saved} weight snaps "
            f"materialized in place, {wrappers_removed} forward wrapper(s) removed "
            f"({skipped} quantizers disabled)"
            + (
                f"; kept the attention wrapper on {skipped_attention} "
                "(KV-cache quantization is active there)"
                if skipped_attention
                else ""
            )
            + "."
        )


@contextlib.contextmanager
def weight_snap_cache(
    model: torch.nn.Module,
    *,
    verbose: bool = False,
    rank: int = 0,
) -> Iterator[None]:
    """Cache snapped weights across the forwards run inside this context.

    Intended to wrap a single frozen-weight stage (e.g. ``get_logprobs``). Enrolls the
    model's weight quantizers in context-local state, enables caching for the duration,
    then frees the cached tensors on exit so the memory (a second bf16 copy of the weight
    shard) is only held during the stage — when vLLM is offloaded and the optimizer is on
    CPU.
    """
    _install_patch()
    context = _SnapCacheContext(
        eligible_quantizers=set(_weight_tensor_quantizers(model)),
        cached_snaps={},
        hits=0,
        misses=0,
        verify_left=_verify_budget(),
    )
    previous_context = _active_context()
    _state.context = context
    try:
        yield
    finally:
        _state.context = previous_context
        if verbose and rank == 0:
            print(
                f"[snap_cache] frozen-weight stage: {context.misses} weight snaps "
                f"computed, {context.hits} reused (=snaps avoided)."
            )
