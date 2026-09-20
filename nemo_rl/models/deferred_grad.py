# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental post-backward gradient offload; CPU parameters remain untouched."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class Pending:
    param: FSDPParam
    gradient: torch.Tensor
    event: torch.cuda.Event


_pending: dict[int, Pending] = {}
_lock = threading.Lock()
_active = False
_ready_for_optimizer = False


def stash(param: FSDPParam, gradient: torch.Tensor, stream: torch.cuda.Stream) -> bool:
    if not _active:
        return False
    if getattr(param.sharded_param, "_post_accumulate_grad_hooks", None):
        raise RuntimeError("Deferred offload does not support post-accumulate hooks")
    if gradient.device.type != "cuda":
        raise RuntimeError("Expected a CUDA gradient shard")
    with _lock:
        key = id(param)
        if key in _pending:
            prior = _pending[key]
            stream.wait_event(prior.event)
            prior.gradient.add_(gradient)
            prior.event = stream.record_event()
        else:
            # Own compact storage: retaining a view can retain a whole RS buffer.
            saved = gradient.clone()
            _pending[key] = Pending(param, saved, stream.record_event())
    return True


def flush() -> None:
    # Called after autograd has returned, before caller's gradient scaling/clipping.
    # Keep tensors alive until their producer events and CPU copies complete.
    with _lock:
        logger.info(
            f"GPU_GRAD_FLUSH_BEGIN time={time.time()} peak_allocated={torch.cuda.max_memory_allocated()}"
        )
        count = len(_pending)
        gpu_bytes = sum(
            e.gradient.numel() * e.gradient.element_size() for e in _pending.values()
        )
        for entry in _pending.values():
            entry.event.synchronize()
            host = entry.gradient.to("cpu", non_blocking=False)
            param = entry.param
            if param.sharded_param.grad is None:
                param.sharded_param.grad = param.to_sharded_dtensor(host)
            else:
                param.sharded_param.grad.to_local().add_(host)
        _pending.clear()
        logger.info(
            f"DEFERRED_GRAD_FLUSH rank={os.environ.get('RANK', '?')} shards={count} bytes={gpu_bytes}"
        )


@contextmanager
def backward_scope(
    enabled: bool | None = None, keep_on_gpu: bool | None = None
) -> Iterator[None]:
    global _active
    if enabled is None:
        enabled = os.environ.get("DS41_DEFER_GRAD_OFFLOAD", "0") == "1"
    if not enabled:
        yield
        return
    if keep_on_gpu is None:
        keep_on_gpu = os.environ.get("DS41_GPU_GRAD_NORM", "0") == "1"
    if _ready_for_optimizer or _active or (_pending and not keep_on_gpu):
        raise RuntimeError("Deferred offload scope reentered or stale gradients exist")
    from nemo_rl.models.fsdp_gradient_compat import install_gradient_stash_hook

    install_gradient_stash_hook()
    _active = True
    try:
        yield
        if not keep_on_gpu:
            flush()
    except BaseException:
        # A failed backward is fatal to this worker; never silently reuse gradients.
        raise
    finally:
        _active = False


class GradientModelView:
    """Named gradient-only DTensor proxies for the existing AutoModel norm helper.

    Proxy data aliases its gradient; no GPU parameter copy is allocated. The
    norm helper only reads metadata and modifies .grad, never parameter data.
    """

    def __init__(self, named: list[tuple[str, torch.Tensor]]) -> None:
        self.named = named

    def parameters(self) -> Iterator[torch.Tensor]:
        return (p for _, p in self.named)

    def named_parameters(self) -> Iterator[tuple[str, torch.Tensor]]:
        return iter(self.named)


@torch.no_grad()
def gpu_scale_and_clip(
    max_grad_norm: float | None,
    model_parts: list[torch.nn.Module],
    stream_to_optimizer: bool | None = None,
    **kwargs: Any,
) -> torch.Tensor | float:
    global _ready_for_optimizer
    from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

    if _active or _ready_for_optimizer:
        raise RuntimeError("Cannot clip during backward or clip twice")
    if not _pending:
        raise RuntimeError("No deferred gradients at GPU norm boundary")
    by_param = {id(e.param.sharded_param): e for e in _pending.values()}
    seen = set()
    views = []
    stream = torch.cuda.current_stream()
    for model in model_parts:
        named = []
        for name, p in model.named_parameters():
            if p.grad is not None:
                raise RuntimeError(
                    f"Unexpected preexisting gradient outside GPU stash: {name}"
                )
            entry = by_param.get(id(p))
            if entry is None:
                continue
            seen.add(id(p))
            stream.wait_event(entry.event)
            entry.gradient.record_stream(stream)
            g = entry.param.to_sharded_dtensor(entry.gradient)
            proxy = g.detach().requires_grad_(True)
            proxy.grad = g
            divisor = getattr(p, "_nemo_model_owned_grad_divisor", None)
            if divisor is not None:
                proxy._nemo_model_owned_grad_divisor = divisor
            named.append((name, proxy))
        views.append(GradientModelView(named))
    if seen != set(by_param):
        raise RuntimeError("Deferred gradients not represented in model parameters")
    # Preserve original parameter order, mesh objects, placements, names and
    # owner divisors: the exact installed helper determines scaling/reductions.
    logger.info(
        f"GPU_GRAD_NORM_BEGIN rank={os.environ.get('RANK', '?')} shards={len(seen)} time={time.time()}"
    )
    norm = scale_grads_and_clip_grad_norm(max_grad_norm, views, **kwargs)
    if isinstance(norm, torch.Tensor):
        norm = norm.detach().cpu()
    if stream_to_optimizer is None:
        stream_to_optimizer = os.environ.get("DS41_STREAM_ADAM", "0") == "1"
    if stream_to_optimizer:
        # Record readiness after scaling/clipping even if norm computation disabled.
        ready = stream.record_event()
        for entry in _pending.values():
            entry.event = ready
        _ready_for_optimizer = True
    else:
        flush()
    logger.info(f"GPU_GRAD_NORM_END rank={os.environ.get('RANK', '?')} norm={norm}")
    return norm


def scale_grads_and_clip_grad_norm(
    max_grad_norm: float | None, model_parts: list[torch.nn.Module], **kwargs: Any
) -> torch.Tensor | float:
    if os.environ.get("DS41_GPU_GRAD_NORM", "0") == "1":
        return gpu_scale_and_clip(max_grad_norm, model_parts, **kwargs)
    from nemo_automodel.components.training.utils import (
        scale_grads_and_clip_grad_norm as original,
    )

    return original(max_grad_norm, model_parts, **kwargs)


@torch.no_grad()
def streamed_optimizer_step(optimizer: torch.optim.Optimizer) -> Any:
    global _ready_for_optimizer
    if _active or not _ready_for_optimizer or not _pending:
        raise RuntimeError("Streaming Adam requires clipped pending GPU gradients")
    shards = {}
    for entry in _pending.values():
        entry.event.synchronize()
        shards[id(entry.param.sharded_param)] = entry.gradient
    start = time.time()
    total = sum(g.numel() * g.element_size() for g in shards.values())
    logger.info(f"STREAM_ADAM_BEGIN rank={os.environ.get('RANK', '?')} bytes={total}")
    result = optimizer.step(gradient_shards=shards)
    _pending.clear()
    shards.clear()
    _ready_for_optimizer = False
    logger.info(
        f"STREAM_ADAM_END rank={os.environ.get('RANK', '?')} seconds={time.time() - start} staging_bytes={optimizer.last_streamed_chunk_bytes}"
    )
    return result


def optimizer_step(optimizer: torch.optim.Optimizer) -> Any:
    if os.environ.get("DS41_STREAM_ADAM", "0") == "1":
        return streamed_optimizer_step(optimizer)
    return optimizer.step()


def validate_configuration(cpu_offload: bool, optimizer: torch.optim.Optimizer) -> None:
    """Reject incompatible switches before entering a large distributed backward."""
    from nemo_rl.optimizers.bf16_cpu_adamw import BF16CPUAdamW

    defer = os.environ.get("DS41_DEFER_GRAD_OFFLOAD", "0") == "1"
    gpu_norm = os.environ.get("DS41_GPU_GRAD_NORM", "0") == "1"
    streaming = os.environ.get("DS41_STREAM_ADAM", "0") == "1"
    if (gpu_norm and not defer) or (streaming and not gpu_norm):
        raise ValueError(
            "Streaming Adam requires GPU norm and deferred gradient offload"
        )
    if defer and not cpu_offload:
        raise ValueError("Deferred gradients require FSDP CPU parameter offload")
    if streaming and not isinstance(optimizer, BF16CPUAdamW):
        raise ValueError("Streaming GPU gradients require BF16CPUAdamW")
