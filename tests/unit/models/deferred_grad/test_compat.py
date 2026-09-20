# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import hashlib
import inspect
from pathlib import Path

import pytest
import torch
from torch.distributed.fsdp._fully_shard import _fsdp_collectives

from nemo_rl.models import deferred_grad
from nemo_rl.models.fsdp_gradient_compat import (
    _find_torch_reduce,
    install_gradient_stash_hook,
)
from nemo_rl.optimizers.bf16_cpu_adamw import BF16CPUAdamW


def test_hook_preserves_function_alias_and_disk_source(monkeypatch):
    function = _find_torch_reduce(_fsdp_collectives.foreach_reduce)
    original_code = function.__code__
    original_signature = inspect.signature(function)
    source = Path(inspect.getfile(function))
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        install_gradient_stash_hook()
        assert function.__code__ is not original_code
        assert inspect.signature(function) == original_signature
        assert hashlib.sha256(source.read_bytes()).hexdigest() == before
        installed_code = function.__code__
        install_gradient_stash_hook()
        assert function.__code__ is installed_code
        # Match the actual AutoModel closure, whose wrapper has no __wrapped__.
        original_foreach_reduce = _fsdp_collectives.foreach_reduce

        def foreach_reduce_uniform_dtype(*args, **kwargs):
            return original_foreach_reduce(*args, **kwargs)

        monkeypatch.setattr(
            _fsdp_collectives, "foreach_reduce", foreach_reduce_uniform_dtype
        )
        install_gradient_stash_hook()
        assert function.__code__ is installed_code
    finally:
        function.__code__ = original_code
        if hasattr(function, "_nrl_gradient_stash_hook"):
            delattr(function, "_nrl_gradient_stash_hook")


def test_hook_rejects_unknown_wrapper():
    with pytest.raises(RuntimeError, match="Unsupported"):
        _find_torch_reduce(lambda: None)


def test_disabled_scope_does_not_install_hook(monkeypatch):
    from nemo_rl.models import fsdp_gradient_compat

    monkeypatch.setattr(
        fsdp_gradient_compat,
        "install_gradient_stash_hook",
        lambda: pytest.fail("disabled hook"),
    )
    with deferred_grad.backward_scope(enabled=False):
        assert not deferred_grad._active


def test_stash_outside_scope_is_noop():
    assert deferred_grad.stash(None, torch.ones(1), None) is False


@pytest.mark.parametrize(
    "flags,cpu,error",
    [
        (("0", "1", "0"), True, "requires"),
        (("1", "0", "1"), True, "requires"),
        (("1", "1", "1"), False, "CPU parameter"),
    ],
)
def test_invalid_switches_fail_early(monkeypatch, flags, cpu, error):
    for key, value in zip(
        ("DS41_DEFER_GRAD_OFFLOAD", "DS41_GPU_GRAD_NORM", "DS41_STREAM_ADAM"), flags
    ):
        monkeypatch.setenv(key, value)
    opt = BF16CPUAdamW(
        [torch.nn.Parameter(torch.ones(3))],
        lr=0.01,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        chunk_numel=2,
    )
    with pytest.raises(ValueError, match=error):
        deferred_grad.validate_configuration(cpu, opt)


def test_streaming_rejects_unclipped_gradients():
    with pytest.raises(RuntimeError, match="clipped pending"):
        deferred_grad.streamed_optimizer_step(None)
