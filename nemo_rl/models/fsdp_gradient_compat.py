# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in in-process FSDP compatibility hook for deferred CPU gradients.

The frozen experiment patched Torch on disk. Here we modify only the existing
function object's code, preserving aliases captured by AutoModel's uniform-dtype
wrapper. No site-packages files are written. An unknown wrapper or source layout
raises before backward; this is a pinned compatibility path, not a public FSDP API.
"""

import ast
import inspect
import textwrap
import threading
from types import FunctionType

_LOCK = threading.Lock()
_MARKER = "_nrl_gradient_stash_hook"
_ANCHOR = "            to_accumulate_grad = fsdp_param.sharded_param.grad is not None"
_INSERTION = """            if fsdp_param.offload_to_cpu:
                from nemo_rl.models.deferred_grad import stash
                if stash(fsdp_param, new_sharded_grad, post_reduce_stream):
                    flat_grad_offset += padded_unsharded_size.numel() // world_size
                    continue
"""


def _find_torch_reduce(function: FunctionType) -> FunctionType:
    """Unwrap only Torch decorators and the known AutoModel dtype wrapper."""
    seen = set()
    while id(function) not in seen:
        seen.add(id(function))
        if hasattr(function, "__wrapped__"):
            function = function.__wrapped__
            continue
        if function.__name__ == "foreach_reduce_uniform_dtype":
            function = inspect.getclosurevars(function).nonlocals[
                "original_foreach_reduce"
            ]
            continue
        if function.__name__ == "foreach_reduce" and function.__module__.endswith(
            "._fsdp_collectives"
        ):
            return function
        raise RuntimeError(f"Unsupported FSDP reduction wrapper: {function}")
    raise RuntimeError("Cyclic FSDP reduction wrapper")


def install_gradient_stash_hook() -> None:
    """Install once in a worker; disabled training paths never call this."""
    from torch.distributed.fsdp._fully_shard import _fsdp_collectives

    with _LOCK:
        function = _find_torch_reduce(_fsdp_collectives.foreach_reduce)
        if getattr(function, _MARKER, False):
            return
        source = textwrap.dedent(inspect.getsource(function))
        if source.count(_ANCHOR) != 1:
            raise RuntimeError(
                "Unsupported Torch foreach_reduce source for deferred offload"
            )
        if "from nemo_rl.models.deferred_grad import stash" in source:
            raise RuntimeError(
                "Torch already patched on disk; use an unpatched environment"
            )
        tree = ast.parse(source.replace(_ANCHOR, _INSERTION + _ANCHOR))
        definition = tree.body[0]
        if not isinstance(definition, ast.FunctionDef):
            raise RuntimeError("Expected a function definition for FSDP reduction")
        definition.decorator_list = []
        namespace = dict(function.__globals__)
        exec(compile(tree, function.__code__.co_filename, "exec"), namespace)
        replacement = namespace[function.__name__]
        if (
            replacement.__code__.co_freevars != function.__code__.co_freevars
            or inspect.signature(replacement) != inspect.signature(function)
        ):
            raise RuntimeError("FSDP reduction signature or closure changed")
        function.__code__ = replacement.__code__
        setattr(function, _MARKER, True)
