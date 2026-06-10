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

"""Forward-hook tensor capture helpers for cross-engine numeric debugging."""

from __future__ import annotations

import inspect
import os
from collections.abc import Iterable, Mapping
from typing import Any

import torch


def _snapshot_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, tuple):
        return tuple(_snapshot_value(v) for v in value)
    if isinstance(value, list):
        return [_snapshot_value(v) for v in value]
    if isinstance(value, Mapping):
        return {k: _snapshot_value(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return type(value).__name__


def _snapshot_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "args": tuple(_snapshot_value(v) for v in args),
        "kwargs": {k: _snapshot_value(v) for k, v in kwargs.items()},
    }


def _get_attr_path(obj: Any, path: str) -> Any | None:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def find_decoder_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Find the decoder layer ModuleList for vLLM or Megatron language models."""
    candidate_paths = (
        "model.layers",
        "model.model.layers",
        "decoder.layers",
        "language_model.decoder.layers",
    )
    for path in candidate_paths:
        layers = _get_attr_path(model, path)
        if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
            return layers

    for _, module in model.named_modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            first = module[0]
            class_name = first.__class__.__name__.lower()
            if "layer" in class_name or "block" in class_name:
                return module

    raise RuntimeError("Could not find decoder layers on model.")


def _callable_info(fn: Any) -> dict[str, Any]:
    func = getattr(fn, "__func__", fn)
    try:
        source = inspect.getsource(func)
    except (OSError, TypeError):
        source = None
    return {
        "repr": repr(fn),
        "module": getattr(func, "__module__", None),
        "qualname": getattr(func, "__qualname__", None),
        "source_head": source[:800] if isinstance(source, str) else None,
    }


def inspect_vllm_layernorm_impl(model: torch.nn.Module) -> dict[str, Any]:
    """Inspect the runtime RMSNorm implementation used by vLLM layer 0."""
    from vllm.model_executor.layers import layernorm as vllm_layernorm
    from vllm.model_executor.layers.layernorm import RMSNorm

    layers = find_decoder_layers(model)
    layer0 = layers[0]
    module = getattr(layer0, "post_attention_layernorm", None)
    if module is None:
        module = next(
            (
                candidate
                for candidate in layer0.modules()
                if isinstance(candidate, RMSNorm)
            ),
            None,
        )
    if module is None:
        raise RuntimeError("layer 0 does not expose a vLLM RMSNorm module")

    return {
        "module_class": module.__class__.__module__
        + "."
        + module.__class__.__qualname__,
        "is_rmsnorm": isinstance(module, RMSNorm),
        "variance_epsilon": getattr(module, "variance_epsilon", None),
        "hidden_size": getattr(module, "hidden_size", None),
        "weight_dtype": str(getattr(getattr(module, "weight", None), "dtype", None)),
        "forward_method": _callable_info(getattr(module, "_forward_method", None)),
        "forward_cuda": _callable_info(getattr(module, "forward_cuda", None)),
        "class_forward_cuda": _callable_info(RMSNorm.forward_cuda),
        "fused_add_rms_norm": _callable_info(vllm_layernorm.fused_add_rms_norm),
        "rms_norm_batch_invariant": _callable_info(
            getattr(vllm_layernorm, "rms_norm_batch_invariant", None)
        ),
    }


def _normalise_capture_layers(
    capture_layers: Iterable[int] | None,
    num_layers: int,
) -> set[int]:
    if capture_layers is None:
        return {0}
    layers = {int(layer_idx) for layer_idx in capture_layers}
    invalid = sorted(
        layer_idx for layer_idx in layers if layer_idx < 0 or layer_idx >= num_layers
    )
    if invalid:
        raise ValueError(
            f"capture layer indices out of range for {num_layers} layers: {invalid}"
        )
    return layers


def install_debug_tensor_hooks(
    model: torch.nn.Module,
    capture_layers: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Capture layer inputs and first-layer module inputs/outputs.

    Selected capture layers get pre/post hooks on every submodule. Other layers
    get only a pre-hook on the layer entrypoint. Call-list fields capture every
    forward call; first-call fields are preserved for quick inspection.
    """
    layers = find_decoder_layers(model)
    capture_layer_set = _normalise_capture_layers(capture_layers, len(layers))
    capture: dict[str, Any] = {
        "module_inputs_by_layer": {},
        "module_outputs_by_layer": {},
        "module_input_calls_by_layer": {},
        "module_output_calls_by_layer": {},
        "first_layer_inputs": {},
        "first_layer_input_kwargs": {},
        "first_layer_outputs": {},
        "num_layers": len(layers),
        "captured_module_layers": sorted(capture_layer_set),
    }
    handles = []

    def save_input(
        layer_idx: int, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        layer_input_calls = capture["module_input_calls_by_layer"].setdefault(
            layer_idx, {}
        )
        layer_inputs = capture["module_inputs_by_layer"].setdefault(layer_idx, {})
        entry = _snapshot_call(args, kwargs)
        layer_input_calls.setdefault(name, []).append(entry)
        if name in layer_inputs:
            return
        layer_inputs[name] = entry
        if layer_idx == 0:
            capture["first_layer_inputs"][name] = entry["args"]
            capture["first_layer_input_kwargs"][name] = entry["kwargs"]

    def save_output(layer_idx: int, name: str, output: Any) -> None:
        layer_output_calls = capture["module_output_calls_by_layer"].setdefault(
            layer_idx, {}
        )
        layer_outputs = capture["module_outputs_by_layer"].setdefault(layer_idx, {})
        output_entry = _snapshot_value(output)
        layer_output_calls.setdefault(name, []).append(output_entry)
        if name in layer_outputs:
            return
        layer_outputs[name] = output_entry
        if layer_idx == 0:
            capture["first_layer_outputs"][name] = layer_outputs[name]

    def make_pre_hook(layer_idx: int, name: str):
        def hook(module, args, kwargs):  # noqa: ARG001
            save_input(layer_idx, name, args, kwargs)

        return hook

    def make_post_hook(layer_idx: int, name: str):
        def hook(module, args, kwargs, output):  # noqa: ARG001
            save_output(layer_idx, name, output)

        return hook

    for layer_idx, layer in enumerate(layers):
        if layer_idx in capture_layer_set:
            for module_name, module in layer.named_modules():
                capture_name = module_name or "<layer>"
                handles.append(
                    module.register_forward_pre_hook(
                        make_pre_hook(layer_idx, capture_name), with_kwargs=True
                    )
                )
                handles.append(
                    module.register_forward_hook(
                        make_post_hook(layer_idx, capture_name), with_kwargs=True
                    )
                )
            continue

        handles.append(
            layer.register_forward_pre_hook(
                make_pre_hook(layer_idx, "<layer>"), with_kwargs=True
            )
        )

    model._debug_tensor_capture = capture
    model._debug_tensor_capture_handles = handles
    return {
        "num_layers": len(layers),
        "num_hooks": len(handles),
        "captured_module_layers": sorted(capture_layer_set),
    }


def get_debug_tensor_capture(model: torch.nn.Module) -> dict[str, Any]:
    capture = getattr(model, "_debug_tensor_capture", None)
    if capture is None:
        return {}
    return capture


def save_debug_tensor_capture_from_env(model: torch.nn.Module) -> dict[str, Any]:
    path = os.environ["DEBUG_TENSOR_CAPTURE_PATH"]
    capture = get_debug_tensor_capture(model)
    torch.save(capture, path)
    layer_input_calls = capture.get("module_input_calls_by_layer", {})
    layer0_calls = layer_input_calls.get(0, {})
    return {
        "path": path,
        "num_layers": capture.get("num_layers"),
        "captured_module_layers": capture.get("captured_module_layers"),
        "num_first_layer_modules": len(capture.get("first_layer_inputs", {})),
        "num_layer0_input_calls": {
            name: len(calls) for name, calls in layer0_calls.items()
        },
    }
