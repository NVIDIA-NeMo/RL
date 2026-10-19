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

"""Opt-in exact vision-feature snapshots for rollout/learner parity debugging."""

from __future__ import annotations

import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F


def _dump_root() -> Path | None:
    explicit = os.environ.get("NEMO_RL_VISION_PARITY_DUMP_DIR")
    if explicit is not None:
        # An explicit empty override disables layer dumps independently of
        # the smaller TMPE outlier snapshots needed for training diagnostics.
        return Path(explicit.strip()).expanduser() if explicit.strip() else None
    outlier_path = os.environ.get("NEMO_RL_TMPE_OUTLIER_DUMP_PATH", "").strip()
    if outlier_path:
        return Path(outlier_path).expanduser().parent / "vision_parity"
    return None


def _tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().to(device="cpu").contiguous().view(torch.uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


@torch.no_grad()
def dump_vision_features(
    engine: str,
    pixels: torch.Tensor,
    features: torch.Tensor,
    imgs_sizes: Any,
    *,
    stage: str = "projected",
) -> Path | None:
    """Save one unique exact input/output pair on distributed rank zero only."""
    root = _dump_root()
    if root is None:
        return None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return None

    input_digest = _tensor_digest(pixels)
    feature_digest = _tensor_digest(features)
    safe_stage = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in stage
    )
    target = root / (
        f"{engine}-{safe_stage}-input-{input_digest[:16]}-feature-{feature_digest[:16]}.pt"
    )
    if target.exists():
        return target

    if torch.is_tensor(imgs_sizes):
        sizes = imgs_sizes.detach().to(device="cpu").reshape(-1, 2).tolist()
    else:
        sizes = [list(size) for size in imgs_sizes]
    payload = {
        "engine": engine,
        "stage": stage,
        "input_digest": input_digest,
        "feature_digest": feature_digest,
        "imgs_sizes": sizes,
        "pixels": pixels.detach().to(device="cpu").contiguous().clone(),
        "features": features.detach().to(device="cpu").contiguous().clone(),
    }

    root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=root, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _hidden_state(value: Any) -> torch.Tensor:
    """Extract a layer hidden state from vLLM or Megatron hook values."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError(f"Vision parity hook expected a tensor output, got {type(value)!r}")


def _batch_sequence_layout(engine: str, hidden_state: torch.Tensor) -> torch.Tensor:
    """Normalize vLLM BSH and Megatron SBH tensors to the same BSH layout."""
    if hidden_state.ndim != 3:
        raise ValueError(
            f"Vision parity hook expected a 3-D hidden state, got {tuple(hidden_state.shape)}"
        )
    if engine == "policy":
        return hidden_state.permute(1, 0, 2).contiguous()
    return hidden_state.contiguous()


def _flatten_batch_sequence_layout(engine: str, value: Any) -> torch.Tensor:
    """Normalize an attention intermediate to BSH, flattening head dimensions."""
    hidden_state = _hidden_state(value)
    if hidden_state.ndim < 3:
        raise ValueError(
            "Vision parity attention hook expected at least a 3-D tensor, "
            f"got {tuple(hidden_state.shape)}"
        )
    if engine == "policy":
        hidden_state = hidden_state.permute(
            1, 0, *range(2, hidden_state.ndim)
        ).contiguous()
    else:
        hidden_state = hidden_state.contiguous()
    return hidden_state.flatten(start_dim=2)


def _residual_branch_state(engine: str, value: Any) -> torch.Tensor:
    """Include Megatron's separately returned bias in a residual branch."""
    hidden_state = _hidden_state(value)
    if (
        engine == "policy"
        and isinstance(value, (tuple, list))
        and len(value) > 1
        and torch.is_tensor(value[1])
    ):
        hidden_state = hidden_state + value[1]
    return _batch_sequence_layout(engine, hidden_state)


@contextmanager
def vision_parity_layer_hooks(
    model: torch.nn.Module,
    engine: str,
    pixels: torch.Tensor,
    imgs_sizes: Any,
) -> Iterator[None]:
    """Capture RADIO's decoder input and per-layer outputs when dumping is enabled."""
    if _dump_root() is None:
        yield
        return

    if engine == "policy":
        layers = model.decoder.layers
    elif engine == "generation":
        layers = model.model.encoder.layers
    else:
        raise ValueError(f"Unsupported vision parity engine: {engine!r}")
    if not layers:
        raise ValueError(
            "Vision parity capture requires at least one RADIO encoder layer"
        )

    handles = []
    forward_restores = []

    def capture_input(_module, args, kwargs):
        value = kwargs.get("hidden_states") if "hidden_states" in kwargs else args[0]
        dump_vision_features(
            engine,
            pixels,
            _batch_sequence_layout(engine, _hidden_state(value)),
            imgs_sizes,
            stage="encoder_input",
        )

    handles.append(layers[0].register_forward_pre_hook(capture_input, with_kwargs=True))

    for layer_index, layer in enumerate(layers):

        def capture_output(_module, _args, _kwargs, output, *, index=layer_index):
            dump_vision_features(
                engine,
                pixels,
                _batch_sequence_layout(engine, _hidden_state(output)),
                imgs_sizes,
                stage=f"layer_{index:02d}",
            )

        if engine == "generation":
            original_forward = layer.forward

            def capture_generation_layer_output(
                *args,
                _original_forward=original_forward,
                _index=layer_index,
                **kwargs,
            ):
                output = _original_forward(*args, **kwargs)
                dump_vision_features(
                    engine,
                    pixels,
                    _batch_sequence_layout(engine, _hidden_state(output)),
                    imgs_sizes,
                    stage=f"layer_{_index:02d}",
                )
                return output

            layer.forward = capture_generation_layer_output
            forward_restores.append((layer, original_forward))
        else:
            handles.append(
                layer.register_forward_hook(capture_output, with_kwargs=True)
            )

    first_layer = layers[0]
    branch_modules = (
        {
            "layer_00_attention_branch": first_layer.self_attention,
            "layer_00_mlp_branch": first_layer.mlp,
        }
        if engine == "policy"
        else {
            "layer_00_attention_branch": first_layer.attn,
            "layer_00_mlp_branch": first_layer.mlp,
        }
    )
    for stage, module in branch_modules.items():

        def capture_branch(_module, _args, _kwargs, output, *, stage_name=stage):
            value = _residual_branch_state(engine, output)
            if (
                engine == "generation"
                and stage_name == "layer_00_mlp_branch"
                and getattr(_module.fc2, "skip_bias_add", False)
                and _module.fc2.bias is not None
            ):
                value = value + _module.fc2.bias
            dump_vision_features(
                engine,
                pixels,
                value,
                imgs_sizes,
                stage=stage_name,
            )
            scaled_value = value
            if engine == "generation":
                layer_scale = (
                    first_layer.ls1
                    if stage_name == "layer_00_attention_branch"
                    else first_layer.ls2
                )
                scaled_value = scaled_value * layer_scale
            dump_vision_features(
                engine,
                pixels,
                scaled_value,
                imgs_sizes,
                stage=f"{stage_name}_scaled",
            )

        handles.append(module.register_forward_hook(capture_branch, with_kwargs=True))

    attention_modules = (
        {
            "layer_00_qkv": first_layer.self_attention.linear_qkv,
            "layer_00_core_attention": first_layer.self_attention.core_attention,
            "layer_00_attention_projection": first_layer.self_attention.linear_proj,
        }
        if engine == "policy"
        else {
            "layer_00_norm1": first_layer.norm1,
            "layer_00_qkv": first_layer.attn.qkv,
            "layer_00_core_attention": first_layer.attn.attn,
            "layer_00_attention_projection": first_layer.attn.proj,
        }
    )
    for stage, module in attention_modules.items():

        def capture_attention_stage(
            _module, _args, _kwargs, output, *, stage_name=stage
        ):
            value = output
            if stage_name == "layer_00_attention_projection":
                value = _residual_branch_state(engine, output)
            else:
                value = _flatten_batch_sequence_layout(engine, output)
            dump_vision_features(
                engine,
                pixels,
                value,
                imgs_sizes,
                stage=stage_name,
            )

        handles.append(
            module.register_forward_hook(capture_attention_stage, with_kwargs=True)
        )

    projection_module = attention_modules["layer_00_attention_projection"]

    def capture_projection_local_reference(module, args, kwargs):
        """Capture the rank-local projection operands before TP reduction."""
        hidden_state = kwargs.get("input_")
        if hidden_state is None:
            hidden_state = kwargs.get("hidden_states")
        if hidden_state is None:
            hidden_state = args[0]
        hidden_state = _hidden_state(hidden_state)
        weight = module.weight
        local_output = F.linear(hidden_state, weight, None)
        dump_vision_features(
            engine,
            pixels,
            _batch_sequence_layout(engine, hidden_state),
            imgs_sizes,
            stage="layer_00_attention_projection_input",
        )
        dump_vision_features(
            engine,
            pixels,
            weight,
            imgs_sizes,
            stage="layer_00_attention_projection_weight",
        )
        dump_vision_features(
            engine,
            pixels,
            _batch_sequence_layout(engine, local_output),
            imgs_sizes,
            stage="layer_00_attention_projection_local_torch_reference",
        )

    handles.append(
        projection_module.register_forward_pre_hook(
            capture_projection_local_reference, with_kwargs=True
        )
    )

    mlp_module = first_layer.mlp
    mlp_fc1 = mlp_module.linear_fc1 if engine == "policy" else mlp_module.fc1
    mlp_fc2 = mlp_module.linear_fc2 if engine == "policy" else mlp_module.fc2

    if engine == "generation":
        dump_vision_features(
            engine,
            pixels,
            first_layer.ls1,
            imgs_sizes,
            stage="layer_00_ls1",
        )
        dump_vision_features(
            engine,
            pixels,
            first_layer.ls2,
            imgs_sizes,
            stage="layer_00_ls2",
        )

        def capture_generation_norm1_input(_module, args, kwargs):
            hidden_state = kwargs.get("hidden_states")
            if hidden_state is None:
                hidden_state = args[0]
            dump_vision_features(
                engine,
                pixels,
                _batch_sequence_layout(engine, _hidden_state(hidden_state)),
                imgs_sizes,
                stage="layer_00_norm1_input",
            )

        handles.append(
            first_layer.norm1.register_forward_pre_hook(
                capture_generation_norm1_input, with_kwargs=True
            )
        )

        def capture_generation_norm2_input(_module, args, kwargs):
            hidden_state = kwargs.get("hidden_states")
            if hidden_state is None:
                hidden_state = args[0]
            dump_vision_features(
                engine,
                pixels,
                _batch_sequence_layout(engine, _hidden_state(hidden_state)),
                imgs_sizes,
                stage="layer_00_norm2_input",
            )

        def capture_generation_norm2(_module, _args, _kwargs, output):
            dump_vision_features(
                engine,
                pixels,
                _batch_sequence_layout(engine, _hidden_state(output)),
                imgs_sizes,
                stage="layer_00_norm2",
            )

        handles.append(
            first_layer.norm2.register_forward_pre_hook(
                capture_generation_norm2_input, with_kwargs=True
            )
        )
        handles.append(
            first_layer.norm2.register_forward_hook(
                capture_generation_norm2, with_kwargs=True
            )
        )

    def capture_mlp_fc1(_module, _args, _kwargs, output):
        dump_vision_features(
            engine,
            pixels,
            _residual_branch_state(engine, output),
            imgs_sizes,
            stage="layer_00_mlp_fc1",
        )

    handles.append(mlp_fc1.register_forward_hook(capture_mlp_fc1, with_kwargs=True))

    def capture_mlp_fc2_local_reference(module, args, kwargs):
        hidden_state = kwargs.get("input_")
        if hidden_state is None:
            hidden_state = kwargs.get("hidden_states")
        if hidden_state is None:
            hidden_state = args[0]
        hidden_state = _hidden_state(hidden_state)
        local_output = F.linear(hidden_state, module.weight, None)
        dump_vision_features(
            engine,
            pixels,
            _batch_sequence_layout(engine, hidden_state),
            imgs_sizes,
            stage="layer_00_mlp_fc2_input",
        )
        dump_vision_features(
            engine,
            pixels,
            module.weight,
            imgs_sizes,
            stage="layer_00_mlp_fc2_weight",
        )
        dump_vision_features(
            engine,
            pixels,
            _batch_sequence_layout(engine, local_output),
            imgs_sizes,
            stage="layer_00_mlp_fc2_local_torch_reference",
        )

    handles.append(
        mlp_fc2.register_forward_pre_hook(
            capture_mlp_fc2_local_reference, with_kwargs=True
        )
    )

    if engine == "policy":
        qkv_module = first_layer.self_attention.linear_qkv

        def capture_policy_qkv_reference(module, args, kwargs):
            hidden_state = kwargs.get("x") if "x" in kwargs else args[0]
            dump_vision_features(
                engine,
                pixels,
                _batch_sequence_layout(engine, hidden_state),
                imgs_sizes,
                stage="layer_00_norm1_input",
            )
            eps = getattr(module, "eps", module.config.layernorm_epsilon)
            normalized = F.layer_norm(
                hidden_state,
                (hidden_state.shape[-1],),
                module.layer_norm_weight,
                getattr(module, "layer_norm_bias", None),
                eps,
            )
            reference_qkv = F.linear(
                normalized,
                module.weight,
                getattr(module, "bias", None),
            )
            dump_vision_features(
                engine,
                pixels,
                _batch_sequence_layout(engine, normalized),
                imgs_sizes,
                stage="layer_00_norm1_reference",
            )
            dump_vision_features(
                engine,
                pixels,
                _batch_sequence_layout(engine, reference_qkv),
                imgs_sizes,
                stage="layer_00_qkv_torch_reference",
            )

        handles.append(
            qkv_module.register_forward_pre_hook(
                capture_policy_qkv_reference, with_kwargs=True
            )
        )

        def capture_policy_mlp_fc1_reference(module, args, kwargs):
            hidden_state = kwargs.get("x") if "x" in kwargs else args[0]
            dump_vision_features(
                engine,
                pixels,
                _batch_sequence_layout(engine, hidden_state),
                imgs_sizes,
                stage="layer_00_norm2_input",
            )
            eps = getattr(module, "eps", module.config.layernorm_epsilon)
            normalized = F.layer_norm(
                hidden_state,
                (hidden_state.shape[-1],),
                module.layer_norm_weight,
                getattr(module, "layer_norm_bias", None),
                eps,
            )
            dump_vision_features(
                engine,
                pixels,
                _batch_sequence_layout(engine, normalized),
                imgs_sizes,
                stage="layer_00_norm2",
            )

        handles.append(
            mlp_fc1.register_forward_pre_hook(
                capture_policy_mlp_fc1_reference, with_kwargs=True
            )
        )

    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        for layer, original_forward in forward_restores:
            layer.forward = original_forward


__all__ = ["dump_vision_features", "vision_parity_layer_hooks"]
