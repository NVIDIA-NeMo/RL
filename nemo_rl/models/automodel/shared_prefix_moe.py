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

"""Shared-prefix adapters for AutoModel's native Qwen3-MoE implementation.

AutoModel owns expert parallelism, expert checkpoint conversion, and the token
dispatcher. This module supplies the semantics changed by a compact ZoRRo
forward plus narrow, source-guarded compatibility fixes for the pinned native
Qwen3-MoE implementation:

* native Qwen3-MoE attention must use the shared-prefix FA2 layout; and
* router-wide quantities must count logical tokens, not physical compact tokens.

The adapters are installed on model instances, so ordinary AutoModel models and
ordinary forwards retain their upstream behavior.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import re
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Iterator, TypeAlias

import torch
from torch import nn

from nemo_rl.models.automodel.shared_prefix import (
    SharedPrefixLayout,
    shared_prefix_flash_attention_forward,
)

_CUSTOM_QWEN3_MOE_FLAG = "_nemo_shared_prefix_custom_qwen3_moe"

_SourceReplacement: TypeAlias = tuple[str, str, int]
_MethodPatch: TypeAlias = tuple[
    Any,
    type[nn.Module],
    str,
    tuple[_SourceReplacement, ...],
]


@dataclass(frozen=True)
class _EPFreeCheckpointStage:
    """One grouped-expert tensor staged through rank-local HF placeholders."""

    native_tensor: Any
    local_tensor: torch.Tensor
    expert_ids: tuple[int, ...]
    hf_prefix: str
    layer_num: str
    expert_segment: str
    projection: str
    expected_shapes: dict[str, torch.Size]


def _qwen3_moe_configured_dtype(config: Any) -> torch.dtype:
    from nemo_automodel.shared.utils import dtype_from_str

    return dtype_from_str(getattr(config, "torch_dtype", None), torch.bfloat16)


def enable_qwen3_moe_configured_dtype() -> None:
    """Backport configured-dtype construction for native Qwen3-MoE.

    The pinned AutoModel revision constructs several native Qwen3-MoE
    parameters with helper defaults (bf16), even when NeMo-RL requests fp32
    master weights. FSDP requires a uniform master dtype before it can shard the
    model, so post-construction casting is too late.

    Patch all affected methods atomically. Every source edit is exact and
    guarded, and no class is mutated until every replacement has compiled. An
    upstream source change therefore fails visibly without leaving a partial
    process-wide patch installed.
    """
    from nemo_automodel.components.models.qwen3_moe import layers as layers_module
    from nemo_automodel.components.models.qwen3_moe import model as model_module

    marker = "_nemo_qwen3_moe_configured_dtype"
    targets: tuple[_MethodPatch, ...] = (
        (
            layers_module,
            layers_module.Qwen3MoeAttention,
            "__init__",
            (
                (
                    "        super().__init__()\n",
                    "        super(Qwen3MoeAttention, self).__init__()\n",
                    1,
                ),
                (
                    '        attention_bias = getattr(config, "attention_bias", False)\n',
                    '        attention_bias = getattr(config, "attention_bias", False)\n'
                    "        dtype = _nemo_qwen3_moe_configured_dtype(config)\n",
                    1,
                ),
                (
                    "attention_bias\n        )",
                    "attention_bias, dtype=dtype\n        )",
                    4,
                ),
                (
                    "self.q_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps)",
                    "self.q_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps, dtype=dtype)",
                    1,
                ),
                (
                    "self.k_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps)",
                    "self.k_norm = initialize_rms_norm_module(backend.rms_norm, self.head_dim, eps=config.rms_norm_eps, dtype=dtype)",
                    1,
                ),
            ),
        ),
        (
            model_module,
            model_module.Block,
            "__init__",
            (
                (
                    "        super().__init__()\n",
                    "        super(Block, self).__init__()\n",
                    1,
                ),
                (
                    "        self.self_attn = Qwen3MoeAttention(config, backend)\n",
                    "        self.self_attn = Qwen3MoeAttention(config, backend)\n"
                    "        dtype = _nemo_qwen3_moe_configured_dtype(config)\n",
                    1,
                ),
                (
                    "self.mlp = MLP(config.hidden_size, config.intermediate_size, backend.linear)",
                    "self.mlp = MLP(config.hidden_size, config.intermediate_size, backend.linear, dtype=dtype)",
                    1,
                ),
                (
                    "self.input_layernorm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)",
                    "self.input_layernorm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)",
                    1,
                ),
                (
                    "backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps\n        )",
                    "backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype\n        )",
                    1,
                ),
            ),
        ),
        (
            model_module,
            model_module.Qwen3MoeModel,
            "__init__",
            (
                (
                    "        super().__init__()\n",
                    "        super(Qwen3MoeModel, self).__init__()\n",
                    1,
                ),
                (
                    '            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")\n',
                    '            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")\n'
                    "\n        model_dtype = _nemo_qwen3_moe_configured_dtype(config)\n",
                    1,
                ),
                (
                    "            softmax_before_topk=True,\n",
                    "            softmax_before_topk=True,\n"
                    "            dtype=model_dtype,\n",
                    1,
                ),
                (
                    "        self.embed_tokens = nn.Embedding(\n"
                    "            config.vocab_size, config.hidden_size, dtype=get_dtype(config.torch_dtype, torch.bfloat16)\n"
                    "        )",
                    "        self.embed_tokens = nn.Embedding(\n"
                    "            config.vocab_size, config.hidden_size, dtype=model_dtype\n"
                    "        )",
                    1,
                ),
                (
                    "self.norm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps)",
                    "self.norm = initialize_rms_norm_module(backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype)",
                    1,
                ),
            ),
        ),
        (
            model_module,
            model_module.Qwen3MoeForCausalLM,
            "__init__",
            (
                (
                    "        super().__init__()\n",
                    "        super(Qwen3MoeForCausalLM, self).__init__()\n",
                    1,
                ),
                (
                    "        self.model = Qwen3MoeModel(config, backend=self.backend, moe_config=moe_config, moe_overrides=moe_overrides)\n",
                    "        self.model = Qwen3MoeModel(config, backend=self.backend, moe_config=moe_config, moe_overrides=moe_overrides)\n"
                    "        model_dtype = _nemo_qwen3_moe_configured_dtype(config)\n",
                    1,
                ),
                (
                    "self.lm_head = initialize_linear_module(self.backend.linear, config.hidden_size, config.vocab_size, bias=False)",
                    "self.lm_head = initialize_linear_module(self.backend.linear, config.hidden_size, config.vocab_size, bias=False, dtype=model_dtype)",
                    1,
                ),
                (
                    "dtype=get_dtype(config.torch_dtype, torch.bfloat16)",
                    "dtype=model_dtype",
                    1,
                ),
            ),
        ),
        (
            model_module,
            model_module.Qwen3MoeForCausalLM,
            "initialize_weights",
            (
                (
                    "        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16\n",
                    "        self, buffer_device: torch.device | None = None, dtype: torch.dtype | None = None\n",
                    1,
                ),
                (
                    '        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")\n',
                    "        dtype = dtype or _nemo_qwen3_moe_configured_dtype(self.config)\n"
                    '        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")\n',
                    1,
                ),
            ),
        ),
    )

    methods = tuple(getattr(owner, name) for _, owner, name, _ in targets)
    installed = tuple(bool(getattr(method, marker, False)) for method in methods)
    if all(installed):
        return
    if any(installed):
        raise RuntimeError(
            "native Qwen3-MoE configured-dtype backport is only partially installed"
        )

    compiled: list[tuple[type[nn.Module], str, Any]] = []
    for (module, owner, name, replacements), original in zip(targets, methods):
        try:
            source = inspect.getsource(original)
        except (OSError, TypeError) as error:
            raise RuntimeError(
                f"Cannot inspect AutoModel {owner.__name__}.{name} for the "
                "Qwen3-MoE configured-dtype backport"
            ) from error

        patched_source = source
        for old, new, expected_count in replacements:
            actual_count = patched_source.count(old)
            if actual_count != expected_count:
                raise RuntimeError(
                    f"AutoModel {owner.__name__}.{name} no longer matches the "
                    "expected Qwen3-MoE configured-dtype source "
                    f"({actual_count} matches, expected {expected_count}); "
                    "remove or update the NeMo-RL backport"
                )
            patched_source = patched_source.replace(old, new)

        namespace = {
            **module.__dict__,
            "_nemo_qwen3_moe_configured_dtype": _qwen3_moe_configured_dtype,
        }
        try:
            exec(textwrap.dedent(patched_source), namespace)
        except (NameError, SyntaxError, TypeError) as error:
            raise RuntimeError(
                f"Cannot compile AutoModel {owner.__name__}.{name} for the "
                "Qwen3-MoE configured-dtype backport"
            ) from error
        patched = namespace[name]
        patched.__module__ = original.__module__
        patched.__qualname__ = original.__qualname__
        patched.__doc__ = original.__doc__
        setattr(patched, marker, True)
        compiled.append((owner, name, patched))

    for owner, name, patched in compiled:
        setattr(owner, name, patched)


def enable_qwen3_moe_ep_router_gradients() -> None:
    """Backport AutoModel's differentiable EP routing-weight gather.

    AutoModel versions before NVIDIA-NeMo/Automodel#2995 gather routing
    weights with a detached collective inside ``GroupedExperts.forward``.
    Replacing the function is deliberately fail-closed: the source guard makes
    an upstream implementation change visible instead of applying a stale
    whole-function patch silently.
    """
    from nemo_automodel.components.moe import experts as experts_module

    grouped_experts = experts_module.GroupedExperts
    if getattr(grouped_experts.forward, "_nemo_ep_router_gradients", False):
        return

    try:
        source = inspect.getsource(grouped_experts.forward)
    except (OSError, TypeError) as error:
        raise RuntimeError(
            "Cannot inspect AutoModel GroupedExperts.forward for the "
            "Qwen3-MoE EP router-gradient backport"
        ) from error
    old = "weights.float(), differentiable=False"
    if old not in source:
        raise RuntimeError(
            "AutoModel GroupedExperts.forward no longer contains the expected "
            "detached EP routing-weight gather; remove or update the NeMo-RL "
            "Qwen3-MoE EP backport"
        )
    if source.count(old) != 1:
        raise RuntimeError(
            "AutoModel GroupedExperts.forward contains an unexpected number of "
            "detached EP routing-weight gathers"
        )

    namespace = dict(experts_module.__dict__)
    exec(
        textwrap.dedent(source).replace(
            old,
            "weights.float(), differentiable=True",
        ),
        namespace,
    )
    patched = namespace["forward"]
    patched.__module__ = grouped_experts.forward.__module__
    patched.__qualname__ = grouped_experts.forward.__qualname__
    patched.__doc__ = grouped_experts.forward.__doc__
    patched._nemo_ep_router_gradients = True
    grouped_experts.forward = patched


def enable_qwen3_moe_checkpoint_wrapper_compatibility() -> None:
    """Make native Qwen3-MoE's MLP dispatch transparent to AC wrappers.

    AutoModel's default activation-checkpointing strategy wraps ``Block.mlp``
    in a ``CheckpointWrapper``.  The pinned native block dispatches dense versus
    MoE MLPs with a direct ``isinstance(self.mlp, ...)`` check, so the wrapper
    makes an otherwise valid forward fail before it is called.  Inspect the
    wrapped module for type dispatch while still calling the wrapper itself.
    """
    from nemo_automodel.components.models.qwen3_moe import model as model_module

    block = model_module.Block
    marker = "_nemo_qwen3_moe_checkpoint_wrapper_compatibility"
    original = block._mlp
    if getattr(original, marker, False):
        return

    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as error:
        raise RuntimeError(
            "Cannot inspect AutoModel Block._mlp for the Qwen3-MoE "
            "checkpoint-wrapper compatibility fix"
        ) from error

    old = (
        "        if isinstance(self.mlp, MLP):\n"
        "            return self.mlp(x)\n"
        "        else:\n"
        "            assert isinstance(self.mlp, MoE)\n"
        "            return self.mlp(x, padding_mask)\n"
    )
    new = (
        "        mlp = self.mlp\n"
        '        target = getattr(mlp, "_checkpoint_wrapped_module", mlp)\n'
        "        if isinstance(target, MLP):\n"
        "            return mlp(x)\n"
        "        assert isinstance(target, MoE)\n"
        "        return mlp(x, padding_mask)\n"
    )
    if source.count(old) != 1:
        raise RuntimeError(
            "AutoModel Block._mlp no longer matches the expected Qwen3-MoE "
            "checkpoint-wrapper source; remove or update the NeMo-RL fix"
        )

    namespace = dict(model_module.__dict__)
    try:
        exec(textwrap.dedent(source.replace(old, new)), namespace)
    except (NameError, SyntaxError, TypeError) as error:
        raise RuntimeError(
            "Cannot compile AutoModel Block._mlp for the Qwen3-MoE "
            "checkpoint-wrapper compatibility fix"
        ) from error
    patched = namespace["_mlp"]
    patched.__module__ = original.__module__
    patched.__qualname__ = original.__qualname__
    patched.__doc__ = original.__doc__
    setattr(patched, marker, True)
    block._mlp = patched


def _guard_automodel_source(owner: type[Any], name: str, expected_sha256: str) -> None:
    """Fail closed when a runtime backport no longer matches its pinned source."""
    try:
        source = inspect.getblock(inspect.getsourcelines(getattr(owner, name))[0])
    except (OSError, TypeError) as error:
        raise RuntimeError(
            f"Cannot inspect AutoModel {owner.__name__}.{name} for the "
            "Qwen3-MoE EP-free checkpoint backport"
        ) from error
    digest = hashlib.sha256("".join(source).encode()).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            f"AutoModel {owner.__name__}.{name} no longer matches the pinned "
            "Qwen3-MoE EP-free checkpoint source; remove or update the "
            f"NeMo-RL backport (expected {expected_sha256}, got {digest})"
        )


def _torch_chunk_shard_size_and_offset(
    dim_size: int,
    num_chunks: int,
    rank: int,
) -> tuple[int, int]:
    """Return the contiguous shard described by pinned DTensor ``Shard``."""
    full_chunk_size = (dim_size + num_chunks - 1) // num_chunks
    start = min(dim_size, full_chunk_size * rank)
    end = min(dim_size, start + full_chunk_size)
    return end - start, start


def _ep_free_expert_ids(tensor: Any, n_experts: int) -> tuple[int, ...] | None:
    """Return rank-local expert IDs for the one supported EP-free FSDP layout."""
    from torch.distributed.tensor import DTensor, Shard

    if not isinstance(tensor, DTensor):
        return None
    mesh = tensor.device_mesh
    if "ep" in mesh.mesh_dim_names:
        return None
    if mesh.ndim != 1 or len(tensor.placements) != 1:
        raise ValueError(
            "Qwen3-MoE EP-free checkpoint staging requires a one-dimensional FSDP mesh"
        )
    placement = tensor.placements[0]
    if not isinstance(placement, Shard) or placement.dim != 0:
        return None
    if tensor.shape[0] != n_experts:
        raise ValueError(
            f"Qwen3-MoE grouped tensor has {tensor.shape[0]} experts, "
            f"expected {n_experts}"
        )

    mesh_size = mesh.size()
    mesh_rank = mesh.get_local_rank()
    # DTensor Shard follows torch.chunk semantics: every non-final chunk has
    # ceil(N / world_size) entries, and trailing ranks can be empty. A balanced
    # divmod partition gives different expert IDs for uneven expert counts.
    local_count, start = _torch_chunk_shard_size_and_offset(
        n_experts, mesh_size, mesh_rank
    )
    end = start + local_count
    if tensor.to_local().shape[0] != local_count:
        raise ValueError(
            "Qwen3-MoE EP-free grouped tensor local expert axis does not "
            f"match its Shard(0) placement ({tensor.to_local().shape[0]} != "
            f"{local_count})"
        )
    return tuple(range(start, end))


def _stage_ep_free_expert_tensor(
    adapter: Any,
    fqn: str,
    tensor: Any,
    *,
    quantization: bool,
) -> list[tuple[str, torch.Tensor]] | None:
    """Create contiguous HF destinations while retaining their grouped target."""
    expert_segment = adapter._expert_path_segment
    if not (
        f".{expert_segment}." in fqn
        and fqn.endswith((".gate_and_up_projs", ".down_projs"))
    ):
        return None

    n_experts = adapter.moe_config.n_routed_experts
    expert_ids = _ep_free_expert_ids(tensor, n_experts)
    if expert_ids is None:
        return None
    if quantization:
        raise ValueError(
            "Qwen3-MoE EP-free staged checkpoint loading does not support "
            "quantized expert destinations"
        )
    if getattr(adapter.backend, "experts", None) == "te":
        raise ValueError(
            "Qwen3-MoE EP-free staged checkpoint loading does not support "
            "Transformer Engine experts"
        )

    layer_match = re.search(r"layers\.(\d+)", fqn)
    if layer_match is None:
        raise ValueError(f"Cannot determine Qwen3-MoE layer from {fqn!r}")
    layer_num = layer_match.group(1)
    local_tensor = tensor.to_local()
    inter_dim = adapter.moe_config.moe_inter_dim
    hf_prefix = adapter._hf_prefix
    result: list[tuple[str, torch.Tensor]] = []
    expected_shapes: dict[str, torch.Size] = {}

    if fqn.endswith(".gate_and_up_projs"):
        expected_last_dim = inter_dim * (2 if adapter._is_gated_moe else 1)
        if local_tensor.ndim != 3 or local_tensor.shape[2] != expected_last_dim:
            raise ValueError(
                "Qwen3-MoE EP-free gate/up tensor has an unexpected local shape "
                f"{tuple(local_tensor.shape)}"
            )
        projection = "gate_and_up_projs"
        for local_index, expert_id in enumerate(expert_ids):
            expert = local_tensor[local_index]
            if adapter._is_gated_moe:
                gate_key = (
                    f"{hf_prefix}layers.{layer_num}.{expert_segment}."
                    f"{expert_id}.gate_proj.weight"
                )
                up_key = (
                    f"{hf_prefix}layers.{layer_num}.{expert_segment}."
                    f"{expert_id}.up_proj.weight"
                )
                gate = expert[:, :inter_dim].transpose(0, 1).contiguous()
                up = expert[:, inter_dim:].transpose(0, 1).contiguous()
                result.extend(((gate_key, gate), (up_key, up)))
                expected_shapes[gate_key] = gate.shape
                expected_shapes[up_key] = up.shape
            else:
                up_key = (
                    f"{hf_prefix}layers.{layer_num}.{expert_segment}."
                    f"{expert_id}.up_proj.weight"
                )
                up = expert.transpose(0, 1).contiguous()
                result.append((up_key, up))
                expected_shapes[up_key] = up.shape
    else:
        if local_tensor.ndim != 3 or local_tensor.shape[1] != inter_dim:
            raise ValueError(
                "Qwen3-MoE EP-free down tensor has an unexpected local shape "
                f"{tuple(local_tensor.shape)}"
            )
        projection = "down_projs"
        for local_index, expert_id in enumerate(expert_ids):
            key = (
                f"{hf_prefix}layers.{layer_num}.{expert_segment}."
                f"{expert_id}.down_proj.weight"
            )
            value = local_tensor[local_index].transpose(0, 1).contiguous()
            result.append((key, value))
            expected_shapes[key] = value.shape

    stages = getattr(adapter, "_nemo_ep_free_checkpoint_stages", None)
    if stages is None:
        stages = {}
        adapter._nemo_ep_free_checkpoint_stages = stages
    if fqn in stages:
        raise RuntimeError(f"Qwen3-MoE checkpoint tensor {fqn!r} was staged twice")
    stages[fqn] = _EPFreeCheckpointStage(
        native_tensor=tensor,
        local_tensor=local_tensor,
        expert_ids=expert_ids,
        hf_prefix=hf_prefix,
        layer_num=layer_num,
        expert_segment=expert_segment,
        projection=projection,
        expected_shapes=expected_shapes,
    )
    return result


def _restore_ep_free_expert_stages(
    adapter: Any,
    hf_state_dict: dict[str, Any],
    original_from_hf: Any,
    device_mesh: Any,
) -> dict[str, Any]:
    """Validate every staged value, then atomically copy into grouped storage."""
    stages: dict[str, _EPFreeCheckpointStage] = (
        getattr(adapter, "_nemo_ep_free_checkpoint_stages", None) or {}
    )
    if not stages:
        return original_from_hf(adapter, hf_state_dict, device_mesh)

    # A staged expert load is specifically the EP-free path; passing a MoE mesh
    # here would mix two different ownership models.
    if device_mesh is not None:
        raise ValueError("Qwen3-MoE EP-free checkpoint stages require device_mesh=None")

    expected_keys = {key for stage in stages.values() for key in stage.expected_shapes}
    expert_pattern = re.compile(
        rf"(?:model\.)?(?:language_model\.)?layers\.\d+\."
        rf"{re.escape(adapter._expert_path_segment)}\.\d+\."
        r"(?:gate_proj|up_proj|down_proj)\.weight$"
    )
    observed_keys = {key for key in hf_state_dict if expert_pattern.fullmatch(key)}
    if observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        unexpected = sorted(observed_keys - expected_keys)
        raise RuntimeError(
            "Qwen3-MoE rank-local expert checkpoint keys do not match the "
            f"staged FSDP shard (missing={missing[:5]}, "
            f"unexpected={unexpected[:5]})"
        )

    by_layer: dict[tuple[str, str, str], dict[str, _EPFreeCheckpointStage]] = {}
    for stage in stages.values():
        group = by_layer.setdefault(
            (stage.hf_prefix, stage.layer_num, stage.expert_segment), {}
        )
        group[stage.projection] = stage
    for layer, projections in by_layer.items():
        required = {"gate_and_up_projs", "down_projs"}
        if set(projections) != required:
            raise RuntimeError(
                f"Qwen3-MoE staged layer {layer} is missing grouped projections"
            )
        if (
            projections["gate_and_up_projs"].expert_ids
            != projections["down_projs"].expert_ids
        ):
            raise RuntimeError(
                f"Qwen3-MoE staged layer {layer} has inconsistent expert IDs"
            )

    # Validate the complete batch before the first destination write.
    for stage in stages.values():
        for key, expected_shape in stage.expected_shapes.items():
            value = hf_state_dict[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Qwen3-MoE checkpoint value {key!r} is not a tensor")
            if value.shape != expected_shape:
                raise ValueError(
                    f"Qwen3-MoE checkpoint value {key!r} has shape "
                    f"{tuple(value.shape)}, expected {tuple(expected_shape)}"
                )
            if value.dtype != stage.local_tensor.dtype:
                raise ValueError(
                    f"Qwen3-MoE checkpoint value {key!r} has dtype {value.dtype}, "
                    f"expected {stage.local_tensor.dtype}"
                )
            if value.device != stage.local_tensor.device:
                raise ValueError(
                    f"Qwen3-MoE checkpoint value {key!r} is on {value.device}, "
                    f"expected {stage.local_tensor.device}"
                )

    # Let the upstream adapter validate/convert every non-expert value before
    # mutating grouped expert storage.  This makes adapter failures as well as
    # expert validation failures transactional with respect to the model.
    remaining = {
        key: value for key, value in hf_state_dict.items() if key not in expected_keys
    }
    restored = original_from_hf(adapter, remaining, device_mesh)

    with torch.no_grad():
        for stage in stages.values():
            inter_dim = adapter.moe_config.moe_inter_dim
            for local_index, expert_id in enumerate(stage.expert_ids):
                base = (
                    f"{stage.hf_prefix}layers.{stage.layer_num}."
                    f"{stage.expert_segment}.{expert_id}"
                )
                destination = stage.local_tensor[local_index]
                if stage.projection == "gate_and_up_projs":
                    if adapter._is_gated_moe:
                        gate = hf_state_dict[f"{base}.gate_proj.weight"]
                        up = hf_state_dict[f"{base}.up_proj.weight"]
                        destination[:, :inter_dim].copy_(gate.transpose(0, 1))
                        destination[:, inter_dim:].copy_(up.transpose(0, 1))
                    else:
                        up = hf_state_dict[f"{base}.up_proj.weight"]
                        destination.copy_(up.transpose(0, 1))
                else:
                    down = hf_state_dict[f"{base}.down_proj.weight"]
                    destination.copy_(down.transpose(0, 1))

    restored.update(
        {native_key: stage.native_tensor for native_key, stage in stages.items()}
    )
    return restored


def enable_qwen3_moe_ep_free_checkpoint_adapter() -> None:
    """Backport rank-local checkpoint loading for Qwen3-MoE without EP.

    AutoModel before #3397 assumes every grouped-expert DTensor has an ``ep``
    mesh dimension. With EP=1, FSDP instead shards the expert axis over ``dp``.
    This Qwen-specific staged-copy keeps DCP's rank-local placeholders, validates
    them as one transaction, and copies them back into the original grouped
    storage. Regular full tensors used by rollout refit retain the upstream path.
    """
    from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import (
        Qwen3MoeStateDictAdapter,
    )
    from nemo_automodel.components.moe.state_dict_mixin import (
        MoESplitExpertsStateDictMixin,
    )

    marker = "_nemo_ep_free_checkpoint_adapter"
    methods = (
        Qwen3MoeStateDictAdapter.to_hf,
        Qwen3MoeStateDictAdapter.convert_single_tensor_to_hf,
        Qwen3MoeStateDictAdapter.from_hf,
    )
    installed = tuple(bool(getattr(method, marker, False)) for method in methods)
    if all(installed):
        return
    if any(installed):
        raise RuntimeError(
            "Qwen3-MoE EP-free checkpoint backport is only partially installed"
        )
    if (
        Qwen3MoeStateDictAdapter._convert_single_merged_expert_to_hf_split_experts
        is not MoESplitExpertsStateDictMixin._convert_single_merged_expert_to_hf_split_experts
        or Qwen3MoeStateDictAdapter._from_hf_w_merged_experts
        is not MoESplitExpertsStateDictMixin._from_hf_w_merged_experts
    ):
        raise RuntimeError(
            "Qwen3-MoE checkpoint adapter inheritance no longer matches the "
            "pinned runtime backport"
        )

    guards = (
        (
            Qwen3MoeStateDictAdapter,
            "to_hf",
            "bdc81cf7ac72b1b82e700e9039154809947f2ba809b7552325f45689d9f7f58c",
        ),
        (
            Qwen3MoeStateDictAdapter,
            "convert_single_tensor_to_hf",
            "2da6c393549c61eb1bbb1fe6bbfe4d01b7534ada9c8b702407b84e0d46b501bb",
        ),
        (
            Qwen3MoeStateDictAdapter,
            "from_hf",
            "6917840b85e7b293db44f04a9d23e35bf10ec65baf17859719e3159ed3843548",
        ),
        (
            MoESplitExpertsStateDictMixin,
            "_convert_single_merged_expert_to_hf_split_experts",
            "75de160476e9d0b3399b676cf14ede98093a309069cc846a9fda6519e6f9b508",
        ),
        (
            MoESplitExpertsStateDictMixin,
            "_from_hf_w_merged_experts",
            "1702d2670caecb433cbb3da177747ad5075096bdf8b775cc5f5333b381cab0ff",
        ),
    )
    for owner, name, digest in guards:
        _guard_automodel_source(owner, name, digest)

    original_to_hf = Qwen3MoeStateDictAdapter.to_hf
    original_convert = Qwen3MoeStateDictAdapter.convert_single_tensor_to_hf
    original_from_hf = Qwen3MoeStateDictAdapter.from_hf
    original_split = (
        MoESplitExpertsStateDictMixin._convert_single_merged_expert_to_hf_split_experts
    )
    original_merge = MoESplitExpertsStateDictMixin._from_hf_w_merged_experts

    def split(self, fqn, tensor, **kwargs):
        staged = _stage_ep_free_expert_tensor(
            self,
            fqn,
            tensor,
            quantization=bool(kwargs.get("quantization", False)),
        )
        if staged is not None:
            return staged
        return original_split(self, fqn, tensor, **kwargs)

    def convert(self, fqn, tensor, **kwargs):
        # Per-tensor conversion is the refit API. It normally receives an
        # already gathered Tensor; reset stale checkpoint staging at the start
        # of such an independent conversion.
        if not hasattr(self, "_nemo_ep_free_checkpoint_bulk_conversion"):
            self._nemo_ep_free_checkpoint_stages = {}
        return original_convert(self, fqn, tensor, **kwargs)

    def bulk_to_hf(self, state_dict, *args, **kwargs):
        self._nemo_ep_free_checkpoint_stages = {}
        self._nemo_ep_free_checkpoint_bulk_conversion = True
        try:
            return original_to_hf(self, state_dict, *args, **kwargs)
        except BaseException:
            self._nemo_ep_free_checkpoint_stages = {}
            raise
        finally:
            del self._nemo_ep_free_checkpoint_bulk_conversion

    def merge(self, hf_state_dict, device_mesh=None):
        try:
            return _restore_ep_free_expert_stages(
                self, hf_state_dict, original_merge, device_mesh
            )
        finally:
            self._nemo_ep_free_checkpoint_stages = {}

    def from_hf(self, hf_state_dict, device_mesh=None, **kwargs):
        return original_from_hf(self, hf_state_dict, device_mesh=device_mesh, **kwargs)

    # The Qwen callers remain intact; only their inherited conversion hooks are
    # specialized. Mark all public entry points so partial process-wide state is
    # detectable on repeated installation.
    for function in (bulk_to_hf, convert, from_hf):
        setattr(function, marker, True)
    Qwen3MoeStateDictAdapter._convert_single_merged_expert_to_hf_split_experts = split
    Qwen3MoeStateDictAdapter._from_hf_w_merged_experts = merge
    Qwen3MoeStateDictAdapter.to_hf = bulk_to_hf
    Qwen3MoeStateDictAdapter.convert_single_tensor_to_hf = convert
    Qwen3MoeStateDictAdapter.from_hf = from_hf


class _LogicalAuxLossAutoScaler(torch.autograd.Function):
    """Attach logical router aux loss without an unused saved tensor.

    AutoModel's scaler retains the auxiliary-loss value, although its backward
    only needs the value's metadata. Non-reentrant activation checkpointing can
    stop replay before that trailing save and then rejects the unequal saved-
    tensor count. This equivalent scaler keeps the aux autograd edge and exact
    backward scale without adding a checkpoint save.
    """

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor) -> torch.Tensor:
        ctx.aux_shape = aux_loss.shape
        ctx.aux_dtype = aux_loss.dtype
        ctx.aux_device = aux_loss.device
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from nemo_automodel.components.moe.megatron.moe_utils import (
            MoEAuxLossAutoScaler,
        )

        scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        if scale is None:
            scale = 1.0
        aux_gradient = torch.ones(
            ctx.aux_shape,
            dtype=ctx.aux_dtype,
            device=ctx.aux_device,
        )
        return grad_output, aux_gradient * scale


@dataclass
class _RouterExecution:
    logical_token_weights: torch.Tensor | None
    valid: bool
    committed: bool = False
    pending: dict[nn.Module, tuple[torch.Tensor, torch.Tensor | None]] = field(
        default_factory=dict
    )

    def record(
        self,
        gate: nn.Module,
        expert_load: torch.Tensor,
        aux_loss: torch.Tensor | None,
    ) -> None:
        # Activation checkpointing replays Gate.forward during backward.  Keep
        # this bookkeeping tail identical between the original forward and
        # replay, including the detached snapshot and Python-side write.  The
        # one-shot ``commit`` guard below, rather than a branch in ``record``,
        # prevents replayed state from being accumulated twice.
        snapshot = (expert_load.detach(), _detach_optional(aux_loss))
        if self.valid:
            self.pending[gate] = snapshot

    def commit(self) -> None:
        """Commit forward-only state before any activation-checkpoint replay."""
        if not self.valid:
            self.pending.clear()
            return
        if self.committed:
            self.pending.clear()
            return
        for gate, (expert_load, aux_loss) in self.pending.items():
            gate._last_expert_load = expert_load
            gate._last_aux_loss = aux_loss

            accumulated = getattr(gate, "_nemo_shared_prefix_logical_expert_load", None)
            gate._nemo_shared_prefix_logical_expert_load = (
                expert_load.clone()
                if accumulated is None
                else accumulated + expert_load
            )
            if aux_loss is not None:
                aux_sum = getattr(gate, "_nemo_shared_prefix_aux_loss_sum", None)
                gate._nemo_shared_prefix_aux_loss_sum = (
                    aux_loss.clone() if aux_sum is None else aux_sum + aux_loss
                )
                gate._nemo_shared_prefix_aux_loss_count = (
                    getattr(gate, "_nemo_shared_prefix_aux_loss_count", 0) + 1
                )

        self.committed = True
        self.pending.clear()


def _detach_optional(value: torch.Tensor | None) -> torch.Tensor | None:
    return None if value is None else value.detach()


_ROUTER_EXECUTION_ATTR = "_nemo_shared_prefix_router_execution"
_MISSING_ROUTER_EXECUTION = object()


@contextmanager
def shared_prefix_moe_context(
    model: nn.Module,
    layout: SharedPrefixLayout | None,
    *,
    valid: bool,
) -> Iterator[_RouterExecution]:
    """Keep logical router metadata active through forward and backward.

    Store the execution on each Gate rather than only in a ``ContextVar``.
    Non-reentrant activation checkpoint replay can run from an autograd engine
    context that does not inherit Python context variables; module state remains
    visible to the replay and is safe here because each worker processes and
    backpropagates one microbatch synchronously.
    """
    from nemo_automodel.components.moe.layers import Gate

    execution = _RouterExecution(
        logical_token_weights=(
            None if layout is None else layout.logical_token_weights
        ),
        valid=valid,
    )
    previous: list[tuple[nn.Module, object]] = []
    for module in model.modules():
        if isinstance(module, Gate):
            previous.append(
                (
                    module,
                    getattr(
                        module,
                        _ROUTER_EXECUTION_ATTR,
                        _MISSING_ROUTER_EXECUTION,
                    ),
                )
            )
            setattr(module, _ROUTER_EXECUTION_ATTR, execution)
    try:
        yield execution
    finally:
        for module, prior_execution in previous:
            if prior_execution is _MISSING_ROUTER_EXECUTION:
                delattr(module, _ROUTER_EXECUTION_ATTR)
            else:
                setattr(module, _ROUTER_EXECUTION_ATTR, prior_execution)


def is_custom_qwen3_moe_shared_prefix(model: nn.Module) -> bool:
    """Return whether the native Qwen3-MoE adapter is installed."""
    return bool(getattr(model, _CUSTOM_QWEN3_MOE_FLAG, False))


def enable_custom_qwen3_moe_shared_prefix(model: nn.Module) -> None:
    """Install native-attention and logical-router adapters on one model."""
    from nemo_automodel.components.models.qwen3_moe.layers import (
        Qwen3MoeAttention,
    )
    from nemo_automodel.components.moe.layers import Gate

    if is_custom_qwen3_moe_shared_prefix(model):
        return

    attention_count = 0
    gate_count = 0
    for module in model.modules():
        if isinstance(module, Qwen3MoeAttention):
            module._nemo_shared_prefix_original_forward = module.forward
            module.forward = MethodType(_custom_qwen3_moe_attention_forward, module)
            attention_count += 1
        elif isinstance(module, Gate):
            module._nemo_shared_prefix_original_forward = module.forward
            module.forward = MethodType(_logical_router_forward, module)
            gate_count += 1

    if attention_count == 0 or gate_count == 0:
        raise TypeError(
            "native Qwen3-MoE shared-prefix setup found no attention or Gate modules"
        )

    setattr(model, _CUSTOM_QWEN3_MOE_FLAG, True)


def _custom_qwen3_moe_attention_forward(
    module: nn.Module,
    x: torch.Tensor,
    *,
    freqs_cis: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    shared_prefix_layout: SharedPrefixLayout | None = None,
    **attn_kwargs: Any,
) -> torch.Tensor:
    """Run native Qwen3-MoE projections with the shared-prefix FA2 core."""
    packed_kwargs = attn_kwargs.get("flash_attn_kwargs")
    if shared_prefix_layout is None and packed_kwargs is None:
        return module._nemo_shared_prefix_original_forward(
            x,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            **attn_kwargs,
        )
    if x.ndim != 3 or x.shape[0] != 1:
        raise ValueError(
            "native Qwen3-MoE shared-prefix attention requires [1, tokens, hidden]"
        )
    if attention_mask is not None:
        raise ValueError("native Qwen3-MoE FA2 expects attention_mask=None")

    from nemo_automodel.components.models.gpt_oss.rope_utils import (
        apply_rotary_emb_qk,
    )

    batch_size, sequence_length, _ = x.shape
    query = module.q_proj(x).view(
        batch_size, sequence_length, module.num_heads, module.head_dim
    )
    key = module.k_proj(x).view(
        batch_size, sequence_length, module.num_kv_heads, module.head_dim
    )
    value = module.v_proj(x).view(
        batch_size, sequence_length, module.num_kv_heads, module.head_dim
    )
    query = module.q_norm(query)
    key = module.k_norm(key)
    query, key = apply_rotary_emb_qk(
        query,
        key,
        freqs_cis,
        format="bshd",
        rope_fusion=module.backend.rope_fusion,
        cu_seqlens=None,
        cp_size=1,
        cp_rank=0,
    )

    # Transformers' attention interface uses BHSD at its boundary. Reuse the
    # exact FA2 implementation used by the HF Qwen/Llama path.
    output, _ = shared_prefix_flash_attention_forward(
        module,
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attention_mask=None,
        dropout=0.0,
        scaling=module.head_dim**-0.5,
        shared_prefix_layout=shared_prefix_layout,
        flash_attn_kwargs=packed_kwargs,
        is_causal=True,
    )
    return module.o_proj(output.flatten(2))


def _logical_router_forward(
    gate: nn.Module,
    x: torch.Tensor,
    token_mask: torch.Tensor,
    cp_mesh: Any,
):
    from nemo_automodel.components.moe import layers as moe_layers

    execution = getattr(gate, _ROUTER_EXECUTION_ATTR, None)
    if execution is None or execution.logical_token_weights is None:
        return gate._nemo_shared_prefix_original_forward(x, token_mask, cp_mesh)
    if cp_mesh is not None:
        raise ValueError("shared-prefix logical router statistics require CP=1")
    if gate.bias_update_factor != 0:
        raise ValueError(
            "native Qwen3-MoE shared-prefix routing does not support dynamic bias"
        )

    if not gate.training:
        return gate._nemo_shared_prefix_original_forward(x, token_mask, cp_mesh)

    logical_token_weights = execution.logical_token_weights
    if logical_token_weights.ndim != 1 or logical_token_weights.numel() != x.shape[0]:
        raise ValueError(
            "shared-prefix logical token weights must match the flattened Gate input"
        )
    logical_token_weights = logical_token_weights.to(device=x.device)

    captured_load: torch.Tensor | None = None

    def logical_expert_load(
        module: nn.Module,
        indices: torch.Tensor,
        physical_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        nonlocal captured_load
        valid_weights = logical_token_weights * physical_token_mask.to(
            logical_token_weights.dtype
        )
        expert_load = indices.new_zeros((module.n_experts,))
        contributions = valid_weights.to(indices.dtype).unsqueeze(1).expand_as(indices)
        expert_load.scatter_add_(0, indices.reshape(-1), contributions.reshape(-1))
        captured_load = expert_load
        return expert_load

    def logical_aux_loss(
        module: nn.Module,
        original_scores: torch.Tensor,
        expert_load: torch.Tensor,
        physical_token_mask: torch.Tensor,
        logical_cp_mesh: Any,
    ) -> torch.Tensor:
        if logical_cp_mesh is not None:
            raise ValueError("shared-prefix logical router statistics require CP=1")
        valid_weights = logical_token_weights * physical_token_mask.to(
            logical_token_weights.dtype
        )
        logical_tokens = valid_weights.sum().clamp_min(1)
        expert_scores = (
            original_scores * valid_weights.to(original_scores.dtype).unsqueeze(1)
        ).sum(dim=0)
        load_fraction = expert_load * module.n_experts / (module.topk * logical_tokens)
        return torch.sum(load_fraction * (expert_scores / logical_tokens))

    original_compute_expert_load = gate._compute_expert_load
    original_compute_aux_loss = gate._compute_aux_loss
    aux_loss_coeff = gate.aux_loss_coeff
    track_load_balance = gate._track_load_balance
    original_aux_scaler = moe_layers.MoEAuxLossAutoScaler
    gate._compute_expert_load = MethodType(logical_expert_load, gate)
    gate._compute_aux_loss = MethodType(logical_aux_loss, gate)
    # Force load computation for logical statistics even when the aux
    # coefficient is zero. Dummy microbatches must not attach an aux gradient:
    # AutoModel's aux autoscaler deliberately ignores the downstream zero loss.
    gate._track_load_balance = execution.valid
    if not execution.valid:
        gate.aux_loss_coeff = 0.0
    moe_layers.MoEAuxLossAutoScaler = _LogicalAuxLossAutoScaler
    try:
        weights, indices, aux_loss = gate._nemo_shared_prefix_original_forward(
            x, token_mask, cp_mesh
        )
    finally:
        moe_layers.MoEAuxLossAutoScaler = original_aux_scaler
        gate._compute_expert_load = original_compute_expert_load
        gate._compute_aux_loss = original_compute_aux_loss
        gate.aux_loss_coeff = aux_loss_coeff
        gate._track_load_balance = track_load_balance

    if execution.valid:
        if captured_load is None:
            raise RuntimeError("logical Qwen3-MoE router did not compute expert load")
        execution.record(gate, captured_load, aux_loss)
    return weights, indices, aux_loss


def reset_shared_prefix_moe_statistics(model: nn.Module) -> None:
    """Reset logical statistics at the beginning of one ``Policy.train`` call."""
    if not is_custom_qwen3_moe_shared_prefix(model):
        return
    from nemo_automodel.components.moe.layers import Gate

    for module in model.modules():
        if isinstance(module, Gate):
            module._nemo_shared_prefix_logical_expert_load = None
            module._nemo_shared_prefix_aux_loss_sum = None
            module._nemo_shared_prefix_aux_loss_count = 0


def collect_shared_prefix_moe_statistics(
    model: nn.Module,
    dp_group: torch.distributed.ProcessGroup,
) -> dict[str, float]:
    """Return DP-reduced logical load statistics accumulated over valid MBs."""
    if not is_custom_qwen3_moe_shared_prefix(model):
        return {}
    from nemo_automodel.components.moe.layers import Gate

    cvs: list[float] = []
    min_utilizations: list[float] = []
    max_utilizations: list[float] = []
    dead_fractions: list[float] = []
    diversities: list[float] = []
    aux_sum = 0.0
    aux_count = 0.0
    logical_tokens = 0.0

    for module in model.modules():
        if not isinstance(module, Gate):
            continue

        # Every DP rank must issue the same collectives. A rank can receive only
        # dummy packing microbatches while another rank has valid tokens, so a
        # missing local accumulator participates as zeros instead of skipping.
        local_load = getattr(module, "_nemo_shared_prefix_logical_expert_load", None)
        load = (
            torch.zeros(
                module.n_experts,
                device=module.weight.device,
                dtype=torch.int64,
            )
            if local_load is None
            else local_load.detach().clone().to(dtype=torch.int64)
        )
        torch.distributed.all_reduce(load, group=dp_group)
        load_float = load.float()
        mean = load_float.mean()
        if mean > 0:
            utilization = load_float / mean
            cvs.append((load_float.std(correction=0) / mean).item())
            min_utilizations.append(utilization.min().item())
            max_utilizations.append(utilization.max().item())
            dead_fractions.append((load_float == 0).float().mean().item())
            distribution = load_float / load_float.sum()
            nonzero = distribution[distribution > 0]
            entropy = -(nonzero * nonzero.log()).sum().item()
            diversities.append(math.exp(entropy) / module.n_experts)
            logical_tokens += (load_float.sum() / module.topk).item()

        layer_aux_sum = getattr(module, "_nemo_shared_prefix_aux_loss_sum", None)
        layer_aux_count = getattr(module, "_nemo_shared_prefix_aux_loss_count", 0)
        reduced = torch.stack(
            (
                torch.zeros((), device=module.weight.device, dtype=torch.float32)
                if layer_aux_sum is None
                else layer_aux_sum.detach().float(),
                torch.tensor(
                    float(layer_aux_count),
                    device=module.weight.device,
                    dtype=torch.float32,
                ),
            )
        )
        torch.distributed.all_reduce(reduced, group=dp_group)
        if reduced[1] > 0:
            aux_sum += reduced[0].item()
            aux_count += reduced[1].item()

    if not cvs:
        return {}
    metrics = {
        "logical_load_cv_mean": sum(cvs) / len(cvs),
        "logical_expert_utilization_min": min(min_utilizations),
        "logical_expert_utilization_max": max(max_utilizations),
        "logical_dead_expert_fraction_mean": sum(dead_fractions) / len(dead_fractions),
        "logical_expert_diversity_mean": sum(diversities) / len(diversities),
        "logical_token_layer_events": logical_tokens,
    }
    if aux_count:
        metrics["logical_router_aux_loss_mean"] = aux_sum / aux_count
    return metrics
