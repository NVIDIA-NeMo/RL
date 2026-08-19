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

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch
import torch.distributed as dist
from megatron.bridge.training.config import (
    OptimizerConfigOverrideProvider,
    OptimizerConfigOverrideProviderContext,
)
from megatron.core import parallel_state
from megatron.core.optimizer import ParamKey
from megatron.core.optimizer_param_scheduler import ParamGroupOverride
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule, TransformerConfig
from megatron.core.utils import unwrap_model
from torch import Tensor

StateDict = dict[str, Tensor]
CheckpointLoader = Callable[[Path], StateDict]

_CHECKPOINT_CANDIDATE_NAMES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
_HF_SNAPSHOT_ALLOW_PATTERNS = [
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model-*.bin",
    "pytorch_model.bin.index.json",
]
_HF_SNAPSHOT_IGNORE_PATTERNS = ["*.pt", "*.pth", "*.ckpt"]
_MODEL_LAYER_QKV_KEY_PATTERN = re.compile(
    r"^eagle_module\.decoder\.layers\.(\d+)\.self_attention\.linear_qkv\.weight$"
)
_CHECKPOINT_LAYER_KEY_PATTERN = re.compile(r"^layers\.(\d+)\.(.+)$")


@dataclass(frozen=True)
class _EagleLayerLayout:
    layer_index: int
    model_prefix: str
    checkpoint_prefix: str
    hidden_norm_key: str | None
    input_layernorm_key: str | None
    post_attention_layernorm_key: str | None

    @property
    def qkv_weight_key(self) -> str:
        return f"{self.model_prefix}.self_attention.linear_qkv.weight"

    @property
    def proj_weight_key(self) -> str:
        return f"{self.model_prefix}.self_attention.linear_proj.weight"

    @property
    def fc1_weight_key(self) -> str:
        return f"{self.model_prefix}.mlp.linear_fc1.weight"

    @property
    def fc2_weight_key(self) -> str:
        return f"{self.model_prefix}.mlp.linear_fc2.weight"


def _resolve_optional_key(
    model_keys: set[str],
    *candidates: str | None,
) -> str | None:
    for candidate in candidates:
        if candidate is not None and candidate in model_keys:
            return candidate
    return None


@dataclass(frozen=True)
class _EagleModelLayout:
    layers: tuple[_EagleLayerLayout, ...]
    final_norm_key: str | None
    lm_head_key: str | None

    @classmethod
    def detect(cls, model_state: Mapping[str, Tensor]) -> _EagleModelLayout:
        model_keys = set(model_state)
        layer_indices = sorted(
            int(match.group(1))
            for key in model_keys
            if (match := _MODEL_LAYER_QKV_KEY_PATTERN.match(key)) is not None
        )

        if layer_indices:
            layer_prefixes = {
                layer_index: f"eagle_module.decoder.layers.{layer_index}"
                for layer_index in layer_indices
            }
        elif "eagle_module.layer.self_attention.linear_qkv.weight" in model_keys:
            layer_prefixes = {0: "eagle_module.layer"}
        else:
            raise RuntimeError(
                "Unable to detect Eagle layer prefix from model state dict."
            )

        final_norm_key = _resolve_optional_key(
            model_keys,
            "eagle_module.decoder.final_layernorm.weight",
            "eagle_module.norm.weight",
        )
        lm_head_key = _resolve_optional_key(
            model_keys,
            "eagle_module.eagle_output_layer.weight",
            "eagle_module.lm_head.weight",
        )
        global_hidden_norm_key = _resolve_optional_key(
            model_keys,
            "eagle_module.hidden_norm.weight",
            "eagle_module.hnorm.weight",
            "eagle_module.pre_fc_norm_hidden.weight",
            "eagle_module.enorm.weight",
        )

        use_midlayer_alias = len(layer_prefixes) == 1 and 0 in layer_prefixes
        layers = tuple(
            _EagleLayerLayout(
                layer_index=layer_index,
                model_prefix=layer_prefix,
                checkpoint_prefix=(
                    "midlayer" if use_midlayer_alias else f"layers.{layer_index}"
                ),
                hidden_norm_key=_resolve_optional_key(
                    model_keys,
                    f"{layer_prefix}.hidden_norm.weight",
                    f"{layer_prefix}.hnorm.weight",
                    f"{layer_prefix}.pre_fc_norm_hidden.weight",
                    global_hidden_norm_key if layer_index == 0 else None,
                ),
                input_layernorm_key=_resolve_optional_key(
                    model_keys,
                    f"{layer_prefix}.input_layernorm.weight",
                    f"{layer_prefix}.self_attention.linear_qkv.layer_norm_weight",
                ),
                post_attention_layernorm_key=_resolve_optional_key(
                    model_keys,
                    f"{layer_prefix}.pre_mlp_layernorm.weight",
                    f"{layer_prefix}.mlp.linear_fc1.layer_norm_weight",
                ),
            )
            for layer_index, layer_prefix in sorted(layer_prefixes.items())
        )

        return cls(
            layers=layers,
            final_norm_key=final_norm_key,
            lm_head_key=lm_head_key,
        )

    @property
    def layer_by_index(self) -> dict[int, _EagleLayerLayout]:
        return {layer.layer_index: layer for layer in self.layers}


def _qkv_head_dims(config: TransformerConfig) -> tuple[int, int, int]:
    """Return ``(num_attention_heads, num_query_groups, head_dim)`` for the qkv weight."""
    nh = int(config.num_attention_heads)
    ng = int(getattr(config, "num_query_groups", None) or nh)
    hd = int(getattr(config, "kv_channels", None) or int(config.hidden_size) // nh)
    return nh, ng, hd


def _interleave_qkv(
    q: Tensor, k: Tensor, v: Tensor, config: TransformerConfig
) -> Tensor:
    """Reorder HF ``[all_q; all_k; all_v]`` into Megatron's interleaved qkv layout.

    Megatron's ``SelfAttention`` reads ``linear_qkv`` per query group as
    ``[g0: q.. k v | g1: q.. k v | ...]`` (shape ``[num_query_groups,
    (heads_per_group + 2) * head_dim, in]``), so a naive ``cat([q, k, v])`` is
    silently mis-read for GQA (``num_query_groups < num_attention_heads``).
    """
    nh, ng, hd = _qkv_head_dims(config)
    r = nh // ng
    fused = torch.cat(
        [q.reshape(ng, r * hd, -1), k.reshape(ng, hd, -1), v.reshape(ng, hd, -1)],
        dim=1,
    )
    return fused.reshape(-1, q.shape[1]).contiguous()


def _deinterleave_qkv(
    fused: Tensor, config: TransformerConfig
) -> tuple[Tensor, Tensor, Tensor]:
    """Inverse of :func:`_interleave_qkv`: recover HF ``(q, k, v)`` projections."""
    nh, ng, hd = _qkv_head_dims(config)
    r = nh // ng
    g = fused.reshape(ng, (r + 2) * hd, -1)
    return (
        g[:, : r * hd].reshape(nh * hd, -1).contiguous(),
        g[:, r * hd : (r + 1) * hd].reshape(ng * hd, -1).contiguous(),
        g[:, (r + 1) * hd :].reshape(ng * hd, -1).contiguous(),
    )


def _combine_or_shard_weight_parts(
    *,
    parameter_name: str,
    fused_weight: Tensor | None,
    component_weights: tuple[Tensor | None, ...],
    target: Tensor | None,
    tp_rank: int,
    incomplete_error: str,
) -> Tensor | None:
    if fused_weight is not None:
        return fused_weight

    if not any(weight is not None for weight in component_weights):
        return None
    if any(weight is None for weight in component_weights):
        raise RuntimeError(incomplete_error)

    full_weight = torch.cat(
        [weight for weight in component_weights if weight is not None],
        dim=0,
    ).contiguous()
    if target is None:
        return full_weight
    if full_weight.shape == target.shape:
        return full_weight.to(dtype=target.dtype)

    full_dim = full_weight.shape[0]
    local_dim = target.shape[0]
    if local_dim <= 0 or full_dim % local_dim != 0:
        raise RuntimeError(
            f"[draft] Cannot infer TP sharding for '{parameter_name}': "
            f"checkpoint={tuple(full_weight.shape)} model={tuple(target.shape)}"
        )

    inferred_tp = full_dim // local_dim
    if tp_rank >= inferred_tp:
        raise RuntimeError(
            f"[draft] tp_rank={tp_rank} out of range for key '{parameter_name}' "
            f"(inferred_tp={inferred_tp})"
        )

    # Fused Megatron weights expect each local TP shard to preserve component
    # boundaries, e.g. [q_local, k_local, v_local] instead of chunk(full[q, k, v]).
    local_weight_parts = []
    for weight in component_weights:
        assert weight is not None
        if weight.shape[0] % inferred_tp != 0:
            raise RuntimeError(
                f"[draft] Cannot TP-shard fused component for '{parameter_name}': "
                f"component={tuple(weight.shape)} inferred_tp={inferred_tp}"
            )
        local_weight_parts.append(
            torch.chunk(weight, inferred_tp, dim=0)[tp_rank].contiguous()
        )

    local_weight = torch.cat(local_weight_parts, dim=0).contiguous()
    if local_weight.shape != target.shape:
        raise RuntimeError(
            f"[draft] Invalid TP shard shape for '{parameter_name}': "
            f"got={tuple(local_weight.shape)} expected={tuple(target.shape)}"
        )
    return local_weight.to(dtype=target.dtype)


@dataclass
class _PendingLayerWeights:
    qkv_weight: Tensor | None = None
    q_weight: Tensor | None = None
    k_weight: Tensor | None = None
    v_weight: Tensor | None = None
    fc1_weight: Tensor | None = None
    gate_weight: Tensor | None = None
    up_weight: Tensor | None = None

    def apply_to(
        self,
        mapped_state: StateDict,
        layer: _EagleLayerLayout,
        model_state: Mapping[str, Tensor],
        tp_rank: int,
        config: TransformerConfig,
    ) -> None:
        if self.qkv_weight is not None:
            # Pre-fused checkpoint qkv is assumed to already be Megatron-interleaved.
            qkv_weight: Tensor | None = self.qkv_weight
        elif (
            self.q_weight is not None
            and self.k_weight is not None
            and self.v_weight is not None
        ):
            # Separate HF q/k/v are head-major; reorder to Megatron's interleaved
            # layout (_shard_to_local_tp applies the dim-0 TP chunk afterwards).
            qkv_weight = _interleave_qkv(
                self.q_weight, self.k_weight, self.v_weight, config
            )
        elif self.q_weight is None and self.k_weight is None and self.v_weight is None:
            qkv_weight = None
        else:
            raise RuntimeError(
                "[draft] Incomplete QKV tensors. Expected q_proj, k_proj, and v_proj."
            )
        if qkv_weight is not None:
            mapped_state[layer.qkv_weight_key] = qkv_weight

        fc1_weight = _combine_or_shard_weight_parts(
            parameter_name=layer.fc1_weight_key,
            fused_weight=self.fc1_weight,
            component_weights=(self.gate_weight, self.up_weight),
            target=model_state.get(layer.fc1_weight_key),
            tp_rank=tp_rank,
            incomplete_error=(
                "[draft] Incomplete MLP tensors. Expected gate_proj and up_proj."
            ),
        )
        if fc1_weight is not None:
            mapped_state[layer.fc1_weight_key] = fc1_weight


def _get_num_aux_hidden_states(config: TransformerConfig) -> int:
    aux_layer_ids = getattr(config, "eagle_aux_hidden_state_layer_ids", None)
    if aux_layer_ids:
        return len(aux_layer_ids)
    if getattr(config, "use_aux_hidden_state", True):
        return 3
    return 0


def _all_gather_tp_shards(local_weight: Tensor) -> list[Tensor]:
    if (
        not parallel_state.model_parallel_is_initialized()
        or not dist.is_available()
        or not dist.is_initialized()
    ):
        return [local_weight]

    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
    if tp_world_size == 1:
        return [local_weight]

    gathered = [torch.empty_like(local_weight) for _ in range(tp_world_size)]
    dist.all_gather(gathered, local_weight.contiguous(), group=tp_group)
    return gathered


def _gather_tp_qkv_weight(
    local_fused_weight: Tensor,
    config: TransformerConfig,
) -> tuple[Tensor, Tensor, Tensor]:
    """Gather TP shards of the Megatron fused qkv weight and split into HF q/k/v.

    Each TP rank owns a contiguous, group-aligned dim-0 chunk, so all-gathering
    and concatenating in rank order reconstructs the full interleaved weight,
    which is de-interleaved back into HF ``(q, k, v)`` (inverse of the load-time
    :func:`_interleave_qkv`).
    """
    shards = _all_gather_tp_shards(local_fused_weight)
    full_fused = (
        local_fused_weight
        if len(shards) == 1
        else torch.cat(shards, dim=0).contiguous()
    )
    return _deinterleave_qkv(full_fused, config)


def _gather_tp_gate_up_weight(
    local_fused_weight: Tensor,
    ffn_hidden_size: int,
) -> tuple[Tensor, Tensor]:
    shards = _all_gather_tp_shards(local_fused_weight)
    if len(shards) == 1 and local_fused_weight.shape[0] == 2 * ffn_hidden_size:
        return local_fused_weight.split([ffn_hidden_size, ffn_hidden_size], dim=0)

    tp_world_size = len(shards)
    if ffn_hidden_size % tp_world_size != 0:
        raise RuntimeError(
            "ffn_hidden_size is not divisible by the tensor-parallel world size."
        )

    gate_shards = []
    up_shards = []
    local_ffn_hidden_size = ffn_hidden_size // tp_world_size
    for shard in shards:
        gate_local, up_local = shard.split(
            [local_ffn_hidden_size, local_ffn_hidden_size],
            dim=0,
        )
        gate_shards.append(gate_local)
        up_shards.append(up_local)

    return (
        torch.cat(gate_shards, dim=0).contiguous(),
        torch.cat(up_shards, dim=0).contiguous(),
    )


def _gather_tp_weight_if_needed(
    local_weight: Tensor,
    expected_shape_or_tp_group: tuple[int, ...] | dist.ProcessGroup | None,
    split_axis: int | None = None,
) -> Tensor:
    if split_axis is None:
        tp_group = expected_shape_or_tp_group
        if tp_group is None or not dist.is_available() or not dist.is_initialized():
            return local_weight

        tp_world_size = dist.get_world_size(tp_group)
        if tp_world_size <= 1:
            return local_weight

        gathered = [torch.empty_like(local_weight) for _ in range(tp_world_size)]
        dist.all_gather(gathered, local_weight.contiguous(), group=tp_group)
        return torch.cat(gathered, dim=0).contiguous()

    expected_shape = expected_shape_or_tp_group
    if not isinstance(expected_shape, tuple):
        raise TypeError(
            "expected_shape_or_tp_group must be a shape tuple when split_axis is set."
        )
    if tuple(local_weight.shape) == expected_shape:
        return local_weight

    shards = _all_gather_tp_shards(local_weight)
    if len(shards) == 1:
        return local_weight
    return torch.cat(shards, dim=split_axis).contiguous()


def _extract_tensor_state_dict(
    checkpoint_obj: object,
    checkpoint_path: Path,
) -> StateDict:
    if (
        isinstance(checkpoint_obj, dict)
        and "state_dict" in checkpoint_obj
        and isinstance(checkpoint_obj["state_dict"], dict)
    ):
        checkpoint_obj = checkpoint_obj["state_dict"]

    if not isinstance(checkpoint_obj, dict):
        raise RuntimeError(
            f"[draft] Unsupported checkpoint payload in '{checkpoint_path}'. "
            "Expected a state dict or a dict containing `state_dict`."
        )

    state_dict = {
        key: value
        for key, value in checkpoint_obj.items()
        if isinstance(key, str) and isinstance(value, Tensor)
    }
    if not state_dict:
        raise RuntimeError(
            f"[draft] Checkpoint '{checkpoint_path}' did not contain any tensors."
        )
    return state_dict


def _load_safetensors_file(checkpoint_path: Path) -> StateDict:
    from safetensors.torch import load_file as load_safetensors

    return _extract_tensor_state_dict(
        load_safetensors(str(checkpoint_path)),
        checkpoint_path,
    )


def _load_torch_file(checkpoint_path: Path) -> StateDict:
    try:
        checkpoint_obj = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint_obj = torch.load(
            str(checkpoint_path),
            map_location="cpu",
        )

    return _extract_tensor_state_dict(checkpoint_obj, checkpoint_path)


def _merge_checkpoint_shards(
    checkpoint_dir: Path,
    shard_names: list[str],
    shard_loader: CheckpointLoader,
    source_name: str,
) -> StateDict:
    merged_state: StateDict = {}

    for shard_name in shard_names:
        shard_path = checkpoint_dir / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(
                f"[draft] Missing shard '{shard_name}' referenced by '{source_name}'."
            )

        shard_state = shard_loader(shard_path)
        duplicate_keys = set(merged_state).intersection(shard_state)
        if duplicate_keys:
            duplicate_preview = ", ".join(sorted(duplicate_keys)[:5])
            raise RuntimeError(
                f"[draft] Duplicate keys found while merging '{source_name}': "
                f"{duplicate_preview}"
            )
        merged_state.update(shard_state)

    return merged_state


def _load_index_checkpoint(index_path: Path) -> StateDict:
    with index_path.open() as handle:
        try:
            index_data = json.load(handle)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"[draft] Failed to parse checkpoint index '{index_path}'."
            ) from exc

    weight_map = index_data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(
            f"[draft] Checkpoint index '{index_path}' does not contain a valid "
            "`weight_map`."
        )

    shard_names = sorted(
        {
            shard_name
            for shard_name in weight_map.values()
            if isinstance(shard_name, str)
        }
    )
    if not shard_names:
        raise RuntimeError(
            f"[draft] Checkpoint index '{index_path}' does not reference any "
            "weight shards."
        )

    if index_path.name == "model.safetensors.index.json":
        return _merge_checkpoint_shards(
            index_path.parent,
            shard_names,
            _load_safetensors_file,
            index_path.name,
        )
    if index_path.name == "pytorch_model.bin.index.json":
        return _merge_checkpoint_shards(
            index_path.parent,
            shard_names,
            _load_torch_file,
            index_path.name,
        )

    raise RuntimeError(
        f"[draft] Unsupported checkpoint index format '{index_path.name}'."
    )


def _load_checkpoint_file(checkpoint_path: Path) -> StateDict:
    if (
        checkpoint_path.name.startswith("model-")
        and checkpoint_path.suffix == ".safetensors"
    ):
        companion_index = checkpoint_path.parent / "model.safetensors.index.json"
        if companion_index.exists():
            return _load_index_checkpoint(companion_index)

        sibling_shards = sorted(
            shard_path.name
            for shard_path in checkpoint_path.parent.glob("model-*.safetensors")
        )
        if len(sibling_shards) > 1:
            return _merge_checkpoint_shards(
                checkpoint_path.parent,
                sibling_shards,
                _load_safetensors_file,
                str(checkpoint_path.parent),
            )

    if (
        checkpoint_path.name.startswith("pytorch_model-")
        and checkpoint_path.suffix == ".bin"
    ):
        companion_index = checkpoint_path.parent / "pytorch_model.bin.index.json"
        if companion_index.exists():
            return _load_index_checkpoint(companion_index)

        sibling_shards = sorted(
            shard_path.name
            for shard_path in checkpoint_path.parent.glob("pytorch_model-*.bin")
        )
        if len(sibling_shards) > 1:
            return _merge_checkpoint_shards(
                checkpoint_path.parent,
                sibling_shards,
                _load_torch_file,
                str(checkpoint_path.parent),
            )

    if checkpoint_path.suffix == ".safetensors":
        return _load_safetensors_file(checkpoint_path)
    if checkpoint_path.suffix == ".bin":
        return _load_torch_file(checkpoint_path)
    if checkpoint_path.name.endswith(".index.json"):
        return _load_index_checkpoint(checkpoint_path)

    raise RuntimeError(
        f"[draft] Unsupported checkpoint file '{checkpoint_path}'. Expected "
        "a `.safetensors`, `.bin`, or `.index.json` file."
    )


def _load_checkpoint_from_directory(checkpoint_dir: Path) -> StateDict:
    for candidate_name in _CHECKPOINT_CANDIDATE_NAMES:
        candidate_path = checkpoint_dir / candidate_name
        if candidate_path.exists():
            return _load_checkpoint_file(candidate_path)

    safetensor_shards = sorted(
        shard_path.name for shard_path in checkpoint_dir.glob("model-*.safetensors")
    )
    if safetensor_shards:
        return _merge_checkpoint_shards(
            checkpoint_dir,
            safetensor_shards,
            _load_safetensors_file,
            str(checkpoint_dir),
        )

    torch_shards = sorted(
        shard_path.name for shard_path in checkpoint_dir.glob("pytorch_model-*.bin")
    )
    if torch_shards:
        return _merge_checkpoint_shards(
            checkpoint_dir,
            torch_shards,
            _load_torch_file,
            str(checkpoint_dir),
        )

    raise FileNotFoundError(
        f"[draft] No supported checkpoint files were found in '{checkpoint_dir}'."
    )


def _load_checkpoint_state(checkpoint_source: str) -> StateDict:
    source_path = Path(checkpoint_source)
    if source_path.is_file():
        return _load_checkpoint_file(source_path)
    if source_path.is_dir():
        return _load_checkpoint_from_directory(source_path)

    try:
        from huggingface_hub import snapshot_download

        source_path = Path(
            snapshot_download(
                repo_id=checkpoint_source,
                allow_patterns=_HF_SNAPSHOT_ALLOW_PATTERNS,
                ignore_patterns=_HF_SNAPSHOT_IGNORE_PATTERNS,
            )
        )
    except Exception as exc:
        raise FileNotFoundError(
            f"[draft] Could not resolve '{checkpoint_source}' as a local checkpoint "
            "path or Hugging Face repo."
        ) from exc

    return _load_checkpoint_from_directory(source_path)


def _normalize_hf_key(raw_hf_key: str) -> str:
    hf_key = raw_hf_key
    prefixes = ("draft.", "module.", "eagle_module.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if hf_key.startswith(prefix):
                hf_key = hf_key.removeprefix(prefix)
                changed = True
    return hf_key


def _parse_layer_checkpoint_key(hf_key: str) -> tuple[int, str] | None:
    if hf_key.startswith("midlayer."):
        return 0, hf_key.removeprefix("midlayer.")

    match = _CHECKPOINT_LAYER_KEY_PATTERN.match(hf_key)
    if match is None:
        return None

    return int(match.group(1)), match.group(2)


def _get_tp_rank() -> int:
    if parallel_state.model_parallel_is_initialized():
        return parallel_state.get_tensor_model_parallel_rank()
    return 0


def _build_split_axis_by_parameter(layout: _EagleModelLayout) -> dict[str, int]:
    split_axis_by_parameter = {
        "eagle_module.fc.weight": 0,
    }
    if layout.lm_head_key is not None:
        split_axis_by_parameter[layout.lm_head_key] = 0
    for layer in layout.layers:
        split_axis_by_parameter[layer.qkv_weight_key] = 0
        split_axis_by_parameter[layer.proj_weight_key] = 1
        split_axis_by_parameter[layer.fc1_weight_key] = 0
        split_axis_by_parameter[layer.fc2_weight_key] = 1
    return split_axis_by_parameter


def _shard_to_local_tp(
    parameter_name: str,
    tensor: Tensor,
    model_state: Mapping[str, Tensor],
    split_axis_by_parameter: Mapping[str, int],
    tp_rank: int,
) -> Tensor:
    target = model_state.get(parameter_name)
    if target is None:
        return tensor

    if tensor.shape == target.shape:
        return tensor.to(dtype=target.dtype)

    split_axis = split_axis_by_parameter.get(parameter_name)
    if split_axis is None:
        raise RuntimeError(
            f"[draft] Unexpected shape mismatch for non-TP key '{parameter_name}': "
            f"checkpoint={tuple(tensor.shape)} model={tuple(target.shape)}"
        )

    full_dim = tensor.shape[split_axis]
    local_dim = target.shape[split_axis]
    if local_dim <= 0 or full_dim % local_dim != 0:
        raise RuntimeError(
            f"[draft] Cannot infer TP sharding for '{parameter_name}': "
            f"checkpoint={tuple(tensor.shape)} model={tuple(target.shape)}"
        )

    inferred_tp = full_dim // local_dim
    if tp_rank >= inferred_tp:
        raise RuntimeError(
            f"[draft] tp_rank={tp_rank} out of range for key '{parameter_name}' "
            f"(inferred_tp={inferred_tp})"
        )

    local_shard = torch.chunk(tensor, inferred_tp, dim=split_axis)[tp_rank]
    local_shard = local_shard.contiguous()
    if local_shard.shape != target.shape:
        raise RuntimeError(
            f"[draft] Invalid TP shard shape for '{parameter_name}': "
            f"got={tuple(local_shard.shape)} expected={tuple(target.shape)}"
        )
    return local_shard.to(dtype=target.dtype)


def _assign_optional_layer_weight(
    *,
    model_key: str | None,
    hf_weight: Tensor,
    mapped_state: StateDict,
) -> bool:
    if model_key is None:
        return False
    mapped_state[model_key] = hf_weight
    return True


def _map_layer_hf_weight(
    layer_key: str,
    hf_weight: Tensor,
    layer: _EagleLayerLayout,
    mapped_state: StateDict,
    pending_weights: _PendingLayerWeights,
) -> None:
    checkpoint_key = f"{layer.checkpoint_prefix}.{layer_key}"

    if layer_key == "self_attn.qkv_proj.weight":
        pending_weights.qkv_weight = hf_weight
    elif layer_key == "self_attn.q_proj.weight":
        pending_weights.q_weight = hf_weight
    elif layer_key == "self_attn.k_proj.weight":
        pending_weights.k_weight = hf_weight
    elif layer_key == "self_attn.v_proj.weight":
        pending_weights.v_weight = hf_weight
    elif layer_key == "self_attn.o_proj.weight":
        mapped_state[layer.proj_weight_key] = hf_weight
    elif layer_key == "mlp.gate_up_proj.weight":
        pending_weights.fc1_weight = hf_weight
    elif layer_key == "mlp.gate_proj.weight":
        pending_weights.gate_weight = hf_weight
    elif layer_key == "mlp.up_proj.weight":
        pending_weights.up_weight = hf_weight
    elif layer_key == "mlp.down_proj.weight":
        mapped_state[layer.fc2_weight_key] = hf_weight
    elif layer_key == "hidden_norm.weight":
        _assign_optional_layer_weight(
            model_key=layer.hidden_norm_key,
            hf_weight=hf_weight,
            mapped_state=mapped_state,
        )
    elif layer_key == "input_layernorm.weight":
        _assign_optional_layer_weight(
            model_key=layer.input_layernorm_key,
            hf_weight=hf_weight,
            mapped_state=mapped_state,
        )
    elif layer_key == "post_attention_layernorm.weight":
        _assign_optional_layer_weight(
            model_key=layer.post_attention_layernorm_key,
            hf_weight=hf_weight,
            mapped_state=mapped_state,
        )
    else:
        raise RuntimeError(
            f"[draft] Unsupported Eagle checkpoint key '{checkpoint_key}'."
        )


def _map_hf_state_to_eagle_state(
    hf_state_dict: Mapping[str, Tensor],
    model_state: Mapping[str, Tensor],
    layout: _EagleModelLayout,
    checkpoint_source: str,
    config: TransformerConfig,
) -> StateDict:
    mapped_state: StateDict = {}
    pending_weights_by_layer = {
        layer.layer_index: _PendingLayerWeights() for layer in layout.layers
    }
    layers_by_index = layout.layer_by_index

    for raw_hf_key, hf_weight in hf_state_dict.items():
        hf_key = _normalize_hf_key(raw_hf_key)

        if hf_key == "fc.weight":
            mapped_state["eagle_module.fc.weight"] = hf_weight
            continue
        if hf_key == "norm.weight":
            if layout.final_norm_key is None:
                raise RuntimeError(
                    "[draft] Checkpoint contains 'norm.weight' but the Eagle model "
                    "does not expose a matching final norm."
                )
            mapped_state[layout.final_norm_key] = hf_weight
            continue
        if hf_key in {"lm_head.weight", "eagle_output_layer.weight"}:
            if layout.lm_head_key is None:
                raise RuntimeError(
                    "[draft] Checkpoint contains draft LM-head weights but the "
                    "Eagle model does not expose a matching output layer."
                )
            mapped_state[layout.lm_head_key] = hf_weight
            continue
        if hf_key == "d2t":
            d2t_key = "eagle_module.d2t"
            if d2t_key in model_state:
                mapped_state[d2t_key] = hf_weight
            continue

        parsed_layer_key = _parse_layer_checkpoint_key(hf_key)
        if parsed_layer_key is None:
            continue

        layer_index, layer_key = parsed_layer_key
        layer = layers_by_index.get(layer_index)
        if layer is None:
            raise RuntimeError(
                f"[draft] Checkpoint '{checkpoint_source}' contains weights for "
                f"layer {layer_index}, but the Eagle model only exposes layers "
                f"{sorted(layers_by_index)}."
            )

        _map_layer_hf_weight(
            layer_key=layer_key,
            hf_weight=hf_weight,
            layer=layer,
            mapped_state=mapped_state,
            pending_weights=pending_weights_by_layer[layer_index],
        )

    tp_rank = _get_tp_rank()
    for layer in layout.layers:
        pending_weights_by_layer[layer.layer_index].apply_to(
            mapped_state,
            layer,
            model_state=model_state,
            tp_rank=tp_rank,
            config=config,
        )

    if not mapped_state:
        raise RuntimeError(
            f"[draft] No Eagle weights were mapped from checkpoint "
            f"'{checkpoint_source}'."
        )

    split_axis_by_parameter = _build_split_axis_by_parameter(layout)
    for parameter_name in list(mapped_state):
        mapped_state[parameter_name] = _shard_to_local_tp(
            parameter_name=parameter_name,
            tensor=mapped_state[parameter_name],
            model_state=model_state,
            split_axis_by_parameter=split_axis_by_parameter,
            tp_rank=tp_rank,
        )

    return mapped_state


def load_hf_weights_to_eagle(
    model: torch.nn.Module,
    model_name: str,
) -> tuple[list[str], list[str]]:
    """Load HF Eagle weights from a local path or Hub repo into a draft model."""
    if not model_name or not model_name.strip():
        raise ValueError(
            "load_hf_weights_to_eagle requires a non-empty model name or path."
        )

    hf_state_dict = _load_checkpoint_state(model_name)
    model_state = model.state_dict()
    layout = _EagleModelLayout.detect(model_state)
    new_state = _map_hf_state_to_eagle_state(
        hf_state_dict=hf_state_dict,
        model_state=model_state,
        layout=layout,
        checkpoint_source=model_name,
        config=unwrap_model(model).config,
    )

    return model.load_state_dict(new_state, strict=False)


def _require_state_tensor(
    source_state: Mapping[str, Tensor],
    parameter_name: str,
) -> Tensor:
    if parameter_name not in source_state:
        raise RuntimeError(
            f"[draft] Missing required draft parameter '{parameter_name}' while "
            "exporting weights."
        )
    return source_state[parameter_name]


def find_draft_owner_chunk(model: list[MegatronModule]) -> MegatronModule | None:
    """Return the post-process chunk that should own the nested draft model."""
    for model_chunk in reversed(model):
        if getattr(model_chunk, "post_process", False):
            return model_chunk
        language_model = getattr(model_chunk, "language_model", None)
        if language_model is not None and getattr(
            language_model, "post_process", False
        ):
            return model_chunk
    return None


def get_attached_draft_model(model: list[MegatronModule]) -> MegatronModule | None:
    """Find an already attached draft model after Megatron wrapping has been applied."""
    for model_chunk in reversed(model):
        unwrapped_chunk = unwrap_model(model_chunk)
        draft_model = getattr(unwrapped_chunk, "draft_model", None)
        if draft_model is not None:
            return draft_model
    return None


def _export_layer_weights_to_hf(
    *,
    source_state: Mapping[str, Tensor],
    layer: _EagleLayerLayout,
    config: TransformerConfig,
    hidden_size: int,
    ffn_hidden_size: int,
) -> list[tuple[str, Tensor]]:
    layer_prefix = layer.checkpoint_prefix
    hf_state: list[tuple[str, Tensor]] = []

    if layer.hidden_norm_key is not None:
        hf_state.append(
            (
                f"{layer_prefix}.hidden_norm.weight",
                _require_state_tensor(source_state, layer.hidden_norm_key),
            )
        )

    if layer.input_layernorm_key is not None:
        hf_state.append(
            (
                f"{layer_prefix}.input_layernorm.weight",
                _require_state_tensor(source_state, layer.input_layernorm_key),
            )
        )

    q_proj, k_proj, v_proj = _gather_tp_qkv_weight(
        _require_state_tensor(source_state, layer.qkv_weight_key),
        config=config,
    )
    hf_state.append((f"{layer_prefix}.self_attn.q_proj.weight", q_proj))
    hf_state.append((f"{layer_prefix}.self_attn.k_proj.weight", k_proj))
    hf_state.append((f"{layer_prefix}.self_attn.v_proj.weight", v_proj))

    o_proj = _gather_tp_weight_if_needed(
        _require_state_tensor(source_state, layer.proj_weight_key),
        (hidden_size, hidden_size),
        split_axis=1,
    )
    hf_state.append((f"{layer_prefix}.self_attn.o_proj.weight", o_proj))

    if layer.post_attention_layernorm_key is not None:
        hf_state.append(
            (
                f"{layer_prefix}.post_attention_layernorm.weight",
                _require_state_tensor(source_state, layer.post_attention_layernorm_key),
            )
        )

    gate_proj, up_proj = _gather_tp_gate_up_weight(
        _require_state_tensor(source_state, layer.fc1_weight_key),
        ffn_hidden_size=ffn_hidden_size,
    )
    hf_state.append((f"{layer_prefix}.mlp.gate_proj.weight", gate_proj))
    hf_state.append((f"{layer_prefix}.mlp.up_proj.weight", up_proj))

    down_proj = _gather_tp_weight_if_needed(
        _require_state_tensor(source_state, layer.fc2_weight_key),
        (hidden_size, ffn_hidden_size),
        split_axis=1,
    )
    hf_state.append((f"{layer_prefix}.mlp.down_proj.weight", down_proj))

    return hf_state


def export_eagle_weights_to_hf(
    model: torch.nn.Module,
) -> list[tuple[str, Tensor]]:
    """Export the standalone Eagle draft model to HF naming."""
    unwrapped_model = unwrap_model(model)
    source_state = unwrapped_model.state_dict()
    config = unwrapped_model.config
    layout = _EagleModelLayout.detect(source_state)

    ffn_hidden_size = config.ffn_hidden_size
    num_aux_hidden_states = _get_num_aux_hidden_states(config)

    fc_weight = _gather_tp_weight_if_needed(
        _require_state_tensor(source_state, "eagle_module.fc.weight"),
        (
            config.hidden_size,
            config.hidden_size * num_aux_hidden_states,
        ),
        split_axis=0,
    )
    hf_state: list[tuple[str, Tensor]] = [("fc.weight", fc_weight)]

    for layer in layout.layers:
        hf_state.extend(
            _export_layer_weights_to_hf(
                source_state=source_state,
                layer=layer,
                config=config,
                hidden_size=config.hidden_size,
                ffn_hidden_size=ffn_hidden_size,
            )
        )

    if layout.final_norm_key is not None:
        hf_state.append(
            (
                "norm.weight",
                _require_state_tensor(source_state, layout.final_norm_key),
            )
        )
    if layout.lm_head_key is not None:
        hf_state.append(
            (
                "lm_head.weight",
                _gather_tp_weight_if_needed(
                    _require_state_tensor(source_state, layout.lm_head_key),
                    (config.draft_vocab_size, config.hidden_size),
                    split_axis=0,
                ),
            )
        )
    if "eagle_module.d2t" in source_state:
        hf_state.append(("d2t", source_state["eagle_module.d2t"]))

    return hf_state


def get_policy_lm_head_weight(policy_model_chunk: MegatronModule) -> torch.Tensor:
    """Return the local policy LM-head shard for draft initialization."""
    unwrapped_policy_model = unwrap_model(policy_model_chunk)
    if getattr(unwrapped_policy_model, "share_embeddings_and_output_weights", False):
        return unwrapped_policy_model.shared_embedding_or_output_weight()
    return unwrapped_policy_model.output_layer.weight


def _get_draft_output_layer(draft_model: MegatronModule):
    # Block drafts (DFlash/DSpark) expose the LM head as `output_layer`;
    # the Eagle path keeps modelopt's `eagle_module.eagle_output_layer`.
    block_output_layer = getattr(draft_model, "output_layer", None)
    if block_output_layer is not None:
        return block_output_layer
    draft_output_layer = getattr(
        getattr(draft_model, "eagle_module", None), "eagle_output_layer", None
    )
    if draft_output_layer is None:
        raise RuntimeError(
            "[draft] Draft model was configured with has_lm_head=True but does not "
            "expose eagle_output_layer."
        )
    return draft_output_layer


def _get_draft_to_target_token_mapping(
    draft_model: MegatronModule,
    device: torch.device,
) -> torch.Tensor:
    draft_vocab_size = int(draft_model.config.draft_vocab_size)
    reverse_mapping = torch.arange(draft_vocab_size, device=device, dtype=torch.long)
    d2t = getattr(getattr(draft_model, "eagle_module", None), "d2t", None)
    if d2t is not None:
        reverse_mapping = reverse_mapping + d2t.to(device=device, dtype=torch.long)
    return reverse_mapping


def copy_policy_lm_head_to_draft(
    *,
    draft_model: MegatronModule,
    policy_model_chunk: MegatronModule,
) -> None:
    """Initialize the draft LM head from the policy LM head shard."""
    draft_output_layer = _get_draft_output_layer(draft_model)
    tp_group = getattr(draft_output_layer, "tp_group", None) or getattr(
        draft_output_layer, "_tp_group", None
    )
    policy_lm_head_weight = get_policy_lm_head_weight(policy_model_chunk).detach()
    policy_lm_head_weight = _gather_tp_weight_if_needed(policy_lm_head_weight, tp_group)
    draft_token_mapping = _get_draft_to_target_token_mapping(
        draft_model,
        device=policy_lm_head_weight.device,
    )
    if draft_token_mapping.numel() == 0:
        raise RuntimeError("[draft] Draft token mapping is empty.")
    if int(draft_token_mapping.max().item()) >= policy_lm_head_weight.shape[0]:
        raise RuntimeError(
            "[draft] Cannot initialize draft LM head from policy LM head because "
            f"the draft token mapping references policy vocab index {int(draft_token_mapping.max().item())}, "
            f"but the gathered policy LM head only has {policy_lm_head_weight.shape[0]} rows."
        )

    selected_policy_weight = policy_lm_head_weight.index_select(0, draft_token_mapping)
    if tp_group is not None and dist.is_initialized():
        tp_world_size = dist.get_world_size(tp_group)
        if tp_world_size > 1:
            if selected_policy_weight.shape[0] % tp_world_size != 0:
                raise RuntimeError(
                    "[draft] Cannot shard selected policy LM head rows across TP "
                    f"world size {tp_world_size}: rows={selected_policy_weight.shape[0]}."
                )
            tp_rank = dist.get_rank(tp_group)
            selected_policy_weight = torch.chunk(
                selected_policy_weight,
                tp_world_size,
                dim=0,
            )[tp_rank].contiguous()

    if draft_output_layer.weight.shape != selected_policy_weight.shape:
        raise RuntimeError(
            "[draft] Cannot initialize draft LM head from policy LM head because "
            f"their local shard shapes differ after draft-vocab selection: "
            f"draft={tuple(draft_output_layer.weight.shape)} "
            f"policy_selected={tuple(selected_policy_weight.shape)}."
        )

    with torch.no_grad():
        draft_output_layer.weight.copy_(
            selected_policy_weight.to(
                device=draft_output_layer.weight.device,
                dtype=draft_output_layer.weight.dtype,
            )
        )


def get_policy_embedding_row(
    policy_model_chunk: MegatronModule, token_id: int
) -> torch.Tensor:
    """Look up one policy embedding row, TP-correctly, via the module forward.

    Used for the block drafts' mask embedding: the official DFlash contract
    embeds mask slots with the target's FROZEN ``embed_tokens[mask_token_id]``
    row (callers detach; the row is never trained). The module forward handles
    the vocab-parallel lookup + all-reduce.
    """
    unwrapped_policy_model = unwrap_model(policy_model_chunk)
    embedding_owner = getattr(unwrapped_policy_model, "embedding", None)
    if embedding_owner is None:
        language_model = getattr(unwrapped_policy_model, "language_model", None)
        embedding_owner = getattr(language_model, "embedding", None)
    if embedding_owner is None:
        raise RuntimeError(
            "[draft] Block draft training requires the policy embedding on "
            "this rank (pipeline_model_parallel_size must be 1)."
        )
    device = embedding_owner.word_embeddings.weight.device
    return embedding_owner.word_embeddings(
        torch.tensor([int(token_id)], device=device)
    )[0]


DRAFT_GRAD_NORM_GROUP = "draft"


def register_draft_grad_norm_group() -> None:
    """Register the 'draft' grad-norm group with Megatron's optimizer.

    Megatron clips parameters in a registered group separately from the main
    gradient norm (see MegatronOptimizer.clip_grad_norm and the 'mtp'
    precedent in multi_token_prediction.py), so the draft head's large
    early-training gradients do not shrink the policy update through the
    shared global clip. Only called when a draft model is built, so baseline
    (no-draft) runs keep Megatron's stock clipping behavior.
    """
    from megatron.core.optimizer import optimizer as mcore_optimizer

    if DRAFT_GRAD_NORM_GROUP not in mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS:
        mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS = (
            *mcore_optimizer.SEPARATE_GRAD_NORM_GROUPS,
            DRAFT_GRAD_NORM_GROUP,
        )


# Keys appended to mcore's ``param_group_identifier_keys`` so checkpoint
# resume can tell the draft param group apart from the policy group (see
# extend_param_group_identifier_keys_for_resume).
_PARAM_GROUP_IDENTIFIER_EXTRA_KEYS = ("max_lr", "min_lr")


def extend_param_group_identifier_keys_for_resume() -> None:
    """Include per-group LR in mcore's param-group identity used at resume.

    mcore matches checkpointed optimizer param groups to runtime param groups
    by the identity tuple ``param_group_identifier_keys = ('wd_mult',
    'lr_mult', 'is_expert_parallel', 'is_decoupled_lr')``. That tuple keys a
    plain dict in both load paths (``MegatronOptimizer.
    _filter_and_reorder_param_groups`` in megatron/core/optimizer/optimizer.py
    and the inline ``make_needed_groups``/``param_groups_map`` in
    ``DistributedOptimizer.load_state_dict`` in distrib_optimizer.py). The
    draft group built by :class:`DraftOptimizerConfigOverrideProvider` differs
    from the policy group only in ``max_lr``/``min_lr``/``start_wd``/``end_wd``
    — none of which are identity keys — so both groups hash to the same key,
    the last group wins the dict insert, and a resumed run silently trains the
    policy at the draft hyperparameters (observed: policy at 50x LR after
    resume, validation-accuracy collapse).

    Appending ``max_lr``/``min_lr`` makes the identity unique whenever the
    draft LR differs from the policy LR. Only these two are safe to append:
    every group built by ``_get_param_groups`` carries them, while
    ``start_wd``/``end_wd`` exist only on override groups and would raise in
    ``make_needed_groups``. Checkpoints written before this patch carry the
    same fields with the same values, so they remain loadable. Changing the
    draft LR between save and resume now fails loudly at checkpoint load
    (key-not-found) instead of silently keeping the checkpointed LR.
    """
    # Deferred import: keep the mcore optimizer modules a load-time dependency
    # of this patch only, not of every draft.utils import.
    from megatron.core import optimizer as mcore_optimizer_pkg
    from megatron.core.optimizer import distrib_optimizer as mcore_distrib_optimizer
    from megatron.core.optimizer import optimizer as mcore_optimizer

    # distrib_optimizer binds the tuple into its own namespace via
    # ``from ... import``, so each consuming module is patched; the package
    # module only re-exports the name but is kept in sync for readers.
    for module in (mcore_optimizer, mcore_distrib_optimizer, mcore_optimizer_pkg):
        keys = module.param_group_identifier_keys
        missing = tuple(
            key for key in _PARAM_GROUP_IDENTIFIER_EXTRA_KEYS if key not in keys
        )
        if missing:
            module.param_group_identifier_keys = (*keys, *missing)


@dataclass
class DraftOptimizerConfigOverrideProvider(OptimizerConfigOverrideProvider):
    """Give ``draft_model.*`` params their own optimizer param group.

    Extends the standard megatron-bridge overrides with a draft-only group so
    the draft model trains at its own learning rate / weight decay while the
    policy keeps the ``megatron_cfg.optimizer`` settings (FastGRPO trains the
    draft at ~50x the policy LR during online draft learning). The schedule
    *shape* (warmup iters, decay style) stays shared; only the per-group
    max/min LR and weight-decay endpoints differ.

    The draft (max_lr, min_lr) pair must differ from the policy's: it is what
    identifies the draft group when a resumed run maps checkpointed param
    groups back onto runtime groups (see
    :func:`extend_param_group_identifier_keys_for_resume`). A draft override
    that matches the policy pair exactly is rejected here rather than risking
    the groups being conflated at the next resume.
    """

    draft_lr: Optional[float]
    draft_min_lr: Optional[float]
    draft_weight_decay: Optional[float]

    def build_config_overrides(
        self, context: OptimizerConfigOverrideProviderContext
    ) -> dict[ParamKey, ParamGroupOverride] | None:
        overrides = super().build_config_overrides(context) or {}
        draft_override = ParamGroupOverride()
        if self.draft_lr is not None:
            draft_override["max_lr"] = float(self.draft_lr)
            # A draft head generally wants to keep a high LR even when the
            # policy LR decays, so min_lr follows the draft LR unless set.
            draft_override["min_lr"] = float(
                self.draft_min_lr if self.draft_min_lr is not None else self.draft_lr
            )
        elif self.draft_min_lr is not None:
            draft_override["min_lr"] = float(self.draft_min_lr)
        if self.draft_weight_decay is not None:
            draft_override["start_wd"] = float(self.draft_weight_decay)
            draft_override["end_wd"] = float(self.draft_weight_decay)

        base_config = context.optimizer_config
        effective_max_lr = draft_override.get("max_lr", base_config.lr)
        effective_min_lr = draft_override.get("min_lr", base_config.min_lr)
        if (effective_max_lr, effective_min_lr) == (base_config.lr, base_config.min_lr):
            raise ValueError(
                "[draft] policy.draft optimizer overrides must give the draft "
                "group a (lr, min_lr) pair distinct from the policy's "
                f"({base_config.lr}, {base_config.min_lr}); otherwise checkpoint "
                "resume cannot tell the two param groups apart and would apply "
                "the wrong hyperparameters to the policy. Set policy.draft.lr "
                "(and/or min_lr) to a different value, or leave all "
                "policy.draft optimizer keys null to share the policy settings."
            )

        overrides[ParamKey(name="*draft_model.*")] = draft_override
        return overrides


def build_draft_optimizer_override_provider(
    draft_config: Mapping[str, Any],
) -> Optional[DraftOptimizerConfigOverrideProvider]:
    """Build the draft optimizer override provider from ``policy.draft`` config.

    Returns None when the config requests no draft-specific optimizer settings,
    so the caller falls back to megatron-bridge's default provider and the
    optimizer param-group partition is byte-identical to a no-override run.
    """
    draft_lr = draft_config.get("lr")
    draft_min_lr = draft_config.get("min_lr")
    draft_weight_decay = draft_config.get("weight_decay")
    if draft_lr is None and draft_min_lr is None and draft_weight_decay is None:
        return None
    # A draft param group exists from the first step, so the resume-matching
    # patch must be in place before the optimizer state is ever saved/loaded.
    extend_param_group_identifier_keys_for_resume()
    return DraftOptimizerConfigOverrideProvider(
        draft_lr=draft_lr,
        draft_min_lr=draft_min_lr,
        draft_weight_decay=draft_weight_decay,
    )


def _build_block_draft_model(
    model_provider,
    draft_config: dict[str, Any],
    pg_collection: ProcessGroupCollection,
    policy_model_chunk: MegatronModule,
) -> MegatronModule:
    """Build a DFlash/DSpark block draft model.

    The full implementation lives in ``draft/dflash.py``; DSpark subclasses
    it in ``draft/dspark.py``.
    """
    from transformers import AutoConfig

    from nemo_rl.models.megatron.draft.dflash import (
        SUPPORTED_BLOCK_DRAFT_METHODS,
    )
    from nemo_rl.models.megatron.draft.hidden_capture import (
        get_eagle3_aux_hidden_state_layers,
    )

    method = draft_config["method"]
    if method not in SUPPORTED_BLOCK_DRAFT_METHODS:
        raise ValueError(
            f"policy.draft.method must be one of {SUPPORTED_BLOCK_DRAFT_METHODS}, "
            f"got '{method}'."
        )
    if int(model_provider.pipeline_model_parallel_size or 1) != 1:
        raise ValueError(
            "policy.draft.method dflash/dspark requires pipeline_model_parallel_size "
            "== 1 (the policy embedding row for the mask token must live on the "
            "draft owner rank)."
        )
    if int(getattr(model_provider, "context_parallel_size", 1) or 1) != 1:
        raise ValueError(
            "policy.draft.method dflash/dspark requires context_parallel_size == 1."
        )
    if bool(model_provider.sequence_parallel):
        raise ValueError(
            "policy.draft.method dflash/dspark requires sequence_parallel == false."
        )

    model_name = draft_config.get("model_name")
    hf_config = AutoConfig.from_pretrained(model_name).to_dict() if model_name else {}
    hf_drafter_config = dict(hf_config.get("dflash_config") or {})
    # Official DSpark configs keep the drafter fields flat at the top level
    # of config.json (no dflash_config sub-dict); sub-dict keys win.
    for drafter_key in (
        "mask_token_id",
        "target_layer_ids",
        "block_size",
        "markov_rank",
    ):
        if drafter_key not in hf_drafter_config and drafter_key in hf_config:
            hf_drafter_config[drafter_key] = hf_config[drafter_key]

    # Training implements full non-causal block attention with no sliding
    # window / attention sink and (DSpark) no bonus-anchor slot. A checkpoint
    # requesting a different serving-side structure (vLLM resolves these
    # fields in qwen3_dflash._resolve_layer_attention and the DSpark
    # speculator) would silently train a mismatched draft — reject it.
    unsupported_structure: list[str] = []
    if hf_drafter_config.get("causal"):
        unsupported_structure.append("dflash_config.causal=true")
    layer_types = hf_config.get("layer_types")
    if layer_types:
        draft_layer_types = layer_types[: int(hf_config.get("num_hidden_layers", 1))]
        if any(layer_type != "full_attention" for layer_type in draft_layer_types):
            unsupported_structure.append(f"layer_types={draft_layer_types}")
    if hf_drafter_config.get("swa_window_size") or hf_config.get("sliding_window"):
        unsupported_structure.append("sliding-window attention")
    if hf_drafter_config.get("add_swa_attention_sink_bias") or hf_config.get(
        "add_swa_attention_sink_bias"
    ):
        unsupported_structure.append("SWA attention-sink bias")
    if method == "dspark" and hf_config.get("dspark_bonus_anchor"):
        unsupported_structure.append("dspark_bonus_anchor=true")
    if unsupported_structure:
        raise NotImplementedError(
            "[draft] Block draft training only implements full non-causal "
            "block attention; the checkpoint requests unsupported structure: "
            + ", ".join(unsupported_structure)
        )

    mask_token_id = draft_config.get("mask_token_id")
    if mask_token_id is None:
        mask_token_id = hf_drafter_config.get("mask_token_id")
    if mask_token_id is None:
        raise ValueError(
            "policy.draft.mask_token_id is required for dflash/dspark (pick a "
            "reserved, unused-in-data token id; its target embedding row "
            "becomes the mask embedding)."
        )
    hf_mask_token_id = hf_drafter_config.get("mask_token_id")
    if hf_mask_token_id is not None and int(hf_mask_token_id) != int(mask_token_id):
        raise ValueError(
            f"[draft] policy.draft.mask_token_id={mask_token_id} conflicts with "
            f"the checkpoint's dflash_config.mask_token_id={hf_mask_token_id}."
        )

    config = TransformerConfig(
        normalization="RMSNorm",
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        hidden_dropout=0.0,
        attention_softmax_in_fp32=False,
        tensor_model_parallel_size=model_provider.tensor_model_parallel_size,
        pipeline_model_parallel_size=model_provider.pipeline_model_parallel_size,
        expert_tensor_parallel_size=model_provider.expert_tensor_parallel_size,
        sequence_parallel=model_provider.sequence_parallel,
        use_cpu_initialization=model_provider.use_cpu_initialization,
        fp16=model_provider.fp16,
        bf16=model_provider.bf16,
        params_dtype=model_provider.params_dtype,
        pipeline_dtype=model_provider.pipeline_dtype,
        num_layers=(
            hf_config.get("num_hidden_layers", 1)
            if model_name is not None
            else draft_config.get("num_layers") or 1
        ),
        ffn_hidden_size=hf_config.get(
            "intermediate_size", model_provider.ffn_hidden_size
        ),
        num_attention_heads=hf_config.get(
            "num_attention_heads", model_provider.num_attention_heads
        ),
        kv_channels=hf_config.get("head_dim", model_provider.kv_channels),
        num_query_groups=hf_config.get(
            "num_key_value_heads", model_provider.num_query_groups
        ),
        init_method_std=model_provider.init_method_std,
        layernorm_epsilon=hf_config.get(
            "rms_norm_eps", model_provider.layernorm_epsilon
        ),
        add_bias_linear=hf_config.get("attention_bias", False),
        attention_dropout=0.0,
        qk_layernorm=bool(draft_config.get("qk_layernorm", True)),
    )
    config.hidden_size = hf_config.get("hidden_size", model_provider.hidden_size)
    config.vocab_size = hf_config.get("vocab_size", model_provider.vocab_size)
    # v1 trains full-vocab block drafts (no d2t).
    config.draft_vocab_size = config.vocab_size
    config.seq_length = model_provider.seq_length
    config.gradient_accumulation_fusion = False
    config.apply_rope_fusion = False
    config.position_embedding_type = "rope"
    config.rotary_percent = model_provider.rotary_percent
    # transformers >= 5 configs nest the theta under rope_parameters.
    hf_rope_parameters = hf_config.get("rope_parameters") or {}
    config.rotary_base = hf_config.get(
        "rope_theta",
        hf_rope_parameters.get("rope_theta", model_provider.rotary_base),
    )
    # Official z-lab configs carry an explicit "rope_scaling": null — a
    # present-but-null key must read as "no scaling", not crash on None.get.
    hf_rope_scaling = (hf_config.get("rope_scaling") or {}) if hf_config else None
    config.rope_scaling = (
        bool(hf_rope_scaling) if hf_config else model_provider.rope_scaling
    )
    config.rope_scaling_factor = (
        hf_rope_scaling.get("factor", model_provider.rope_scaling_factor)
        if hf_config
        else model_provider.rope_scaling_factor
    )

    if int(mask_token_id) < 0 or int(mask_token_id) >= int(config.vocab_size):
        raise ValueError(
            f"[draft] mask_token_id={mask_token_id} is outside the vocab "
            f"(vocab_size={config.vocab_size})."
        )

    # Block width vs the checkpoint (dspark: gamma slots; dflash: gamma + the
    # bonus anchor slot) — a mismatch would train blocks vLLM never runs.
    gamma = int(draft_config["gamma"])
    ckpt_block_size = hf_drafter_config.get("block_size")
    expected_block_size = gamma if method == "dspark" else gamma + 1
    if ckpt_block_size is not None and int(ckpt_block_size) != expected_block_size:
        raise ValueError(
            f"[draft] policy.draft.gamma={gamma} implies {method} "
            f"block_size={expected_block_size}, but the checkpoint records "
            f"block_size={ckpt_block_size}."
        )

    aux_layer_ids = [
        int(i)
        for i in (
            hf_drafter_config.get("target_layer_ids")
            or draft_config.get("aux_layer_indices")
            or get_eagle3_aux_hidden_state_layers(model_provider.num_layers)
        )
    ]
    num_policy_layers = int(model_provider.num_layers)
    if (
        not aux_layer_ids
        or aux_layer_ids != sorted(set(aux_layer_ids))
        or aux_layer_ids[0] < 0
        or aux_layer_ids[-1] >= num_policy_layers
    ):
        raise ValueError(
            f"[draft] aux layers {aux_layer_ids} (checkpoint target_layer_ids / "
            "policy.draft.aux_layer_indices) must be unique, ascending, and in "
            f"[0, {num_policy_layers}) of the policy — the trainer capture "
            "concatenates taps in ascending layer order, matching vLLM's "
            "target_layer_ids order."
        )
    config.eagle_aux_hidden_state_layer_ids = list(aux_layer_ids)

    shared_kwargs = dict(
        config=config,
        gamma=gamma,
        mask_token_id=int(mask_token_id),
        num_aux_hidden_states=len(aux_layer_ids),
        target_hidden_size=model_provider.hidden_size,
        trunk_chunk=int(draft_config.get("trunk_chunk") or 1024),
    )
    if method == "dflash":
        from nemo_rl.models.megatron.draft.dflash import DFlashDraftModel

        draft_model = DFlashDraftModel(**shared_kwargs)
    else:
        from nemo_rl.models.megatron.draft.dspark import DSparkDraftModel

        # Checkpoint config wins (its weight shapes must match); the recipe
        # value only seeds checkpoint-less builds.
        markov_rank = (
            hf_drafter_config.get("markov_rank")
            or draft_config.get("markov_rank")
            or 64
        )
        draft_model = DSparkDraftModel(
            markov_rank=int(markov_rank),
            **shared_kwargs,
        )
    tp_group = getattr(pg_collection, "tp", None)
    if tp_group is not None:
        for module in draft_model.modules():
            if hasattr(module, "pg_collection"):
                module.pg_collection = pg_collection
            if hasattr(module, "_pg_collection"):
                module._pg_collection = pg_collection
            if hasattr(module, "tp_group"):
                module.tp_group = tp_group
            if hasattr(module, "_tp_group"):
                module._tp_group = tp_group

    if model_name is not None:
        missing_keys, unexpected_keys = load_hf_weights_to_block_draft(
            draft_model, model_name
        )
        # TE _extra_state entries (fp8 scales etc.) legitimately have no
        # checkpoint counterpart; anything else unexplained means the draft
        # would silently train from partially-random weights (or the mapping
        # is out of date) — fail instead of printing.
        missing_keys = [key for key in missing_keys if "_extra_state" not in key]
        unexpected_keys = [key for key in unexpected_keys if "_extra_state" not in key]
        if missing_keys or unexpected_keys:
            raise RuntimeError(
                "[draft] Block draft checkpoint does not match the model: "
                f"missing keys {missing_keys}, unexpected keys {unexpected_keys}."
            )

    return draft_model


def build_draft_model(
    model_provider,
    draft_config: dict[str, Any],
    pg_collection: ProcessGroupCollection,
    policy_model_chunk: MegatronModule,
) -> MegatronModule | None:
    """Build a draft model (Eagle or DFlash/DSpark) before parent DDP wrapping."""
    if not draft_config["enabled"]:
        return None

    if draft_config.get("method", "eagle3") in ("dflash", "dspark"):
        draft_model = _build_block_draft_model(
            model_provider=model_provider,
            draft_config=draft_config,
            pg_collection=pg_collection,
            policy_model_chunk=policy_model_chunk,
        )
        # Same separate grad-norm group + param tagging as the Eagle path.
        register_draft_grad_norm_group()
        for param in draft_model.parameters():
            param.grad_norm_group = DRAFT_GRAD_NORM_GROUP
        return draft_model

    from transformers import AutoConfig

    from nemo_rl.models.megatron.draft.eagle import EagleModel
    from nemo_rl.models.megatron.draft.hidden_capture import (
        get_eagle3_aux_hidden_state_layers,
    )

    model_name = draft_config.get("model_name")
    hf_config = AutoConfig.from_pretrained(model_name).to_dict() if model_name else {}
    draft_num_layers = draft_config.get("num_layers")
    config = TransformerConfig(
        normalization="RMSNorm",
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        hidden_dropout=0.0,
        attention_softmax_in_fp32=False,
        tensor_model_parallel_size=model_provider.tensor_model_parallel_size,
        pipeline_model_parallel_size=model_provider.pipeline_model_parallel_size,
        expert_tensor_parallel_size=model_provider.expert_tensor_parallel_size,
        sequence_parallel=model_provider.sequence_parallel,
        use_cpu_initialization=model_provider.use_cpu_initialization,
        fp16=model_provider.fp16,
        bf16=model_provider.bf16,
        params_dtype=model_provider.params_dtype,
        pipeline_dtype=model_provider.pipeline_dtype,
        num_layers=(
            hf_config.get("num_hidden_layers", 1)
            if model_name is not None
            else draft_num_layers or 1
        ),
        ffn_hidden_size=hf_config.get(
            "intermediate_size", model_provider.ffn_hidden_size
        ),
        num_attention_heads=hf_config.get(
            "num_attention_heads", model_provider.num_attention_heads
        ),
        kv_channels=hf_config.get("head_dim", model_provider.kv_channels),
        num_query_groups=hf_config.get(
            "num_key_value_heads", model_provider.num_query_groups
        ),
        init_method_std=model_provider.init_method_std,
        layernorm_epsilon=hf_config.get(
            "rms_norm_eps", model_provider.layernorm_epsilon
        ),
        add_bias_linear=hf_config.get("mlp_bias", model_provider.add_bias_linear),
        attention_dropout=hf_config.get(
            "attention_dropout", model_provider.attention_dropout
        ),
    )

    config.transformer_layer_spec = None
    config.hidden_size = hf_config.get("hidden_size", model_provider.hidden_size)
    config.vocab_size = hf_config.get("vocab_size", model_provider.vocab_size)
    config.draft_vocab_size = hf_config.get("draft_vocab_size", config.vocab_size)
    config.seq_length = model_provider.seq_length
    config.gradient_accumulation_fusion = False
    config.position_embedding_type = hf_config.get(
        "position_embedding_type", model_provider.position_embedding_type
    )
    config.rotary_percent = model_provider.rotary_percent
    config.rotary_base = hf_config.get("rope_theta", model_provider.rotary_base)
    # Official z-lab configs carry an explicit "rope_scaling": null — a
    # present-but-null key must read as "no scaling", not crash on None.get.
    hf_rope_scaling = (hf_config.get("rope_scaling") or {}) if hf_config else None
    config.rope_scaling = (
        bool(hf_rope_scaling) if hf_config else model_provider.rope_scaling
    )
    config.rope_scaling_factor = (
        hf_rope_scaling.get("factor", model_provider.rope_scaling_factor)
        if hf_config
        else model_provider.rope_scaling_factor
    )

    config.use_input_layernorm_in_first_layer = hf_config.get(
        "use_input_layernorm_in_first_layer", True
    )
    config.use_last_layernorm = hf_config.get("use_last_layernorm", True)
    config.use_aux_hidden_state = hf_config.get("use_aux_hidden_state", True)
    if model_name is not None:
        config.eagle_aux_hidden_state_layer_ids = hf_config.get(
            "eagle_aux_hidden_state_layer_ids", []
        )
    else:
        config.eagle_aux_hidden_state_layer_ids = (
            draft_config.get("aux_layer_indices") or []
        )
    if (
        config.use_aux_hidden_state
        and len(config.eagle_aux_hidden_state_layer_ids) == 0
    ):
        config.eagle_aux_hidden_state_layer_ids = get_eagle3_aux_hidden_state_layers(
            model_provider.num_layers
        )

    config.parallel_draft_step = 1
    config.use_mtp_layernorm = config.parallel_draft_heads_num_layers = None
    config.has_lm_head = True

    ttt_steps = int(draft_config.get("ttt_steps", 1) or 1)
    if ttt_steps > 1:
        # The TTT attention/loss path slices sequences locally and stashes
        # per-pass KV; both require every rank to see the full sequence.
        if int(getattr(model_provider, "context_parallel_size", 1) or 1) != 1:
            raise ValueError(
                "policy.draft.ttt_steps > 1 requires context_parallel_size == 1."
            )
        if bool(model_provider.sequence_parallel):
            raise ValueError(
                "policy.draft.ttt_steps > 1 requires sequence_parallel == false."
            )

    draft_model = EagleModel(
        config=config,
        ttt_steps=ttt_steps,
    )
    tp_group = getattr(pg_collection, "tp", None)
    if tp_group is not None:
        for module in draft_model.modules():
            if hasattr(module, "pg_collection"):
                module.pg_collection = pg_collection
            if hasattr(module, "_pg_collection"):
                module._pg_collection = pg_collection
            if hasattr(module, "tp_group"):
                module.tp_group = tp_group
            if hasattr(module, "_tp_group"):
                module._tp_group = tp_group

    if model_name is not None:
        missing_keys, unexpected_keys = load_hf_weights_to_eagle(
            draft_model, model_name
        )
        draft_lm_head_key = "eagle_module.eagle_output_layer.weight"
        if draft_lm_head_key in missing_keys:
            copy_policy_lm_head_to_draft(
                draft_model=draft_model,
                policy_model_chunk=policy_model_chunk,
            )
            missing_keys = [key for key in missing_keys if key != draft_lm_head_key]
            print(
                "[draft] Draft checkpoint did not contain lm_head.weight; "
                "initialized draft LM head from the policy output layer."
            )
        if missing_keys:
            print(f"[draft] Missing keys after draft load: {missing_keys}")
        if unexpected_keys:
            print(f"[draft] Unexpected keys after draft load: {unexpected_keys}")
    else:
        copy_policy_lm_head_to_draft(
            draft_model=draft_model,
            policy_model_chunk=policy_model_chunk,
        )
        print("[draft] Initialized draft LM head from the policy output layer.")

    # Tag draft params before optimizer construction so
    # copy_optimizer_param_metadata propagates the group to the distributed
    # optimizer's shard/fp32 main params and they are clipped separately.
    register_draft_grad_norm_group()
    for param in draft_model.parameters():
        param.grad_norm_group = DRAFT_GRAD_NORM_GROUP

    return draft_model


# ---------------------------------------------------------------------------
# HF <-> Megatron weight mapping for DFlash/DSpark block draft models.
#
# The checkpoint-side names are exactly what vLLM 0.26's
# ``DFlashQwen3ForCausalLM.load_weights`` / ``Qwen3DSparkForCausalLM.load_weights``
# consume (root-level ``fc.weight`` / ``hidden_norm.weight`` / ``norm.weight`` /
# ``layers.{i}.*`` in Qwen3 naming, plus ``markov_head.markov_w{1,2}.weight``
# for DSpark; NO lm_head — official DFlash contract, vLLM shares the target's)
# — the trainer streams these under a ``draft.`` prefix at refit time and
# ``vllm_backend`` routes them into the drafter. Sibling of the Eagle mapping
# above, sharing its TP-aware primitives; the model-side names are the fixed
# ``DFlashDraftModel`` layout, so no layout detection is needed.
# ---------------------------------------------------------------------------

# Checkpoint keys that are deliberately not represented in the Megatron model.
_SKIPPED_KEY_SUBSTRINGS = (
    # Shared from the target at serving time; training reads the captured
    # target embeddings directly.
    "embed_tokens",
    # Official DFlash contract: the draft owns no LM head (it projects
    # through the target's live head; vLLM shares the target module with a
    # head-less drafter). Tolerated here so pre-contract checkpoints load.
    "lm_head",
    # Official contract: mask slots embed via the target's frozen
    # embed_tokens[mask_token_id] row; a separately-shipped mask embedding
    # (interim checkpoints / trained-mask variants) is ignored.
    "mask_embedding",
    # Training-only vocab map of speculators-format checkpoints.
    "t2d",
)


def _block_layer_model_prefix(layer_index: int) -> str:
    return f"decoder.layers.{layer_index}"


def _block_split_axis_map(model_state: StateDict) -> dict[str, int]:
    split_axis: dict[str, int] = {
        "markov_w2.weight": 0,
    }
    for key in model_state:
        if key.endswith("self_attention.linear_qkv.weight"):
            split_axis[key] = 0
        elif key.endswith("self_attention.linear_proj.weight"):
            split_axis[key] = 1
        elif key.endswith("mlp.linear_fc1.weight"):
            split_axis[key] = 0
        elif key.endswith("mlp.linear_fc2.weight"):
            split_axis[key] = 1
    return split_axis


def load_hf_weights_to_block_draft(
    model: torch.nn.Module,
    model_name: str,
) -> tuple[list[str], list[str]]:
    """Load a DFlash/DSpark HF checkpoint (local path or Hub repo) into the draft.

    Returns ``(missing_keys, unexpected_keys)`` from the (non-strict) state
    dict load, after removing the deliberately unmapped model params.
    """
    if not model_name or not model_name.strip():
        raise ValueError(
            "load_hf_weights_to_block_draft requires a non-empty model name or path."
        )

    hf_state = _load_checkpoint_state(model_name)
    unwrapped = unwrap_model(model)
    model_state = unwrapped.state_dict()
    config = unwrapped.config
    tp_rank = _get_tp_rank()

    mapped_state: StateDict = {}
    pending_by_layer: dict[int, _PendingLayerWeights] = {}
    skipped: list[str] = []

    for raw_key, weight in hf_state.items():
        hf_key = raw_key.removeprefix("model.").removeprefix("draft.")
        if hf_key.startswith("midlayer."):
            hf_key = "layers.0." + hf_key.removeprefix("midlayer.")

        if hf_key == "d2t":
            raise NotImplementedError(
                "Block draft training only supports full-vocab drafts; the "
                f"checkpoint '{model_name}' ships a reduced-vocab d2t map."
            )
        if any(substr in hf_key for substr in _SKIPPED_KEY_SUBSTRINGS):
            skipped.append(hf_key)
            continue

        if hf_key == "fc.weight":
            mapped_state["fc.weight"] = weight
            continue
        if hf_key == "hidden_norm.weight":
            mapped_state["hidden_norm.weight"] = weight
            continue
        if hf_key == "norm.weight":
            mapped_state["decoder.final_layernorm.weight"] = weight
            continue
        if hf_key == "markov_head.markov_w1.weight":
            mapped_state["markov_w1.weight"] = weight
            continue
        if hf_key == "markov_head.markov_w2.weight":
            mapped_state["markov_w2.weight"] = weight
            continue
        if hf_key == "confidence_head.proj.weight":
            mapped_state["confidence_head.weight"] = weight
            continue
        if hf_key == "confidence_head.proj.bias":
            mapped_state["confidence_head.bias"] = weight
            continue

        layer_match = _CHECKPOINT_LAYER_KEY_PATTERN.match(hf_key)
        if layer_match is None:
            skipped.append(hf_key)
            continue
        layer_index = int(layer_match.group(1))
        layer_key = layer_match.group(2)
        prefix = _block_layer_model_prefix(layer_index)
        pending = pending_by_layer.setdefault(layer_index, _PendingLayerWeights())

        if layer_key == "self_attn.q_proj.weight":
            pending.q_weight = weight
        elif layer_key == "self_attn.k_proj.weight":
            pending.k_weight = weight
        elif layer_key == "self_attn.v_proj.weight":
            pending.v_weight = weight
        elif layer_key == "self_attn.qkv_proj.weight":
            pending.qkv_weight = weight
        elif layer_key == "self_attn.o_proj.weight":
            mapped_state[f"{prefix}.self_attention.linear_proj.weight"] = weight
        elif layer_key == "self_attn.q_norm.weight":
            mapped_state[f"{prefix}.self_attention.q_layernorm.weight"] = weight
        elif layer_key == "self_attn.k_norm.weight":
            mapped_state[f"{prefix}.self_attention.k_layernorm.weight"] = weight
        elif layer_key == "input_layernorm.weight":
            mapped_state[f"{prefix}.self_attention.linear_qkv.layer_norm_weight"] = (
                weight
            )
        elif layer_key == "post_attention_layernorm.weight":
            mapped_state[f"{prefix}.mlp.linear_fc1.layer_norm_weight"] = weight
        elif layer_key == "mlp.gate_proj.weight":
            pending.gate_weight = weight
        elif layer_key == "mlp.up_proj.weight":
            pending.up_weight = weight
        elif layer_key == "mlp.gate_up_proj.weight":
            pending.fc1_weight = weight
        elif layer_key == "mlp.down_proj.weight":
            mapped_state[f"{prefix}.mlp.linear_fc2.weight"] = weight
        else:
            raise RuntimeError(
                f"[draft] Unsupported block-draft checkpoint key "
                f"'layers.{layer_index}.{layer_key}'."
            )

    for layer_index, pending in pending_by_layer.items():
        prefix = _block_layer_model_prefix(layer_index)
        qkv_key = f"{prefix}.self_attention.linear_qkv.weight"
        if pending.qkv_weight is not None:
            mapped_state[qkv_key] = pending.qkv_weight
        elif (
            pending.q_weight is not None
            and pending.k_weight is not None
            and pending.v_weight is not None
        ):
            mapped_state[qkv_key] = _interleave_qkv(
                pending.q_weight, pending.k_weight, pending.v_weight, config
            )
        elif not (
            pending.q_weight is None
            and pending.k_weight is None
            and pending.v_weight is None
        ):
            raise RuntimeError(
                "[draft] Incomplete QKV tensors. Expected q_proj, k_proj, and v_proj."
            )
        fc1_key = f"{prefix}.mlp.linear_fc1.weight"
        fc1_weight = _combine_or_shard_weight_parts(
            parameter_name=fc1_key,
            fused_weight=pending.fc1_weight,
            component_weights=(pending.gate_weight, pending.up_weight),
            target=model_state.get(fc1_key),
            tp_rank=tp_rank,
            incomplete_error=(
                "[draft] Incomplete MLP tensors. Expected gate_proj and up_proj."
            ),
        )
        if fc1_weight is not None:
            mapped_state[fc1_key] = fc1_weight

    if not mapped_state:
        raise RuntimeError(
            f"[draft] No block-draft weights were mapped from '{model_name}'."
        )
    if skipped:
        print(f"[draft] Skipped block-draft checkpoint keys: {sorted(skipped)[:10]}")

    split_axis_map = _block_split_axis_map(model_state)
    for parameter_name in list(mapped_state):
        mapped_state[parameter_name] = _shard_to_local_tp(
            parameter_name=parameter_name,
            tensor=mapped_state[parameter_name],
            model_state=model_state,
            split_axis_by_parameter=split_axis_map,
            tp_rank=tp_rank,
        )

    return unwrapped.load_state_dict(mapped_state, strict=False)


def export_block_draft_weights_to_hf(
    model: torch.nn.Module,
) -> list[tuple[str, Tensor]]:
    """Export the block draft model to the vLLM DFlash/DSpark HF naming."""
    unwrapped = unwrap_model(model)
    source_state = unwrapped.state_dict()
    config = unwrapped.config
    hidden_size = int(config.hidden_size)
    ffn_hidden_size = int(config.ffn_hidden_size)

    hf_state: list[tuple[str, Tensor]] = [
        ("fc.weight", _require_state_tensor(source_state, "fc.weight")),
        (
            "hidden_norm.weight",
            _require_state_tensor(source_state, "hidden_norm.weight"),
        ),
        (
            "norm.weight",
            _require_state_tensor(source_state, "decoder.final_layernorm.weight"),
        ),
    ]

    layer_indices = sorted(
        int(match.group(1))
        for key in source_state
        if (match := re.match(r"^decoder\.layers\.(\d+)\.", key)) is not None
    )
    for layer_index in sorted(set(layer_indices)):
        prefix = _block_layer_model_prefix(layer_index)
        hf_prefix = f"layers.{layer_index}"

        q_proj, k_proj, v_proj = _gather_tp_qkv_weight(
            _require_state_tensor(
                source_state, f"{prefix}.self_attention.linear_qkv.weight"
            ),
            config=config,
        )
        hf_state.append((f"{hf_prefix}.self_attn.q_proj.weight", q_proj))
        hf_state.append((f"{hf_prefix}.self_attn.k_proj.weight", k_proj))
        hf_state.append((f"{hf_prefix}.self_attn.v_proj.weight", v_proj))
        hf_state.append(
            (
                f"{hf_prefix}.self_attn.o_proj.weight",
                _gather_tp_weight_if_needed(
                    _require_state_tensor(
                        source_state, f"{prefix}.self_attention.linear_proj.weight"
                    ),
                    (hidden_size, hidden_size),
                    split_axis=1,
                ),
            )
        )
        hf_state.append(
            (
                f"{hf_prefix}.self_attn.q_norm.weight",
                _require_state_tensor(
                    source_state, f"{prefix}.self_attention.q_layernorm.weight"
                ),
            )
        )
        hf_state.append(
            (
                f"{hf_prefix}.self_attn.k_norm.weight",
                _require_state_tensor(
                    source_state, f"{prefix}.self_attention.k_layernorm.weight"
                ),
            )
        )
        hf_state.append(
            (
                f"{hf_prefix}.input_layernorm.weight",
                _require_state_tensor(
                    source_state,
                    f"{prefix}.self_attention.linear_qkv.layer_norm_weight",
                ),
            )
        )
        hf_state.append(
            (
                f"{hf_prefix}.post_attention_layernorm.weight",
                _require_state_tensor(
                    source_state, f"{prefix}.mlp.linear_fc1.layer_norm_weight"
                ),
            )
        )
        gate_proj, up_proj = _gather_tp_gate_up_weight(
            _require_state_tensor(source_state, f"{prefix}.mlp.linear_fc1.weight"),
            ffn_hidden_size=ffn_hidden_size,
        )
        hf_state.append((f"{hf_prefix}.mlp.gate_proj.weight", gate_proj))
        hf_state.append((f"{hf_prefix}.mlp.up_proj.weight", up_proj))
        hf_state.append(
            (
                f"{hf_prefix}.mlp.down_proj.weight",
                _gather_tp_weight_if_needed(
                    _require_state_tensor(
                        source_state, f"{prefix}.mlp.linear_fc2.weight"
                    ),
                    (hidden_size, ffn_hidden_size),
                    split_axis=1,
                ),
            )
        )

    if "markov_w1.weight" in source_state:
        hf_state.append(
            ("markov_head.markov_w1.weight", source_state["markov_w1.weight"])
        )
        markov_rank = source_state["markov_w1.weight"].shape[1]
        hf_state.append(
            (
                "markov_head.markov_w2.weight",
                _gather_tp_weight_if_needed(
                    _require_state_tensor(source_state, "markov_w2.weight"),
                    (int(config.draft_vocab_size), markov_rank),
                    split_axis=0,
                ),
            )
        )

    if "confidence_head.weight" in source_state:
        # vLLM's drafter loader ignores these at refit (it loads the official
        # ckpt's copies at init and skips them); kept for the HF export.
        hf_state.append(
            ("confidence_head.proj.weight", source_state["confidence_head.weight"])
        )
        hf_state.append(
            ("confidence_head.proj.bias", source_state["confidence_head.bias"])
        )

    return hf_state


def export_draft_weights_to_hf(
    model: torch.nn.Module,
) -> list[tuple[str, Tensor]]:
    """Export any supported draft model (Eagle or block draft) to HF naming."""
    from nemo_rl.models.megatron.draft.dflash import DFlashDraftModel

    # DSparkDraftModel subclasses DFlashDraftModel, so this covers both.
    if isinstance(unwrap_model(model), DFlashDraftModel):
        return export_block_draft_weights_to_hf(model)
    return export_eagle_weights_to_hf(model)
