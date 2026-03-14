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
from typing import Callable, Mapping

import torch
import torch.distributed as dist
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig
from megatron.training.utils import unwrap_model
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
        )

    @property
    def layer_by_index(self) -> dict[int, _EagleLayerLayout]:
        return {layer.layer_index: layer for layer in self.layers}


def _combine_weight_parts(
    *,
    fused_weight: Tensor | None,
    component_weights: tuple[Tensor | None, ...],
    incomplete_error: str,
) -> Tensor | None:
    if fused_weight is not None:
        return fused_weight

    if not any(weight is not None for weight in component_weights):
        return None
    if any(weight is None for weight in component_weights):
        raise RuntimeError(incomplete_error)

    return torch.cat(
        [weight for weight in component_weights if weight is not None],
        dim=0,
    )


@dataclass
class _PendingLayerWeights:
    qkv_weight: Tensor | None = None
    q_weight: Tensor | None = None
    k_weight: Tensor | None = None
    v_weight: Tensor | None = None
    fc1_weight: Tensor | None = None
    gate_weight: Tensor | None = None
    up_weight: Tensor | None = None

    def apply_to(self, mapped_state: StateDict, layer: _EagleLayerLayout) -> None:
        qkv_weight = _combine_weight_parts(
            fused_weight=self.qkv_weight,
            component_weights=(self.q_weight, self.k_weight, self.v_weight),
            incomplete_error=(
                "[draft] Incomplete QKV tensors. Expected q_proj, k_proj, and v_proj."
            ),
        )
        if qkv_weight is not None:
            mapped_state[layer.qkv_weight_key] = qkv_weight

        fc1_weight = _combine_weight_parts(
            fused_weight=self.fc1_weight,
            component_weights=(self.gate_weight, self.up_weight),
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
    q_dim: int,
    kv_dim: int,
) -> tuple[Tensor, Tensor, Tensor]:
    shards = _all_gather_tp_shards(local_fused_weight)
    if len(shards) == 1 and local_fused_weight.shape[0] == q_dim + 2 * kv_dim:
        return local_fused_weight.split([q_dim, kv_dim, kv_dim], dim=0)

    tp_world_size = len(shards)
    if q_dim % tp_world_size != 0 or kv_dim % tp_world_size != 0:
        raise RuntimeError(
            "QKV dimensions are not divisible by the tensor-parallel world size."
        )

    q_shards = []
    k_shards = []
    v_shards = []
    local_q_dim = q_dim // tp_world_size
    local_kv_dim = kv_dim // tp_world_size
    for shard in shards:
        q_local, k_local, v_local = shard.split(
            [local_q_dim, local_kv_dim, local_kv_dim],
            dim=0,
        )
        q_shards.append(q_local)
        k_shards.append(k_local)
        v_shards.append(v_local)

    return (
        torch.cat(q_shards, dim=0).contiguous(),
        torch.cat(k_shards, dim=0).contiguous(),
        torch.cat(v_shards, dim=0).contiguous(),
    )


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
    expected_shape: tuple[int, ...],
    split_axis: int,
) -> Tensor:
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

    for layer in layout.layers:
        pending_weights_by_layer[layer.layer_index].apply_to(mapped_state, layer)

    if not mapped_state:
        raise RuntimeError(
            f"[draft] No Eagle weights were mapped from checkpoint "
            f"'{checkpoint_source}'."
        )

    split_axis_by_parameter = _build_split_axis_by_parameter(layout)
    tp_rank = _get_tp_rank()
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
    )

    return model.load_state_dict(new_state, strict=False)


def _require_state_tensor(
    source_state: Mapping[str, Tensor],
    parameter_name: str,
) -> Tensor:
    if parameter_name not in source_state:
        raise RuntimeError(
            f"[draft] Missing required Eagle parameter '{parameter_name}' while "
            "exporting weights."
        )
    return source_state[parameter_name]


def _export_layer_weights_to_hf(
    *,
    source_state: Mapping[str, Tensor],
    layer: _EagleLayerLayout,
    q_dim: int,
    kv_dim: int,
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
        q_dim=q_dim,
        kv_dim=kv_dim,
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

    q_dim = config.num_attention_heads * config.kv_channels
    kv_dim = config.num_query_groups * config.kv_channels
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
                q_dim=q_dim,
                kv_dim=kv_dim,
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

    return hf_state
