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
import json
import logging
import os
from collections.abc import Callable
from glob import glob
from typing import Any, Dict

import huggingface_hub

# Side-effect import: installs the resolver hook that routes FP8-native Mistral 3.5
# configs to Mistral3FP8VLM. Without it, HF's stock FP8Linear path runs and produces
# 0-d weight_scale_inv params that FSDP2 rejects.
import nemo_automodel.components.models.mistral3_vlm  # noqa: F401
from nemo_automodel._transformers import auto_model as automodel_auto_model
from nemo_automodel._transformers.auto_model import (
    NeMoAutoModelForCausalLM,
    NeMoAutoModelForImageTextToText,
    NeMoAutoModelForTextToWaveform,
)
from nemo_automodel.components.checkpoint._backports.hf_storage import (
    _get_key_renaming_mapping,
)
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    _unwrap_ddp_model,
)
from nemo_automodel.components.moe.megatron.token_dispatcher import (
    MoEFlexTokenDispatcher,
)
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
from safetensors import safe_open
from torch import nn

logger = logging.getLogger(__name__)

# Files Automodel's base-model load reads from a resolved Hub snapshot.
BASE_LOAD_SNAPSHOT_FILES = (
    "config.json",
    "model.safetensors",
    "model-*.safetensors",
    "model.safetensors.index.json",
)


def snapshot_download_base_load_files(*args: Any, **kwargs: Any) -> str:
    """Resolve a cached Hub snapshot, requiring only the files a base-model load reads.

    Automodel's ``_build_model`` resolves the pinned snapshot with
    ``snapshot_download(..., local_files_only=True)`` and no file filter, so
    huggingface_hub rejects a snapshot that lacks any file in its cached tree listing,
    including files such as README.md or LICENSE that weight-only downloads (vLLM's,
    for one) never fetch. Missing config or weight files still raise.
    """
    kwargs.setdefault("allow_patterns", list(BASE_LOAD_SNAPSHOT_FILES))
    return huggingface_hub.snapshot_download(*args, **kwargs)


# TODO: remove once Automodel's auto_model.py passes allow_patterns to snapshot_download;
# test_automodel_snapshot_download_patch_still_needed fails when it does.
automodel_auto_model.snapshot_download = snapshot_download_base_load_files


def _checkpoint_tensor_names(model_path: str) -> set[str] | None:
    """Return the tensor names of the HF safetensors checkpoint in a local directory, or None."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            return set(json.load(f)["weight_map"])
    weight_files = glob(os.path.join(model_path, "*.safetensors"))
    if not weight_files:
        return None
    names: set[str] = set()
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt") as f:
            names.update(f.keys())
    return names


def new_task_head_keys(
    model: nn.Module,
    model_path: str,
    key_mapping: dict[str, str] | Callable[[str], str] | None,
) -> list[str]:
    """Return keys of heads the model adds to its checkpoint's architecture that the checkpoint lacks.

    Mirrors NVIDIA-NeMo/Automodel#4710, e.g. ``score.*`` of a ``*ForTokenClassification``
    value model built on a ``*ForCausalLM`` checkpoint.
    """
    unwrapped = _unwrap_ddp_model(model)
    prefix = getattr(unwrapped, "base_model_prefix", "")
    backbone = getattr(unwrapped, prefix, None) if prefix else None
    # A model that is its own backbone (e.g. T5ForConditionalGeneration) has no task heads.
    if not isinstance(backbone, nn.Module) or backbone is unwrapped:
        return []
    architectures = getattr(getattr(unwrapped, "config", None), "architectures", None)
    # FSDP2 swaps in a generated subclass, so match any class in the MRO.
    if not architectures or any(
        cls.__name__ in architectures for cls in type(unwrapped).__mro__
    ):
        return []
    head_keys = [
        key
        for key in map(canonical_parameter_fqn, unwrapped.state_dict())
        if not key.startswith(f"{prefix}.") and not key.endswith("_extra_state")
    ]
    checkpoint_names = _checkpoint_tensor_names(model_path) if head_keys else None
    if checkpoint_names is None:
        return []
    if key_mapping:
        checkpoint_names = {
            _get_key_renaming_mapping(name, key_mapping) for name in checkpoint_names
        }
    return [key for key in head_keys if key not in checkpoint_names]


_automodel_checkpointer_load_model = Checkpointer.load_model


def load_model_keeping_new_task_heads(
    self: Checkpointer,
    model: nn.Module,
    model_path: str,
    is_init_step: bool = False,
    use_checkpoint_id: bool = True,
    key_mapping: dict[str, str] | Callable[[str], str] | None = None,
    allow_checkpoint_key_subset: bool = False,
) -> None:
    """Checkpointer.load_model that keeps new task heads absent from a base checkpoint at their init."""
    head_keys = (
        new_task_head_keys(model, model_path, key_mapping)
        if is_init_step
        and not allow_checkpoint_key_subset
        and isinstance(model, nn.Module)
        and not hasattr(_unwrap_ddp_model(model), "state_dict_adapter")
        and os.path.isdir(model_path)
        else []
    )
    skip_prefixes = self.config.skip_task_head_prefixes_for_base_model
    if head_keys:
        logger.warning(
            "Checkpoint %s has no weights for task-head keys %s; keeping their initialization.",
            model_path,
            head_keys[:10],
        )
        self.config.skip_task_head_prefixes_for_base_model = [
            *(skip_prefixes or []),
            *head_keys,
        ]
    try:
        _automodel_checkpointer_load_model(
            self,
            model,
            model_path,
            is_init_step=is_init_step,
            use_checkpoint_id=use_checkpoint_id,
            key_mapping=key_mapping,
            allow_checkpoint_key_subset=allow_checkpoint_key_subset,
        )
    finally:
        self.config.skip_task_head_prefixes_for_base_model = skip_prefixes


# TODO: remove once NVIDIA-NeMo/Automodel#4710 lands in the pinned Automodel;
# test_automodel_new_task_head_patch_still_needed fails when it does.
Checkpointer.load_model = load_model_keeping_new_task_heads

# Add an entry whenever a model's architecture isn't loadable via
# NeMoAutoModelForCausalLM (e.g. VLMs using ForConditionalGeneration /
# ForImageTextToText). Check MODEL_ARCH_MAPPING in the NeMo automodel registry to see
# which architectures have custom impls:
# https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/_transformers/registry.py#L32-L146
AUTOMODEL_FACTORY: Dict[str, Any] = {
    "qwen2_5_vl": NeMoAutoModelForImageTextToText,
    "qwen2_vl": NeMoAutoModelForImageTextToText,
    "qwen2_5_omni": NeMoAutoModelForTextToWaveform,
    "qwen3_5": NeMoAutoModelForImageTextToText,
    "llava": NeMoAutoModelForImageTextToText,
    "internvl": NeMoAutoModelForImageTextToText,
    "gemma3": NeMoAutoModelForImageTextToText,
    "gemma4": NeMoAutoModelForImageTextToText,
    "gemma4_unified": NeMoAutoModelForImageTextToText,
    "smolvlm": NeMoAutoModelForImageTextToText,
    "mistral3": NeMoAutoModelForImageTextToText,
    "llama4": NeMoAutoModelForImageTextToText,
}


def resolve_model_class(model_name: str) -> Any:
    """Resolve the model class for a model type.

    Args:
        model_name: Model type to resolve.
    """
    return AUTOMODEL_FACTORY.get(model_name.lower(), NeMoAutoModelForCausalLM)


# Per-dispatch tensors that Automodel's flex token-dispatch managers (DeepEP,
# HybridEP, UCCL-EP) keep as attributes. Each dispatch reassigns them, so none
# is needed across steps.
# TODO(https://github.com/NVIDIA-NeMo/Automodel/issues/4401): drop this workaround
# once Automodel releases this state itself.
_MOE_DISPATCH_STATE_ATTRS = (
    "token_probs",
    "token_indices",
    "routing_map",
    "handle",
    "tokens_per_expert",
    "num_permuted_tokens",
    "dispatched_indices",
    "dispatched_probs",
    "dispatched_routing_map",
    "reversed_mapping_for_combine",
)


def release_moe_dispatcher_state(model: nn.Module) -> int:
    """Drop the per-dispatch tensors cached on the model's MoE dispatch managers.

    The managers keep the last dispatch's routing tensors as attributes;
    ``combine`` clears only ``handle`` (and HybridEP's ``num_permuted_tokens``).
    After a backward with activation checkpointing, a manager's last dispatch
    is a recompute, so ``token_probs`` and ``dispatched_probs`` carry a
    ``grad_fn`` that keeps that recompute graph and its saved activations
    alive. With the default manager shared by all layers that is the first MoE
    layer (about 1 GB for Qwen3.5-35B-A3B at 9216 tokens per micro-batch); with
    per-layer managers every layer keeps one. The live blocks pin cached
    allocator segments, so ``torch.cuda.empty_cache()`` cannot return them
    before colocated generation wakes up.

    Args:
        model: Model whose ``token_dispatcher`` attributes are scanned. A
            manager shared across layers is cleared once.

    Returns:
        The number of attributes set to ``None``.
    """
    managers = {}
    for module in model.modules():
        # MoEFlexTokenDispatcher is a plain object held by the experts module,
        # not a submodule, so model.modules() does not yield it.
        dispatcher = getattr(module, "token_dispatcher", None)
        if isinstance(dispatcher, MoEFlexTokenDispatcher):
            manager = getattr(dispatcher, "_comm_manager", None)
            if manager is not None:
                managers[id(manager)] = manager

    cleared = 0
    for manager in managers.values():
        for attr in _MOE_DISPATCH_STATE_ATTRS:
            if getattr(manager, attr, None) is not None:
                setattr(manager, attr, None)
                cleared += 1
    return cleared
