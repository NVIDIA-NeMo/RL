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

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable, Optional

import torch
from megatron.bridge import AutoBridge
from megatron.core.transformer import ModuleSpec

from nemo_rl.models.policy import MegatronConfig


def iter_vlm_config_overrides(
    megatron_config: MegatronConfig,
) -> Iterator[tuple[str, Any]]:
    """Yield explicitly configured Nemotron Omni provider overrides.

    Only non-null keys present in the recipe are yielded, so omitting one (or
    explicitly nulling an inherited key) keeps the provider's own default
    rather than silently forcing a tower control onto an incompatible model.
    """
    keys = (
        "radio_force_cpe_eval_mode",
        "freeze_vision_model",
        "freeze_vision_projection",
        "freeze_sound_encoder",
        "freeze_sound_projection",
    )
    for key in keys:
        if key in megatron_config and megatron_config[key] is not None:
            yield key, megatron_config[key]


def to_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        key = dtype.lower()
        aliases = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        if key in aliases:
            return aliases[key]
    raise ValueError(f"Unknown dtype: {dtype}")


def _register_super_omni_auto_map_alias() -> None:
    """Register the HF remote-code class name used by Super Omni checkpoints.

    Super Omni configs identify the architecture as
    ``NemotronH_Super_Omni_Reasoning_V3`` but map AutoModelForCausalLM to the
    shared implementation class ``NemotronH_Omni_Reasoning_V3``. AutoBridge
    dispatches on the auto-map class name when it is present, so both names
    must resolve to the same bridge.
    """
    from megatron.bridge.models.conversion import model_bridge

    architecture = "NemotronH_Super_Omni_Reasoning_V3"
    auto_map_class = "NemotronH_Omni_Reasoning_V3"
    registry = model_bridge.get_model_bridge._exact_types
    if auto_map_class in registry or architecture not in registry:
        return

    from megatron.bridge.models.conversion.model_bridge import (
        register_bridge_implementation,
    )
    from megatron.bridge.models.nemotron_omni import (
        NemotronOmniBridge,
        NemotronOmniModel,
    )

    register_bridge_implementation(
        source=auto_map_class,
        target=NemotronOmniModel,
        bridge_class=NemotronOmniBridge,
    )


def _is_legacy_nano_v2_omni_bridge(bridge: Any) -> bool:
    """Return whether a V2-labeled Nano checkpoint has the Omni MoE layout."""
    hf_pretrained = getattr(bridge, "hf_pretrained", None)
    hf_config = getattr(hf_pretrained, "config", None) or getattr(
        bridge, "hf_config", None
    )
    architectures = getattr(hf_config, "architectures", None) or []
    llm_config = getattr(hf_config, "llm_config", None)
    return (
        "NemotronH_Nano_VL_V2" in architectures
        and llm_config is not None
        and getattr(llm_config, "n_routed_experts", None) is not None
    )


def resolve_hf_bridge(hf_model_name: str, **config_overrides: Any) -> Any:
    """Resolve an HF bridge, including legacy labels for canonical Omni models.

    Early Nano Omni checkpoints kept the historical ``NemotronH_Nano_VL_V2``
    architecture label even though their language tower is the MoE Omni model.
    The older bridge bundled in prefetched containers sends every checkpoint
    with that label through the dense VL/LLaVA provider, whose FP8 RADIO default
    constructs 16 class tokens. Those checkpoints contain 10 class tokens and
    use the canonical expanded-sequence Omni namespace, so route only the
    MoE-shaped V2 variant through the container's existing Omni bridge.
    """
    _register_super_omni_auto_map_alias()
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_name, trust_remote_code=True, **config_overrides
    )
    if not _is_legacy_nano_v2_omni_bridge(bridge):
        return bridge

    from megatron.bridge.models.conversion.model_bridge import (
        register_bridge_implementation,
    )
    from megatron.bridge.models.nemotron_omni import (
        NemotronOmniBridge,
        NemotronOmniModel,
    )

    # AutoBridge is the public wrapper that owns to_megatron_provider(), weight
    # streaming, and refit. Replace its registry dispatch for this worker
    # process instead of returning the underlying bridge implementation.
    register_bridge_implementation(
        source="NemotronH_Nano_VL_V2",
        target=NemotronOmniModel,
        bridge_class=NemotronOmniBridge,
    )
    bridge._nrl_legacy_nano_v2_omni = True
    return bridge


def _apply_legacy_nano_v2_token_ids(bridge: Any, model_provider: Any) -> None:
    """Preserve the historical V2 image wrapper IDs on the Omni provider."""
    if not getattr(bridge, "_nrl_legacy_nano_v2_omni", False):
        return
    hf_config = bridge.hf_pretrained.config
    model_provider.img_start_token_id = (
        getattr(hf_config, "img_start_token_id", None) or 19
    )
    model_provider.img_end_token_id = getattr(hf_config, "img_end_token_id", None) or 20


@contextmanager
def _prefer_nvrx_for_dist_ckpt_save():
    """Prefer NVRx async strategy for torch_dist save in HF->Megatron import.

    Megatron-LM's torch_dist sync save currently routes through the MCore async
    finalize path, which can fail when write results contain non-picklable
    objects (e.g., code objects) during gather_object.
    """
    try:
        from megatron.core.dist_checkpointing.strategies.torch import (
            TorchDistSaveShardedStrategy,
        )
    except ImportError:
        # If dist-checkpoint strategy cannot be imported, leave behavior unchanged.
        yield
        return

    original_save = TorchDistSaveShardedStrategy.save

    def _save_with_nvrx_fallback(self, sharded_state_dict, checkpoint_dir):
        try:
            async_request = self.async_save(
                sharded_state_dict, checkpoint_dir, async_strategy="nvrx"
            )
            async_request.execute_sync()
            del async_request
        except (ImportError, ModuleNotFoundError):
            # Keep backward compatibility on environments without nvrx.
            original_save(self, sharded_state_dict, checkpoint_dir)

    TorchDistSaveShardedStrategy.save = _save_with_nvrx_fallback
    try:
        yield
    finally:
        TorchDistSaveShardedStrategy.save = original_save


def import_model_from_hf_name(
    hf_model_name: str,
    output_path: str,
    megatron_config: Optional[MegatronConfig] = None,
    model_post_wrap_hook: Optional[Callable] = None,
    transformer_layer_spec: Optional[ModuleSpec | Callable] = None,
    mamba_stack_spec: Optional[ModuleSpec | Callable] = None,
    **config_overrides: Any,
):
    """Import a Hugging Face model into Megatron checkpoint format and save the Megatron checkpoint to the output path.

    Args:
        hf_model_name: Hugging Face model ID or local path (e.g., 'meta-llama/Llama-3.1-8B-Instruct').
        output_path: Directory to write the Megatron checkpoint (e.g., /tmp/megatron_ckpt).
        megatron_config: Optional megatron config with parallelism settings for distributed megatron model import.
        model_post_wrap_hook: Optional callable invoked on each Megatron model
            chunk after it is built (and before DDP wrapping). Forwarded to
            ``provide_distributed_model(post_wrap_hook=...)``.
        transformer_layer_spec: Optional Megatron ``ModuleSpec`` (or callable
            returning one) overriding the default layer spec selected by the
            model provider.
        mamba_stack_spec: Optional Megatron ``ModuleSpec`` (or callable
            returning one) overriding the default Mamba stack spec selected by
            Mamba model providers.
        **config_overrides: Extra keyword arguments forwarded to
            ``AutoBridge.from_hf_pretrained``.
    """
    bridge = resolve_hf_bridge(hf_model_name, **config_overrides)

    model_provider = bridge.to_megatron_provider(load_weights=True)
    _apply_legacy_nano_v2_token_ids(bridge, model_provider)

    if megatron_config is not None:
        for key, value in iter_vlm_config_overrides(megatron_config):
            # Match the import-time behaviour in megatron/setup.py: a key the
            # recipe set explicitly must not be dropped just because this
            # provider lacks the field, or a frozen tower silently trains.
            if not hasattr(model_provider, key):
                raise ValueError(
                    f"megatron_cfg set '{key}' but "
                    f"{type(model_provider).__name__} has no such field; this "
                    "provider does not support that tower control."
                )
            setattr(model_provider, key, value)

    # Keep track of defaults so can restore them to the config after loading the model
    orig_tensor_model_parallel_size = model_provider.tensor_model_parallel_size
    orig_pipeline_model_parallel_size = model_provider.pipeline_model_parallel_size
    orig_context_parallel_size = model_provider.context_parallel_size
    orig_expert_model_parallel_size = model_provider.expert_model_parallel_size
    orig_expert_tensor_parallel_size = model_provider.expert_tensor_parallel_size
    orig_num_layers_in_first_pipeline_stage = (
        model_provider.num_layers_in_first_pipeline_stage
    )
    orig_num_layers_in_last_pipeline_stage = (
        model_provider.num_layers_in_last_pipeline_stage
    )
    orig_pipeline_dtype = model_provider.pipeline_dtype

    if megatron_config is not None:
        model_provider.tensor_model_parallel_size = megatron_config[
            "tensor_model_parallel_size"
        ]
        model_provider.pipeline_model_parallel_size = megatron_config[
            "pipeline_model_parallel_size"
        ]
        model_provider.context_parallel_size = megatron_config["context_parallel_size"]
        model_provider.expert_model_parallel_size = megatron_config[
            "expert_model_parallel_size"
        ]
        model_provider.expert_tensor_parallel_size = megatron_config[
            "expert_tensor_parallel_size"
        ]
        model_provider.num_layers_in_first_pipeline_stage = megatron_config[
            "num_layers_in_first_pipeline_stage"
        ]
        model_provider.num_layers_in_last_pipeline_stage = megatron_config[
            "num_layers_in_last_pipeline_stage"
        ]
        model_provider.pipeline_dtype = to_torch_dtype(
            megatron_config["pipeline_dtype"]
        )
        model_provider.sequence_parallel = megatron_config["sequence_parallel"]
        model_provider.gradient_accumulation_fusion = megatron_config[
            "gradient_accumulation_fusion"
        ]
    if transformer_layer_spec is not None:
        model_provider.transformer_layer_spec = transformer_layer_spec
    if mamba_stack_spec is not None:
        # HybridModelProvider superseded the deprecated Mamba-only field.  A
        # MambaModelProvider normalizes mamba_stack_spec only in __post_init__,
        # so assignments made here must target the canonical field directly.
        if hasattr(model_provider, "hybrid_stack_spec"):
            model_provider.hybrid_stack_spec = mamba_stack_spec
        elif hasattr(model_provider, "mamba_stack_spec"):
            model_provider.mamba_stack_spec = mamba_stack_spec
    model_provider.finalize()

    from megatron.core import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        model_provider.initialize_model_parallel(seed=0)
    else:
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(0)

    megatron_model = model_provider.provide_distributed_model(
        wrap_with_ddp=False,
        post_wrap_hook=model_post_wrap_hook,
    )

    # The above parallelism settings are used to load the model in a distributed manner.
    # However, we do not want to save the parallelism settings to the checkpoint config
    # because they may result in validation errors when loading the checkpoint.
    config = megatron_model[0].config
    config.tensor_model_parallel_size = orig_tensor_model_parallel_size
    config.pipeline_model_parallel_size = orig_pipeline_model_parallel_size
    config.context_parallel_size = orig_context_parallel_size
    config.expert_model_parallel_size = orig_expert_model_parallel_size
    config.expert_tensor_parallel_size = orig_expert_tensor_parallel_size
    config.num_layers_in_first_pipeline_stage = orig_num_layers_in_first_pipeline_stage
    config.num_layers_in_last_pipeline_stage = orig_num_layers_in_last_pipeline_stage
    config.pipeline_dtype = orig_pipeline_dtype

    with _prefer_nvrx_for_dist_ckpt_save():
        bridge.save_megatron_model(megatron_model, output_path)

    # resetting mcore state
    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()

    # The seeding above created the global RNG tracker with default flags.
    # Such a stale tracker will ignore flags that the real model build requests later.
    from megatron.core.tensor_parallel import random as mcore_random

    mcore_random._CUDA_RNG_STATE_TRACKER = None
    mcore_random._CUDA_RNG_STATE_TRACKER_INITIALIZED = False


def export_model_from_megatron(
    hf_model_name: str,
    input_path: str,
    output_path: str,
    hf_tokenizer_path: str,
    overwrite: bool = False,
    hf_overrides: Optional[dict[str, Any]] = {},
    strict: bool = True,
):
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"HF checkpoint already exists at {output_path}. Delete it to run or set overwrite=True."
        )

    try:
        from megatron.bridge.training.model_load_save import (
            temporary_distributed_context,
        )
    except ImportError:
        raise ImportError("megatron.bridge.training is not available.")

    _register_super_omni_auto_map_alias()
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_name, trust_remote_code=True, **hf_overrides
    )

    # Export performs on CPU with proper distributed context
    with temporary_distributed_context(backend="gloo"):
        # Need to set model parallel cuda manual seed for mamba mixer
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(0)

        # Load the Megatron model
        megatron_model = bridge.load_megatron_model(
            input_path, skip_temp_dist_context=True
        )

        # Save in HuggingFace format
        bridge.save_hf_pretrained(megatron_model, output_path, strict=strict)

    # resetting mcore state
    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()
