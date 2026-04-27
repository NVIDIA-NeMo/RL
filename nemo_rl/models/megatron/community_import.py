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
from typing import Any, Optional

from megatron.bridge import AutoBridge

from nemo_rl.models.policy import MegatronConfig


def import_model_from_hf_name(
    hf_model_name: str,
    output_path: str,
    megatron_config: Optional[MegatronConfig] = None,
    **config_overrides: Any,
):
    """Import a Hugging Face model into Megatron checkpoint format and save the Megatron checkpoint to the output path.

    Args:
        hf_model_name: Hugging Face model ID or local path (e.g., 'meta-llama/Llama-3.1-8B-Instruct').
        output_path: Directory to write the Megatron checkpoint (e.g., /tmp/megatron_ckpt).
        megatron_config: Optional megatron config with paralellism settings for distributed megatron model import.
    """
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_name, trust_remote_code=True, **config_overrides
    )

    model_provider = bridge.to_megatron_provider(load_weights=True)

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

    radio_config_keys = [
        "radio_force_eval_mode",
        "radio_force_cpe_eval_mode",
        "radio_interpolate_only_cpe",
        "radio_cpe_aspect_ratio_select",
        "radio_disable_cpe",
    ]
    if megatron_config is not None:
        for key in radio_config_keys:
            if key in megatron_config:
                setattr(model_provider, key, megatron_config[key])

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
        model_provider.pipeline_dtype = megatron_config["pipeline_dtype"]
        model_provider.sequence_parallel = megatron_config["sequence_parallel"]
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

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

    bridge.save_megatron_model(megatron_model, output_path)

    # resetting mcore state
    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()


def export_model_from_megatron(
    hf_model_name: str,
    input_path: str,
    output_path: str,
    hf_tokenizer_path: str,
    overwrite: bool = False,
    hf_overrides: Optional[dict[str, Any]] = {},
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

        # Save with frozen tensors included
        _save_hf_with_frozen_tensors(bridge, megatron_model, output_path)

    # resetting mcore state
    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()


def _save_hf_with_frozen_tensors(
    bridge: AutoBridge,
    megatron_model: list,
    output_path: str,
) -> None:
    """Save HF checkpoint, including constant buffers from the base model.

    The Megatron model only contains trainable parameters. Constant buffers
    like vision encoder normalization values (input_conditioner.norm_mean/std)
    are not part of the Megatron state_dict. This function wraps the export
    to include those missing tensors from the base HF model.
    """
    import torch.distributed as dist

    from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource

    # Save artifacts (config, tokenizer, etc.) - only rank 0
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            bridge.hf_pretrained.save_artifacts(output_path)
        dist.barrier()
    else:
        bridge.hf_pretrained.save_artifacts(output_path)

    # Get the base export generator
    base_generator = bridge.export_hf_weights(megatron_model, cpu=True)

    # Wrap it to also yield missing tensors (constant buffers) from the base model
    generator = _generator_with_missing_tensors(
        base_generator,
        bridge.hf_pretrained,
    )

    # Save using the wrapped generator
    state_source = bridge.hf_pretrained.state.source
    if not isinstance(state_source, SafeTensorsStateSource):
        raise ValueError("Expected SafeTensorsStateSource for streaming save")
    state_source.save_generator(generator, output_path, strict=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _generator_with_missing_tensors(base_generator, hf_pretrained):
    """Wrap export generator to include missing tensors from the base HF model."""
    from megatron.bridge.models.hf_pretrained.state import SafeTensorsStateSource

    state_source = hf_pretrained.state.source
    if not isinstance(state_source, SafeTensorsStateSource):
        yield from base_generator
        return

    # Get all expected tensor names from the base model
    all_expected_keys = set(state_source.get_all_keys())

    # Track which tensors we've yielded
    yielded_keys = set()

    # Yield all tensors from the base generator
    for name, tensor in base_generator:
        yielded_keys.add(name)
        yield name, tensor

    # Find and yield missing tensors (constant buffers) from the base model
    missing_keys = all_expected_keys - yielded_keys
    if missing_keys:
        print(f"Adding {len(missing_keys)} missing tensors from base model: {sorted(missing_keys)}")
        missing_tensors = state_source.load_tensors(list(missing_keys))
        for name, tensor in missing_tensors.items():
            yield name, tensor
