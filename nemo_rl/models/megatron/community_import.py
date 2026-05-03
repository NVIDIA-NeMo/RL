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
from pathlib import Path
from typing import Any, Optional

import torch
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
        model_provider.gradient_accumulation_fusion = megatron_config[
            "gradient_accumulation_fusion"
        ]
    model_provider.finalize()

    from megatron.core import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        model_provider.initialize_model_parallel(seed=0)
    else:
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(0)

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


def extract_value_head_from_hf_checkpoint(
    hf_model_path: str,
) -> dict[str, torch.Tensor]:
    """Extract value head weights (score.*) from an HF checkpoint.

    Supports both safetensors and pytorch_model.bin formats.

    Args:
        hf_model_path: Path to a HuggingFace checkpoint directory.

    Returns:
        Dict mapping score key names to tensors, e.g.
        {"score.weight": tensor, "score.bias": tensor}.

    Raises:
        FileNotFoundError: If no checkpoint files are found.
        ValueError: If no score.* keys are found in the checkpoint.
    """
    model_path = Path(hf_model_path)
    score_weights: dict[str, torch.Tensor] = {}

    # Try safetensors first
    try:
        from safetensors import safe_open

        safetensor_files = sorted(model_path.glob("*.safetensors"))
        if safetensor_files:
            for sf_file in safetensor_files:
                with safe_open(str(sf_file), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        if key.startswith("score."):
                            score_weights[key] = f.get_tensor(key)
            if score_weights:
                return score_weights
    except ImportError:
        pass

    # Fall back to pytorch_model.bin
    bin_path = model_path / "pytorch_model.bin"
    if bin_path.exists():
        state_dict = torch.load(str(bin_path), map_location="cpu", weights_only=True)
        for key, tensor in state_dict.items():
            if key.startswith("score."):
                score_weights[key] = tensor
        if score_weights:
            return score_weights

    if not list(model_path.glob("*.safetensors")) and not bin_path.exists():
        raise FileNotFoundError(
            f"No checkpoint files found in {hf_model_path}"
        )

    raise ValueError(
        f"No score.* keys found in checkpoint at {hf_model_path}. "
        "The checkpoint may not contain value head weights."
    )
