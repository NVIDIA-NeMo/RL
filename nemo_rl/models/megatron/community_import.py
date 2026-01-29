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
from typing import Any, Callable, Optional

import torch
from megatron.bridge import AutoBridge
from megatron.core.transformer import ModuleSpec

from nemo_rl.models.policy import MegatronConfig


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


def import_model_from_hf_name(
    hf_model_name: str,
    output_path: str,
    megatron_config: Optional[MegatronConfig] = None,
    model_post_wrap_hook: Optional[Callable] = None,
    transformer_layer_spec: Optional[ModuleSpec | Callable] = None,
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
        model_provider.pipeline_dtype = to_torch_dtype(
            megatron_config["pipeline_dtype"]
        )
        model_provider.sequence_parallel = megatron_config["sequence_parallel"]
        model_provider.gradient_accumulation_fusion = (
            False  # megatron_config.get("gradient_accumulation_fusion", True)
        )
    if transformer_layer_spec is not None:
        model_provider.transformer_layer_spec = transformer_layer_spec
    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)
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


def mp_overrides_model_from_megatron(
    hf_model_name: str,
    input_path: str,
    output_path: str,
    hf_tokenizer_path: str,
    overwrite: bool = False,
    hf_overrides: Optional[dict[str, Any]] = {},
    strict: bool = True,
    mp_overrides: Optional[dict[str, Any]] = {},
    quant_cfg: Optional[str] = None,
    quant_calib_data: str = "random",
    quant_calib_size: int = 1,
    quant_batch_size: int = 1,
    quant_sequence_length: int = 5,
    save_hf=False,
    restore_modelopt_state=False,
):
    """Convert a Megatron checkpoint to a format that can be loaded with different parallelism.

    This function loads a Megatron checkpoint (which may have been saved with custom tokenizers
    like SFTTokenizer that aren't supported by Megatron Bridge) and re-saves it in a format
    that can be loaded with different TP/PP settings.

    The process:
    1. Load checkpoint with TP=1, PP=1 using the checkpoint's model config
    2. Re-import weights using the HuggingFace bridge (avoids tokenizer issues)
    3. Save as Megatron checkpoint with TP=1, PP=1 (loadable with any parallelism via resharding)

    NOTE: This runs in single-process mode. The output checkpoint is saved with TP=1, PP=1.
    When you load this checkpoint with multiple GPUs and different parallelism (e.g., TP=4),
    Megatron's dist_checkpointing will automatically reshard the weights.

    Args:
        hf_model_name: HuggingFace model name for bridge configuration
        input_path: Path to input Megatron checkpoint
        output_path: Path to save the converted checkpoint
        hf_tokenizer_path: Path to tokenizer for the output checkpoint
        overwrite: Whether to overwrite existing output
        hf_overrides: HuggingFace config overrides
        strict: Whether to use strict mode for weight loading
        mp_overrides: Dict with parallelism overrides (stored in config, actual resharding
            happens when loading with multiple GPUs)
    """
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Checkpoint already exists at {output_path}. Delete it to run or set overwrite=True."
        )

    try:
        from megatron.bridge.training.checkpointing import (
            _load_model_weights_from_checkpoint,
            get_checkpoint_run_config_filename,
            read_run_config,
        )
        from megatron.bridge.training.model_load_save import (
            file_exists,
            temporary_distributed_context,
        )
        from megatron.bridge.training.post_training.checkpointing import (
            has_modelopt_state,
            load_modelopt_state,
        )
        from megatron.bridge.utils.instantiate_utils import instantiate
    except ImportError:
        raise ImportError("megatron.bridge.training is not available.")

    # Create bridge from HuggingFace model (this gives us the correct model structure
    # and avoids issues with unsupported tokenizer types in the checkpoint)
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_name, trust_remote_code=True, **hf_overrides
    )
    print("bridge:", bridge)

    with temporary_distributed_context(backend="gloo"):
        from pathlib import Path

        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(0)

        # Find the actual checkpoint path (handle iter_* subdirectories)
        checkpoint_path = Path(input_path)
        iter_folders = [
            f
            for f in checkpoint_path.iterdir()
            if f.is_dir() and f.name.startswith("iter_")
        ]
        if iter_folders:

            def get_iter_number(folder_name):
                try:
                    return int(folder_name.replace("iter_", ""))
                except ValueError:
                    return -1

            latest_iter = max(iter_folders, key=lambda f: get_iter_number(f.name))
            checkpoint_path = checkpoint_path / latest_iter.name

        # Load model config from checkpoint's run_config.yaml
        run_config_filename = get_checkpoint_run_config_filename(str(checkpoint_path))

        if not file_exists(run_config_filename):
            raise RuntimeError(
                f"Checkpoint at {checkpoint_path} does not have run_config.yaml. "
                "This converter only supports Megatron Bridge checkpoints."
            )

        run_config = read_run_config(run_config_filename)
        model_cfg = instantiate(run_config["model"])

        # Load checkpoint with TP=1, PP=1 (single process) to gather all weights
        print("Loading checkpoint with TP=1, PP=1...")
        model_cfg.tensor_model_parallel_size = 1
        model_cfg.pipeline_model_parallel_size = 1
        model_cfg.context_parallel_size = 1
        model_cfg.expert_model_parallel_size = 1
        model_cfg.expert_tensor_parallel_size = 1
        model_cfg.sequence_parallel = False
        model_cfg.virtual_pipeline_model_parallel_size = None
        has_modelopt = has_modelopt_state(str(checkpoint_path))
        if has_modelopt and restore_modelopt_state:
            #     if hasattr(model_cfg, "mamba_stack_spec"):
            #         pass
            #         # from megatron.core.post_training.modelopt.mamba.model_specs import (
            #         #     get_mamba_stack_modelopt_spec,
            #         # )

            #         # model_cfg.mamba_stack_spec = lambda: get_mamba_stack_modelopt_spec(
            #         #     remap_te_layernorm=False
            #         # )
            #     else:
            #         from nemo_rl.models.policy.workers.quantization.utils import (
            #             quantization_layer_spec,
            #         )

            #         model_cfg.transformer_layer_spec = quantization_layer_spec
            model_cfg.restore_modelopt_state = True
        else:
            model_cfg.restore_modelopt_state = False

        if hasattr(model_cfg, "finalize"):
            model_cfg.finalize()

        model_post_wrap_hook = None
        if quant_cfg is not None:
            from megatron.core.utils import unwrap_model

            from nemo_rl.models.policy.workers.quantization.utils import (
                get_tokenizer,
                quantize_model,
            )

            def _quantize(model):
                print("Quantizing model with config:", quant_cfg)
                print("quant_calib_data:", quant_calib_data)
                print("quant_calib_size:", quant_calib_size)
                print("quant_sequence_length:", quant_sequence_length)
                tokenizer = get_tokenizer(
                    hf_tokenizer_path,
                    max_seq_len=quant_sequence_length,
                    trust_remote_code=True,
                )
                unwrapped_model = unwrap_model(model)[0]
                quantize_model(
                    model=unwrapped_model,
                    quant_cfg=quant_cfg,  # pyrefly: ignore[bad-argument-type]
                    tokenizer=tokenizer,
                    calib_size=quant_calib_size,
                    is_megatron=True,
                    batch_size=quant_batch_size,
                    data=quant_calib_data,
                    force_all_expert_routing=True,
                    max_sample_length=quant_sequence_length,
                )
                return model

            model_post_wrap_hook = _quantize

        megatron_model = model_cfg.provide_distributed_model(
            wrap_with_ddp=False,
            use_cpu_initialization=True,
            post_wrap_hook=model_post_wrap_hook,
        )

        # Load modelopt state if present (e.g., quantization metadata)
        if has_modelopt and restore_modelopt_state:
            # pass
            # print('load_modelopt_state:', load_modelopt_state)
            print(f"Loading modelopt_state from {checkpoint_path}...")
            load_modelopt_state(megatron_model, str(checkpoint_path))
        # print('model after load_modelopt_state:', megatron_model)

        # Load weights from the original checkpoint
        print(f"Loading weights from {checkpoint_path}...")
        _load_model_weights_from_checkpoint(
            str(checkpoint_path), megatron_model, return_state_dict=False
        )

        # Save the checkpoint with HuggingFace bridge (this ensures proper config format)
        # The checkpoint is saved with TP=1, PP=1 weights but can be loaded with any parallelism
        if save_hf:
            print(f"Saving HF checkpoint to {output_path}...")
            bridge.save_hf_pretrained(megatron_model, output_path)
        else:
            print(f"Saving Megatron checkpoint to {output_path}...")
            bridge.save_megatron_model(
                megatron_model, output_path, hf_tokenizer_path=hf_tokenizer_path
            )

        print(f"Done! Checkpoint saved to {output_path}")
        print("This checkpoint can be loaded with any TP/PP configuration.")
        if mp_overrides:
            print(
                f"Note: mp_overrides {mp_overrides} will take effect when loading with multiple GPUs."
            )

    # resetting mcore state
    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()
