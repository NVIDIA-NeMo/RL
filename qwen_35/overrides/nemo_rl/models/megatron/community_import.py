# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import os
from typing import Any, Optional

from megatron.bridge import AutoBridge

from nemo_rl.models.policy import MegatronConfig


def _use_random_init() -> bool:
    return os.environ.get("NRL_RANDOM_INIT", "0") == "1"


def import_model_from_hf_name(
    hf_model_name: str,
    output_path: str,
    megatron_config: Optional[MegatronConfig] = None,
    **config_overrides: Any,
):
    """Import a Hugging Face model into Megatron checkpoint format.

    This file is mounted only for MLPerf functionality smoke runs. With
    NRL_RANDOM_INIT=1, it keeps the upstream config/tokenizer path but asks
    Megatron-Bridge to instantiate random weights instead of loading checkpoint
    shards. Normal runs do not mount this file and keep upstream behavior.
    """

    bridge = AutoBridge.from_hf_pretrained(
        hf_model_name, trust_remote_code=True, **config_overrides
    )

    model_provider = bridge.to_megatron_provider(load_weights=not _use_random_init())

    orig_tensor_model_parallel_size = model_provider.tensor_model_parallel_size
    orig_pipeline_model_parallel_size = model_provider.pipeline_model_parallel_size
    orig_context_parallel_size = model_provider.context_parallel_size
    orig_expert_model_parallel_size = model_provider.expert_model_parallel_size
    orig_expert_tensor_parallel_size = model_provider.expert_tensor_parallel_size
    orig_num_layers_in_first_pipeline_stage = (
        model_provider.num_layers_in_first_pipeline_stage
    )
    orig_num_layers_in_last_pipeline_stage = model_provider.num_layers_in_last_pipeline_stage
    orig_pipeline_dtype = model_provider.pipeline_dtype
    orig_gradient_accumulation_fusion = getattr(
        model_provider, "gradient_accumulation_fusion", None
    )

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
        if (
            "gradient_accumulation_fusion" in megatron_config
            and hasattr(model_provider, "gradient_accumulation_fusion")
        ):
            model_provider.gradient_accumulation_fusion = megatron_config[
                "gradient_accumulation_fusion"
            ]

    model_provider.finalize()
    model_provider.initialize_model_parallel(seed=0)
    megatron_model = model_provider.provide_distributed_model(wrap_with_ddp=False)

    config = megatron_model[0].config
    config.tensor_model_parallel_size = orig_tensor_model_parallel_size
    config.pipeline_model_parallel_size = orig_pipeline_model_parallel_size
    config.context_parallel_size = orig_context_parallel_size
    config.expert_model_parallel_size = orig_expert_model_parallel_size
    config.expert_tensor_parallel_size = orig_expert_tensor_parallel_size
    config.num_layers_in_first_pipeline_stage = orig_num_layers_in_first_pipeline_stage
    config.num_layers_in_last_pipeline_stage = orig_num_layers_in_last_pipeline_stage
    config.pipeline_dtype = orig_pipeline_dtype
    if (
        orig_gradient_accumulation_fusion is not None
        and hasattr(config, "gradient_accumulation_fusion")
    ):
        config.gradient_accumulation_fusion = orig_gradient_accumulation_fusion

    bridge.save_megatron_model(megatron_model, output_path)

    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()


def export_model_from_megatron(
    hf_model_name: str,
    input_path: str,
    output_path: str,
    hf_tokenizer_path: str,
    overwrite: bool = False,
    hf_overrides: Optional[dict[str, Any]] = None,
):
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"HF checkpoint already exists at {output_path}. Delete it to run or set overwrite=True."
        )

    try:
        from megatron.bridge.training.model_load_save import (
            temporary_distributed_context,
        )
    except ImportError as exc:
        raise ImportError("megatron.bridge.training is not available.") from exc

    bridge = AutoBridge.from_hf_pretrained(
        hf_model_name, trust_remote_code=True, **(hf_overrides or {})
    )

    with temporary_distributed_context(backend="gloo"):
        from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

        model_parallel_cuda_manual_seed(0)
        megatron_model = bridge.load_megatron_model(
            input_path, skip_temp_dist_context=True
        )
        bridge.save_hf_pretrained(megatron_model, output_path)

    import megatron.core.rerun_state_machine

    megatron.core.rerun_state_machine.destroy_rerun_state_machine()
