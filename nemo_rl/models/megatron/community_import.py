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
import shutil
import threading
import uuid
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

    Only keys present in the recipe are yielded, so omitting one keeps the
    provider's own default rather than silently forcing False.
    """
    keys = (
        "radio_force_cpe_eval_mode",
        "freeze_vision_model",
        "freeze_vision_projection",
        "freeze_sound_encoder",
        "freeze_sound_projection",
    )
    for key in keys:
        if key in megatron_config:
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


def megatron_conversion_is_complete(pretrained_path: str) -> bool:
    """Whether a completed HF->Megatron conversion exists at `pretrained_path`."""
    return os.path.exists(
        os.path.join(pretrained_path, "iter_0000000", "run_config.yaml")
    )


def publish_megatron_conversion(
    staging_path: str, pretrained_path: str, *, overwrite: bool = False
) -> None:
    """Atomically publish a staged HF->Megatron conversion.

    Args:
        staging_path: Directory the conversion was saved into.
        pretrained_path: Final conversion-cache path.
        overwrite: Replace any existing complete conversion.
    """
    displaced: Optional[str] = None
    for attempt in range(2):
        if not overwrite and megatron_conversion_is_complete(pretrained_path):
            # A concurrent producer sharing the cache won; use its artifact.
            print(
                f"Completed conversion already published at {pretrained_path}; "
                "discarding the staged copy.",
                flush=True,
            )
            threading.Thread(
                target=shutil.rmtree,
                args=(staging_path,),
                kwargs={"ignore_errors": True},
                daemon=True,
            ).start()
            break
        try:
            os.rename(staging_path, pretrained_path)
            break
        except OSError:
            if attempt == 1:
                if not overwrite and megatron_conversion_is_complete(pretrained_path):
                    # A peer published into the slot we freed; use its artifact.
                    threading.Thread(
                        target=shutil.rmtree,
                        args=(staging_path,),
                        kwargs={"ignore_errors": True},
                        daemon=True,
                    ).start()
                    break
                raise
        if not overwrite and megatron_conversion_is_complete(pretrained_path):
            # A complete artifact raced in between the probe and the rename;
            # concede to it on the next pass.
            continue
        # The final path is occupied (stale partial conversion, or a complete one under overwrite);
        # displace it, retry the rename, then delete the displaced copy.
        displaced = f"{pretrained_path}.displaced-{uuid.uuid4().hex[:8]}"
        try:
            os.rename(pretrained_path, displaced)
        except FileNotFoundError:
            # A concurrent publisher displaced it first; retry from the top.
            displaced = None
    if displaced is not None:
        # Doomed uniquely-named copy nothing reads; delete without blocking startup.
        threading.Thread(
            target=shutil.rmtree,
            args=(displaced,),
            kwargs={"ignore_errors": True},
            daemon=True,
        ).start()


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
    *,
    overwrite: bool = False,
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
        overwrite: Publish over an existing complete conversion.
        **config_overrides: Extra keyword arguments forwarded to
            ``AutoBridge.from_hf_pretrained``.
    """
    bridge = AutoBridge.from_hf_pretrained(
        hf_model_name, trust_remote_code=True, **config_overrides
    )

    model_provider = bridge.to_megatron_provider(load_weights=True)

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

    # Stage the save next to the final path, then atomically rename into place
    # so concurrent readers of the shared cache never see a partial checkpoint.
    output_path = os.path.normpath(output_path)
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    dist_active = (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    staging_token = uuid.uuid4().hex[:8]
    if dist_active:
        token_box = [staging_token]
        torch.distributed.broadcast_object_list(token_box, src=0)
        staging_token = token_box[0]
    staging_path = os.path.join(
        output_parent, f".{os.path.basename(output_path)}.staging-{staging_token}"
    )

    with _prefer_nvrx_for_dist_ckpt_save():
        bridge.save_megatron_model(megatron_model, staging_path)

    # Every rank must finish writing before rank 0 publishes the staging dir,
    # and no rank may read output_path before the rename lands.
    if dist_active:
        torch.distributed.barrier()
    if not dist_active or torch.distributed.get_rank() == 0:
        publish_megatron_conversion(staging_path, output_path, overwrite=overwrite)
    if dist_active:
        torch.distributed.barrier()

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
