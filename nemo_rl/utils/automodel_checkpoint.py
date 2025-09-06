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

"""Checkpoint management utilities for HF models."""

import os
from typing import Any, Optional

import torch

# Apply torch backports for compatibility with torch==2.7.1
from nemo_automodel.components.checkpoint._torch_backports import apply_patches

# Import from nemo-automodel
from nemo_automodel.components.checkpoint.checkpointing import (
    CheckpointingConfig,
    load_model,
    load_optimizer,
    save_model,
    save_optimizer,
)
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from transformers import AutoConfig, AutoTokenizer

# Apply torch backports for compatibility with torch==2.7.1
apply_patches()


def detect_checkpoint_format(weights_path: str) -> tuple[str, bool]:
    """Detect model save format and PEFT status from checkpoint path.

    Args:
        weights_path: Path to the checkpoint file or directory

    Returns:
        tuple: (model_save_format, is_peft) where:
               model_save_format is "torch_save" for DCP or "safetensors" for safetensors
               is_peft is True if PEFT/adapter patterns are detected
    """
    path_lower = weights_path.lower()

    # Detect model save format based on file extension for direct files
    if path_lower.endswith(".safetensors"):
        model_save_format = "safetensors"
    elif path_lower.endswith((".bin", ".pt", ".pth")):
        model_save_format = "torch_save"
    else:
        # For directories, check the structure
        model_save_format = "safetensors"  # Default fallback

        if os.path.isdir(weights_path):
            # Check for nested model/ directory structure
            search_dirs = [weights_path]
            model_dir = os.path.join(weights_path, "model")
            if os.path.isdir(model_dir):
                search_dirs.append(model_dir)

            for search_dir in search_dirs:
                try:
                    files = os.listdir(search_dir)

                    # Check for DCP format (__.distcp files and .metadata)
                    dcp_files = [f for f in files if f.endswith(".distcp")]
                    metadata_files = [
                        f
                        for f in files
                        if f.startswith(".") and "metadata" in f.lower()
                    ]

                    if dcp_files and metadata_files:
                        model_save_format = "torch_save"  # DCP uses torch_save format
                        break

                    # Check for safetensors files
                    safetensors_files = [f for f in files if f.endswith(".safetensors")]
                    if safetensors_files:
                        model_save_format = "safetensors"
                        break

                    # Check for other torch save files
                    torch_files = [
                        f for f in files if f.endswith((".bin", ".pt", ".pth"))
                    ]
                    if torch_files:
                        model_save_format = "torch_save"
                        break

                except (OSError, PermissionError):
                    continue

    # Detect PEFT based on path patterns and file contents
    is_peft = "adapter" in path_lower or "lora" in path_lower or "peft" in path_lower

    # Additional check for adapter files in directories
    if not is_peft and os.path.isdir(weights_path):
        search_dirs = [weights_path]
        model_dir = os.path.join(weights_path, "model")
        if os.path.isdir(model_dir):
            search_dirs.append(model_dir)

        for search_dir in search_dirs:
            try:
                files = os.listdir(search_dir)
                is_peft = any("adapter" in f.lower() for f in files)
                if is_peft:
                    break
            except (OSError, PermissionError):
                continue

    return model_save_format, is_peft


def save_checkpoint(
    model: torch.nn.Module,
    weights_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    optimizer_path: Optional[str] = None,
    tokenizer: Optional[Any] = None,
    tokenizer_path: Optional[str] = None,
    model_save_format: str = "safetensors",
    is_peft: bool = False,
    peft_config: Optional[Any] = None,
    save_consolidated: bool = False,
) -> None:
    """Save a checkpoint of the model and optionally optimizer state.

    Args:
        model: The PyTorch model to save
        weights_path: Path to save model weights
        optimizer: Optional optimizer to save
        scheduler: Optional scheduler to save
        optimizer_path: Path to save optimizer state (required if optimizer provided)
        tokenizer: Optional tokenizer to save
        tokenizer_path: Path to save tokenizer state (required if tokenizer provided)
        model_save_format: Format for saving model ("torch_save" or "safetensors")
        is_peft: Whether the model uses PEFT
        peft_config: PEFT configuration if is_peft is True
    """
    # Create checkpoint config
    checkpoint_config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=os.path.dirname(weights_path),
        model_save_format=model_save_format,
        model_cache_dir="",
        model_repo_id="",
        save_consolidated=save_consolidated,
        is_peft=is_peft,
    )

    # Save model using nemo-automodel API
    save_model(
        model=model,
        weights_path=weights_path,
        checkpoint_config=checkpoint_config,
        peft_config=peft_config,
        tokenizer=tokenizer if tokenizer_path is None else None,
    )

    # Save optimizer if provided
    if optimizer is not None:
        if optimizer_path is None:
            raise ValueError(
                "optimizer_path must be provided when saving optimizer state"
            )
        save_optimizer(
            optimizer=optimizer,
            model=model,
            weights_path=optimizer_path,
            scheduler=scheduler,
        )

    # Save tokenizer separately if tokenizer_path provided
    if tokenizer is not None and tokenizer_path is not None:
        print(f"Saving tokenizer (or processor) to {tokenizer_path}")
        tokenizer.save_pretrained(tokenizer_path)


def load_checkpoint(
    model: torch.nn.Module,
    weights_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    optimizer_path: Optional[str] = None,
    model_save_format: str = "safetensors",
    is_peft: bool = False,
) -> None:
    """Load a model weights and optionally optimizer state.

    Args:
        model: The PyTorch model whose weights to update
        weights_path: Path to load model weights from
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        optimizer_path: Path to load optimizer state from (required if optimizer provided)
        model_save_format: Format model was saved in ("torch_save" or "safetensors")
        is_peft: Whether the model uses PEFT
    """
    print(f"Loading weights from {weights_path}")

    # Auto-detect format and PEFT status if not provided
    if model_save_format is None or is_peft is None:
        detected_format, detected_peft = detect_checkpoint_format(weights_path)
        if model_save_format is None:
            model_save_format = detected_format
            print(f"Auto-detected model save format: {model_save_format}")
        if is_peft is None:
            is_peft = detected_peft
            print(f"Auto-detected PEFT status: {is_peft}")
    # Create checkpoint config
    checkpoint_config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=os.path.dirname(weights_path),
        model_save_format=model_save_format,
        model_cache_dir="",  # Not used for basic loading
        model_repo_id="",  # Not used for basic loading
        save_consolidated=False,  # Keep original behavior
        is_peft=is_peft,
    )

    # Load model using nemo-automodel API
    load_model(
        model=model,
        weights_path=weights_path,
        checkpoint_config=checkpoint_config,
    )

    if optimizer is not None:
        if optimizer_path is None:
            raise ValueError(
                "optimizer_path must be provided when loading optimizer state"
            )
        print(f"Loading optimizer from {optimizer_path}")
        load_optimizer(
            optimizer=optimizer,
            model=model,
            weights_path=optimizer_path,
            scheduler=scheduler,
        )


def convert_dcp_to_hf(
    dcp_ckpt_path: str,
    hf_ckpt_path: str,
    model_name_or_path: str,
    tokenizer_name_or_path: str,
    overwrite: bool = False,
) -> str:
    """Convert a Torch DCP checkpoint to a Hugging Face checkpoint.

    This is not an optimized utility. If checkpoint is too large, consider saving DCP during training
    and using this utility to convert to HF format.

    Args:
        dcp_ckpt_path (str): Path to DCP checkpoint
        hf_ckpt_path (str): Path to save HF checkpoint
        model_name_or_path (str): Model name or path for config
        tokenizer_name_or_path (str, optional): Tokenizer name or path.
                                               Defaults to model_name_or_path if None.
        overwrite (bool, optional): Whether to overwrite existing checkpoint. Defaults to False.

    Returns:
        str: Path to the saved HF checkpoint

    Raises:
        FileExistsError: If HF checkpoint already exists and overwrite is False
    """
    if os.path.exists(hf_ckpt_path) and not overwrite:
        raise FileExistsError(
            f"HF checkpoint already exists at {hf_ckpt_path}. Delete it to run or set overwrite=True."
        )

    os.makedirs(hf_ckpt_path, exist_ok=True)
    weights_path = os.path.join(hf_ckpt_path, "pytorch_model.bin")
    dcp_to_torch_save(dcp_ckpt_path, weights_path)

    # Need to reload and save b/c the state dict is scoped inside the model key {"model": actual_state_dict}
    state_dict = torch.load(weights_path)
    assert set(state_dict.keys()) == {"model"}, (
        f"We expect that the state dict only has the top level model key, but found: {state_dict.keys()}"
    )
    torch.save(state_dict["model"], weights_path)

    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.save_pretrained(hf_ckpt_path)

    # TODO: After the following PR gets merged:
    # https://github.com/NVIDIA-NeMo/RL/pull/148/files
    # tokenizer should be copied from policy/tokenizer/* instead of relying on the model name
    # We can expose a arg at the top level --tokenizer_path to plumb that through.
    # This is more stable than relying on the current NeMo-RL get_tokenizer() which can
    # change release to release.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, trust_remote_code=True
    )
    tokenizer.save_pretrained(hf_ckpt_path)

    return hf_ckpt_path
