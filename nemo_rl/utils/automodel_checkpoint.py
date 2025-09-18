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
from nemo_automodel.components.checkpoint._backports.filesystem import (
    SerializationFormat,
)

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

# Apply torch backports for compatibility with torch==2.7.1
apply_patches()


def _infer_checkpoint_root(weights_path: str) -> str:
    """Infer checkpoint root directory from weights path.

    When weights_path ends with "…/weights/model", we need the parent of
    the weights directory (the checkpoint root), not the weights directory itself.

    Args:
        weights_path: Path to model weights (e.g., "/path/to/policy/weights/model")

    Returns:
        str: Checkpoint root directory (e.g., "/path/to/policy")
    """
    weights_dir = os.path.dirname(weights_path)
    if weights_dir.endswith("weights"):
        return os.path.dirname(weights_dir)
    return weights_dir


def detect_checkpoint_format(weights_path: str) -> tuple[str, bool]:
    """Detect model save format and PEFT status from checkpoint directory.

    Args:
        weights_path: Path to the checkpoint directory (e.g., weights/model)

    Returns:
        tuple: (model_save_format, is_peft) where:
               model_save_format is "torch_save" for DCP or "safetensors" for safetensors
               is_peft is True if PEFT/adapter patterns are detected
    """
    is_peft = False
    model_save_format = "safetensors"
    try:
        # Iterate through all subdirectories and files recursively
        all_files = []
        for root, dirs, files in os.walk(weights_path):
            all_files.extend(files)

        if any(f.endswith(".distcp") for f in all_files):
            model_save_format = "torch_save"
        elif any(f.endswith(".safetensors") for f in all_files):
            model_save_format = "safetensors"
        elif any(f.endswith((".bin", ".pt", ".pth")) for f in all_files):
            model_save_format = "torch_save"

        if not is_peft:
            is_peft = any("adapter" in f.lower() for f in all_files)

    except (OSError, PermissionError):
        pass

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
    model_state_dict_keys: Optional[list[str]] = None,
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
        save_consolidated: Whether to save consolidated checkpoints (for HF compatibility)
        model_state_dict_keys: Copy of the model state dict keys before any parallelization.
                             If None, will be extracted from the model's current state dict.
    """
    # Create checkpoint config

    # Extract model state dict keys if not provided
    if model_state_dict_keys is None:
        model_state_dict_keys = list(model.state_dict().keys())

    valid_formats = {"safetensors", "torch_save"}
    if model_save_format not in valid_formats:
        raise ValueError(
            f"Unsupported model_save_format='{model_save_format}'. "
            f"Expected one of {sorted(valid_formats)}."
        )

    # Ensure target directories exist
    os.makedirs(weights_path, exist_ok=True)
    if optimizer_path:
        os.makedirs(optimizer_path, exist_ok=True)
    if tokenizer_path:
        os.makedirs(tokenizer_path, exist_ok=True)

    checkpoint_config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=_infer_checkpoint_root(weights_path),
        model_save_format=model_save_format,
        model_cache_dir="",
        model_repo_id="",
        save_consolidated=save_consolidated,
        is_peft=is_peft,
        model_state_dict_keys=model_state_dict_keys,
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
) -> None:
    """Load a model weights and optionally optimizer state.

    Args:
        model: The PyTorch model whose weights to update
        weights_path: Path to load model weights from
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into
        optimizer_path: Path to load optimizer state from (required if optimizer provided)
    """
    print(f"Loading weights from {weights_path}")

    model_save_format, is_peft = detect_checkpoint_format(weights_path)

    try:
        format_enum = SerializationFormat[model_save_format.upper()]

        # append /model to the weights_path if it doesn't exist
        # TODO: remove this once nemo-automodel is updated
        if not weights_path.endswith("/model"):
            weights_path = os.path.join(weights_path, "model")

        # Load model using nemo-automodel API
        load_model(
            model=model,
            model_path=weights_path,
            model_save_format=format_enum,
            is_peft=is_peft,
        )
    except FileNotFoundError as e:
        msg = (
            f"Failed to load model from '{weights_path}': {e}\n"
            "Note: DTensorPolicyWorkerV2 expects:\n"
            "  - Model shards under '<checkpoint_root>/weights/model'\n"
            "  - Optimizer states under '<checkpoint_root>/optimizer/optim'\n"
            "Please verify your checkpoint layout."
        )
        raise FileNotFoundError(msg) from e

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
