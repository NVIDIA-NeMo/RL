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
"""Automodel checkpoint utilities for DTensor policy workers.

This module provides a wrapper class around the nemo_automodel Checkpointer
for saving and loading model checkpoints in DTensor-based policy workers.
"""

import os
from typing import Any, Optional

import torch
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.checkpoint._backports.filesystem import (
    SerializationFormat,
)
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
)
from nemo_automodel.components.checkpoint.checkpointing import (
    CheckpointingConfig as AutomodelCheckpointingConfig,
)
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoTokenizer
from huggingface_hub.constants import HF_HUB_CACHE as TRANSFORMERS_CACHE

from nemo_rl.utils.checkpoint import CheckpointingConfig


class AutomodelCheckpointManager:
    """Manages checkpointing for DTensor-based models using nemo_automodel's Checkpointer.

    This class provides a clean interface for saving and loading model checkpoints,
    wrapping the nemo_automodel Checkpointer with configuration management.

    Attributes:
        checkpointer: The underlying nemo_automodel Checkpointer instance.
        checkpoint_config: The current checkpoint configuration.
        model_state_dict_keys: List of model state dict keys for checkpoint validation.
    """

    def __init__(
        self,
        dp_mesh: DeviceMesh,
        tp_mesh: DeviceMesh,
        model_state_dict_keys: Optional[list[str]] = None,
        moe_mesh: Optional[DeviceMesh] = None,
    ):
        """Initialize the AutomodelCheckpointManager.

        Args:
            dp_mesh: The data parallel device mesh.
            tp_mesh: The tensor parallel device mesh.
            model_state_dict_keys: Optional list of model state dict keys.
            moe_mesh: Optional MoE device mesh.
        """
        self.checkpointer: Optional[Checkpointer] = None
        self.checkpoint_config: Optional[AutomodelCheckpointingConfig] = None
        self.model_state_dict_keys = model_state_dict_keys
        self.dp_mesh = dp_mesh
        self.tp_mesh = tp_mesh
        self.moe_mesh = moe_mesh

    def _get_dp_rank(self) -> int:
        """Get the data parallel rank."""
        return torch.distributed.get_rank(self.dp_mesh.get_group())

    def _get_tp_rank(self) -> int:
        """Get the tensor parallel rank."""
        return torch.distributed.get_rank(self.tp_mesh.get_group())

    def init_checkpointer(
        self,
        config_updates: Optional[dict[str, Any]] = None,
        checkpoint_root: Optional[str] = None,
    ) -> None:
        """Initialize the Automodel Checkpointer if not already created.

        This method creates a new Checkpointer instance with the provided configuration.
        If a checkpointer already exists, this method does nothing.

        Args:
            config_updates: Dict of CheckpointingConfig fields to set during initialization.
            checkpoint_root: Optional root directory for checkpoints.
        """
        if self.checkpointer is not None:
            return

        if config_updates is None:
            config_updates = {}

        dp_rank = self._get_dp_rank()
        tp_rank = self._get_tp_rank()
        pp_rank = 0

        # Initialize a base config with sensible defaults
        base_cfg = AutomodelCheckpointingConfig(
            enabled=True,
            checkpoint_dir=checkpoint_root or "",
            model_save_format=config_updates.get("model_save_format", "safetensors"),
            model_cache_dir=config_updates.get("model_cache_dir", ""),
            model_repo_id=config_updates.get("model_repo_id", ""),
            save_consolidated=config_updates.get("save_consolidated", False),
            is_peft=config_updates.get("is_peft", False),
            model_state_dict_keys=self.model_state_dict_keys,
            is_async=config_updates.get("is_async", False),
            dequantize_base_checkpoint=config_updates.get(
                "dequantize_base_checkpoint", False
            ),
        )
        self.checkpoint_config = base_cfg
        self.checkpointer = Checkpointer(
            config=base_cfg,
            dp_rank=dp_rank,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            moe_mesh=self.moe_mesh,
        )

    def update_checkpointer_config(
        self,
        config_updates: Optional[dict[str, Any]] = None,
        checkpoint_root: Optional[str] = None,
    ) -> None:
        """Update the configuration of an existing Checkpointer.

        This method updates the mutable config fields on the existing Checkpointer instance.
        If no checkpointer exists, this method does nothing.

        Note: Some config changes (like model_save_format) require rebuilding the
        checkpointer's internal addons list. This method handles that automatically.

        Args:
            config_updates: Dict of CheckpointingConfig fields to update.
            checkpoint_root: Optional root directory for checkpoints.
        """
        if self.checkpointer is None:
            return

        if config_updates is None:
            config_updates = {}

        cfg = self.checkpointer.config
        if checkpoint_root is not None:
            cfg.checkpoint_dir = checkpoint_root
        for k, v in config_updates.items():
            if k == "model_save_format":
                # Ensure enum type
                v = SerializationFormat[v.upper()] if isinstance(v, str) else v
            setattr(cfg, k, v)
        # Ensure model_state_dict_keys is current
        if self.model_state_dict_keys is not None:
            cfg.model_state_dict_keys = self.model_state_dict_keys

        # Rebuild _addons list based on updated config
        # This is necessary because _addons is populated during __init__ based on config
        self._rebuild_checkpointer_addons()

    def _rebuild_checkpointer_addons(self) -> None:
        """Rebuild the checkpointer's _addons list based on current config.

        The Checkpointer's _addons list is populated during __init__ based on config.
        When config changes (e.g., model_save_format or is_peft), we need to rebuild
        the addons list to match the new config.
        """
        if self.checkpointer is None:
            return

        from nemo_automodel.components.checkpoint.addons import (
            ConsolidatedHFAddon,
            PeftAddon,
        )

        self.checkpointer._addons = []
        if self.checkpointer._should_write_hf_metadata():
            self.checkpointer._addons.append(ConsolidatedHFAddon())
        if self.checkpointer.config.is_peft:
            self.checkpointer._addons.append(PeftAddon())

    def set_model_state_dict_keys(self, keys: list[str]) -> None:
        """Set the model state dict keys for checkpoint validation.

        Args:
            keys: List of model state dict keys.
        """
        self.model_state_dict_keys = keys
        if self.checkpointer is not None:
            self.checkpointer.config.model_state_dict_keys = keys

    def load_base_model(
        self,
        model: nn.Module,
        model_name: str,
        hf_cache_dir: Optional[str] = None,
        dequantize_base_checkpoint: bool = False,
        peft_init_method: Optional[str] = None,
    ) -> None:
        """Load base model weights using the Automodel Checkpointer.

        This method loads the initial HuggingFace model weights into the parallelized model.

        Args:
            model: The model to load weights into.
            model_name: Name or path of the model.
            hf_cache_dir: Optional HuggingFace cache directory.
            dequantize_base_checkpoint: Whether to dequantize the base checkpoint.

        Raises:
            AssertionError: If checkpointer has not been initialized.
        """
        assert self.checkpointer is not None, (
            "Checkpointer must be initialized before loading base model. "
            "Call init_checkpointer() first."
        )

        self.update_checkpointer_config(
            config_updates={
                "model_repo_id": model_name,
                "dequantize_base_checkpoint": dequantize_base_checkpoint,
            },
            checkpoint_root=None,
        )
        self.checkpointer.config.model_state_dict_keys = self.model_state_dict_keys

        self.checkpointer.load_base_model(
            model,
            device=torch.cuda.current_device(),
            root_dir=hf_cache_dir or TRANSFORMERS_CACHE,
            model_name=model_name,
            peft_init_method=peft_init_method,
            load_base_model=True,
        )

    def save_checkpoint(
        self,
        model: nn.Module,
        weights_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_path: Optional[str] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        tokenizer_path: Optional[str] = None,
        checkpointing_cfg: Optional[CheckpointingConfig] = None,
        lora_enabled: bool = False,
        peft_config: Optional[PeftConfig] = None,
    ) -> None:
        """Save a checkpoint of the model.

        The optimizer states are saved only if `optimizer` and `optimizer_path` are provided.

        Args:
            model: The model to save.
            weights_path: Path to save model weights.
            optimizer: Optional optimizer to save.
            optimizer_path: Optional path to save optimizer state.
            scheduler: Optional learning rate scheduler.
            tokenizer: Optional tokenizer to save with the checkpoint.
            tokenizer_path: Optional path to save tokenizer separately.
            checkpointing_cfg: Checkpointing configuration.
            lora_enabled: Whether LoRA is enabled.
            peft_config: Optional PEFT configuration.
        """
        print(f"Saving checkpoint to {weights_path}")
        assert self.checkpointer is not None, (
            "Checkpointer must be initialized before saving checkpoint. "
            "Call init_checkpointer() first."
        )
        if checkpointing_cfg is None:
            raise ValueError(
                "checkpointing_cfg must be provided when saving checkpoint"
            )

        # Extract only the checkpointing configuration keys that exist
        checkpoint_kwargs = {
            key: value
            for key, value in checkpointing_cfg.items()
            if key
            in {
                "model_save_format",
                "save_consolidated",
                "is_peft",
                "peft_config",
                "model_cache_dir",
                "model_repo_id",
                "is_async",
                "dequantize_base_checkpoint",
            }
        }
        if lora_enabled:
            checkpoint_kwargs["is_peft"] = True
            checkpoint_kwargs["peft_config"] = peft_config

        checkpoint_root = _infer_checkpoint_root(weights_path)

        # Update checkpointer configuration
        self.update_checkpointer_config(
            config_updates=checkpoint_kwargs, checkpoint_root=checkpoint_root
        )

        self.checkpointer.save_model(
            model=model,
            weights_path=weights_path,
            peft_config=checkpoint_kwargs.get("peft_config"),
            tokenizer=tokenizer if tokenizer_path is None else None,
        )

        if optimizer_path and optimizer is not None:
            self.checkpointer.save_optimizer(
                optimizer=optimizer,
                model=model,
                weights_path=optimizer_path,
                scheduler=scheduler,
            )

        if tokenizer_path and tokenizer is not None:
            print(f"Saving tokenizer (or processor) to {tokenizer_path}")
            tokenizer.save_pretrained(tokenizer_path)

    def load_checkpoint(
        self,
        model: nn.Module,
        weights_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        optimizer_path: Optional[str] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> None:
        """Load a checkpoint into the model using Automodel Checkpointer.

        Args:
            model: The model to load weights into.
            weights_path: Path to the checkpoint weights.
            optimizer: Optional optimizer to load state into.
            optimizer_path: Optional path to optimizer checkpoint.
            scheduler: Optional learning rate scheduler.
        """
        print(f"Loading weights from {weights_path}")
        assert self.checkpointer is not None, (
            "Checkpointer must be initialized before loading checkpoint. "
            "Call init_checkpointer() first."
        )

        model_save_format, is_peft = detect_checkpoint_format(weights_path)

        weights_dir = os.path.dirname(weights_path)
        checkpoint_root = (
            os.path.dirname(weights_dir)
            if weights_dir.endswith("weights")
            else weights_dir
        )

        # Update checkpointer configuration
        self.update_checkpointer_config(
            config_updates={
                "model_save_format": model_save_format,
                "is_peft": is_peft,
                "dequantize_base_checkpoint": False,  # the saved checkpoint is already dequantized
            },
            checkpoint_root=checkpoint_root,
        )

        model_dir = (
            weights_path
            if weights_path.endswith("/model")
            else os.path.join(weights_path, "model")
        )

        self.checkpointer.load_model(
            model=model,
            model_path=model_dir,
        )

        if optimizer_path and optimizer is not None:
            self.checkpointer.load_optimizer(
                optimizer=optimizer,
                model=model,
                weights_path=optimizer_path,
                scheduler=scheduler,
            )


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
