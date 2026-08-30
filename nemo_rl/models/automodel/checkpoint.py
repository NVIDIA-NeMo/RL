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
from dataclasses import fields, replace
from typing import Any, Optional

import torch
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.checkpoint._backports.filesystem import (
    SerializationFormat,
)
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
)
from nemo_automodel.components.checkpoint.config import (
    CheckpointingConfig as AutomodelCheckpointingConfig,
)
from nemo_automodel.components.checkpoint.config import SaveConsolidatedMode
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoTokenizer

from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.native_checkpoint import save_tokenizer_on_rank0


_CHECKPOINTER_RESOURCE_FIELDS = frozenset(
    {
        "consolidation_timeout_minutes",
        "is_async",
        "is_peft",
        "model_save_format",
        "single_rank_consolidation",
    }
)
_AUTOMODEL_CONFIG_FIELDS = frozenset(
    {
        "consolidation_timeout_minutes",
        "dequantize_base_checkpoint",
        "is_async",
        "is_peft",
        "model_cache_dir",
        "model_repo_id",
        "model_save_format",
        "save_consolidated",
        "single_rank_consolidation",
        "skip_task_head_prefixes_for_base_model",
    }
)


def _patch_qwen_vl_vision_key_mapping() -> None:
    """Re-add the Qwen2.5-VL ``^visual`` -> ``model.visual`` checkpoint key rename.

    Workaround for a transformers v5.5.0 regression. transformers #44627 moved
    VLM checkpoint conversions into the main mapping, but copied the Qwen-VL
    visual key mapping incorrectly: ``visual.*`` checkpoint keys no longer map
    to ``model.visual.*``. transformers #45358 fixed those VLM mappings in v5.6,
    but the Automodel commit NeMo-RL can currently pin to still depends on
    transformers v5.5.0. Automodel's ``get_combined_key_mapping`` mirrors the
    transformers ``WeightRenaming`` entries, so the bad v5.5.0 mapping leaves
    vision-tower checkpoint keys unmapped and FSDP2
    ``set_model_state_dict(strict=False)`` drops them in ``load_base_model``.
    The vision tower is then left randomly initialized, making the training
    forward diverge from vLLM (token_mult_prob_error).

    This wraps ``get_combined_key_mapping`` to inject the missing rule for
    ``qwen2_5_vl``/``qwen2_vl``. It is idempotent: the rule is only added when no
    existing rule already targets ``model.visual``. Remove this after Automodel
    upgrades its transformers dependency to a version that includes #45358.
    """
    # Escape hatch (also used for A/B validation of this workaround).
    if os.environ.get("NRL_DISABLE_QWENVL_VISION_PATCH") == "1":
        return

    import nemo_automodel.components.checkpoint.checkpointing as _am_ckpt

    _vision_nested = {"qwen2_5_vl", "qwen2_vl"}
    _orig = _am_ckpt.get_combined_key_mapping

    if getattr(_orig, "_nrl_vision_patch", False):
        return

    def _patched_get_combined_key_mapping(model_type, model_key_mapping=None):
        result = _orig(model_type, model_key_mapping)
        if model_type in _vision_nested:
            result = dict(result or {})
            if not any(str(t).startswith("model.visual") for t in result.values()):
                result[r"^visual\."] = "model.visual."
            if not any(
                str(t).startswith("model.language_model") for t in result.values()
            ):
                result[r"^model(?!\.(language_model|visual))"] = "model.language_model"
        return result or None

    _patched_get_combined_key_mapping._nrl_vision_patch = True
    # Expose the wrapped original so the removal tripwire test
    # (test_qwen_vl_vision_key_mapping_workaround_still_needed) can query the real
    # (unpatched) mapping and detect when transformers #45358 (>=5.6) makes this obsolete.
    _patched_get_combined_key_mapping._nrl_orig = _orig
    _am_ckpt.get_combined_key_mapping = _patched_get_combined_key_mapping


try:
    _patch_qwen_vl_vision_key_mapping()
except Exception as e:  # pragma: no cover - defensive: never break import
    import logging

    logging.getLogger(__name__).warning(
        "Failed to apply Qwen2.5-VL vision-tower key-mapping patch "
        "(transformers #44627/#45358 workaround): %s",
        e,
    )


class AutomodelCheckpointManager:
    """Manages checkpointing for DTensor-based models using nemo_automodel's Checkpointer.

    This class provides a clean interface for saving and loading model checkpoints,
    wrapping the nemo_automodel Checkpointer with configuration management.

    Attributes:
        checkpointer: The underlying nemo_automodel Checkpointer instance.
        checkpoint_config: The current checkpoint configuration.
    """

    def __init__(
        self,
        dp_mesh: DeviceMesh,
        tp_mesh: DeviceMesh,
        moe_mesh: Optional[DeviceMesh] = None,
    ):
        """Initialize the AutomodelCheckpointManager.

        Args:
            dp_mesh: The data parallel device mesh.
            tp_mesh: The tensor parallel device mesh.
            moe_mesh: Optional MoE device mesh.
        """
        self.checkpointer: Optional[Checkpointer] = None
        self.checkpoint_config: Optional[AutomodelCheckpointingConfig] = None
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

        # Let Automodel own defaults and normalization for its checkpoint fields.
        # In particular, its canonical default is save_consolidated="final" and
        # legacy booleans are normalized to false/every in __post_init__.
        automodel_config_updates = {
            key: value
            for key, value in config_updates.items()
            if key in _AUTOMODEL_CONFIG_FIELDS
        }
        base_cfg = AutomodelCheckpointingConfig(
            enabled=True,
            checkpoint_dir=checkpoint_root or "",
            **automodel_config_updates,
        )
        self.checkpoint_config = base_cfg
        self.checkpointer = self._build_checkpointer(base_cfg)

    def _build_checkpointer(self, config: AutomodelCheckpointingConfig) -> Checkpointer:
        """Build an Automodel Checkpointer for this worker's mesh ranks."""
        return config.build(
            dp_rank=self._get_dp_rank(),
            tp_rank=self._get_tp_rank(),
            pp_rank=0,
            moe_mesh=self.moe_mesh,
        )

    @staticmethod
    def _updated_config(
        config: AutomodelCheckpointingConfig,
        config_updates: dict[str, Any],
        checkpoint_root: Optional[str],
    ) -> AutomodelCheckpointingConfig:
        """Create and validate the prospective Automodel configuration."""
        updates = dict(config_updates)
        if checkpoint_root is not None:
            updates["checkpoint_dir"] = checkpoint_root

        model_save_format = updates.get("model_save_format", config.model_save_format)
        if isinstance(model_save_format, SerializationFormat):
            model_save_format = model_save_format.value
        updates["model_save_format"] = model_save_format

        return replace(config, **updates)

    @staticmethod
    def _requires_checkpointer_rebuild(
        current_config: AutomodelCheckpointingConfig,
        updated_config: AutomodelCheckpointingConfig,
    ) -> bool:
        """Return whether an update changes Checkpointer-owned resources."""
        resource_field_changed = any(
            getattr(current_config, field_name) != getattr(updated_config, field_name)
            for field_name in _CHECKPOINTER_RESOURCE_FIELDS
        )
        consolidation_group_requirement_changed = (
            current_config.save_consolidated != SaveConsolidatedMode.FALSE
        ) != (updated_config.save_consolidated != SaveConsolidatedMode.FALSE)
        return resource_field_changed or consolidation_group_requirement_changed

    @staticmethod
    def _apply_config_in_place(
        target: AutomodelCheckpointingConfig,
        source: AutomodelCheckpointingConfig,
    ) -> None:
        """Copy a validated config while preserving Checkpointer references."""
        for field in fields(source):
            setattr(target, field.name, getattr(source, field.name))

    def _replace_checkpointer(self, config: AutomodelCheckpointingConfig) -> None:
        """Replace constructor-owned resources without invalidating on build failure."""
        assert self.checkpointer is not None
        old_checkpointer = self.checkpointer

        # Do not overlap a previous save with construction of new stagers/groups.
        # Keep the old groups alive until construction succeeds so a failed build
        # leaves the manager usable.
        old_checkpointer.maybe_wait_for_staging()
        old_checkpointer.async_wait()
        new_checkpointer = self._build_checkpointer(config)

        try:
            old_checkpointer.close()
        except Exception as close_error:
            try:
                new_checkpointer.close()
            except Exception as cleanup_error:
                close_error.add_note(
                    f"Replacement Checkpointer cleanup also failed: {cleanup_error!r}"
                )
            raise

        self.checkpointer = new_checkpointer
        self.checkpoint_config = config

    def update_checkpointer_config(
        self,
        config_updates: Optional[dict[str, Any]] = None,
        checkpoint_root: Optional[str] = None,
    ) -> None:
        """Update the configuration of an existing Checkpointer.

        This method updates the mutable config fields on the existing Checkpointer instance.
        If no checkpointer exists, this method does nothing.

        Checkpointer construction creates async stagers and dedicated process groups.
        Changes to fields that own those resources close and rebuild the Checkpointer on
        every rank. Other changes are copied onto the existing config after validation.

        Args:
            config_updates: Dict of CheckpointingConfig fields to update.
            checkpoint_root: Optional root directory for checkpoints.
        """
        if self.checkpointer is None:
            return

        if config_updates is None:
            config_updates = {}

        assert self.checkpoint_config is not None
        cfg = self.checkpoint_config
        updated_cfg = self._updated_config(cfg, config_updates, checkpoint_root)

        if self._requires_checkpointer_rebuild(cfg, updated_cfg):
            self._replace_checkpointer(updated_cfg)
            return

        self._apply_config_in_place(cfg, updated_cfg)
        self.checkpoint_config = cfg

    def finalize_async_save(self) -> None:
        """Block until in-flight async checkpoint writes have landed on disk.

        With ``is_async=True`` the Automodel Checkpointer hands both the model
        and optimizer state to ``dcp.async_save``, which stages them and uploads
        from a separate process. Those writes address files by path, so the
        caller must not rename ``tmp_step_N`` to ``step_N`` until they finish --
        otherwise the writer re-creates ``tmp_step_N`` and the promoted
        checkpoint is missing its optimizer shards and ``.metadata``.

        Safe to call when async saving is off or no save is in flight; both
        underlying calls are no-ops in that case.
        """
        if self.checkpointer is None:
            return
        self.checkpointer.maybe_wait_for_staging()
        self.checkpointer.async_wait()

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
        *,
        is_final_checkpoint: bool,
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
            is_final_checkpoint: Whether this is the terminal training checkpoint.
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
            if key in _AUTOMODEL_CONFIG_FIELDS
        }
        save_peft_config = checkpointing_cfg.get("peft_config")
        if lora_enabled:
            checkpoint_kwargs["is_peft"] = True
            save_peft_config = peft_config

        checkpoint_root = _infer_checkpoint_root(weights_path)

        # Update checkpointer configuration
        self.update_checkpointer_config(
            config_updates=checkpoint_kwargs, checkpoint_root=checkpoint_root
        )

        self.checkpointer.save_model(
            model=model,
            weights_path=weights_path,
            peft_config=save_peft_config,
            tokenizer=tokenizer if tokenizer_path is None else None,
            is_final_checkpoint=is_final_checkpoint,
        )

        if optimizer_path and optimizer is not None:
            self.checkpointer.save_optimizer(
                optimizer=optimizer,
                model=model,
                weights_path=optimizer_path,
                scheduler=scheduler,
            )

        if tokenizer_path and tokenizer is not None:
            # Rank-0 guarded: passing tokenizer_path bypasses save_model()'s
            # ConsolidatedHFAddon (we pass tokenizer=None above), which is where
            # nemo_automodel applies its own rank-0 guard, so we must apply it here.
            save_tokenizer_on_rank0(tokenizer, tokenizer_path)

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
