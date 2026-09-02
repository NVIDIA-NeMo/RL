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

import logging
import os
from typing import Any, Mapping, Optional

import torch
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.checkpoint import (
    CheckpointingConfig as AutomodelCheckpointingConfig,
)
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
)
from nemo_automodel.components.checkpoint.utils import is_cloud_path
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoTokenizer

from nemo_rl.utils.native_checkpoint import save_tokenizer_on_rank0

logger = logging.getLogger(__name__)


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
# NeMo-RL always finalizes async saves before promoting tmp_step_N, owns
# checkpoint selection/retention, and does not expose Diffusers checkpoints.
_UNSUPPORTED_AUTOMODEL_CONFIG_FIELDS = frozenset(
    {
        "allow_legacy_pickle_restore",
        "best_metric_key",
        "diffusers_compatible",
        "max_recent_checkpoints",
        "wait_for_staging",
    }
)


def _extract_automodel_config_updates(
    config_updates: Mapping[str, Any],
) -> dict[str, Any]:
    """Return supported Automodel fields and reject known unsupported ones."""
    unsupported_fields = sorted(
        _UNSUPPORTED_AUTOMODEL_CONFIG_FIELDS.intersection(config_updates)
    )
    if unsupported_fields:
        raise ValueError(
            "Unsupported checkpointing field(s) for NeMo-RL's Automodel "
            f"integration: {', '.join(unsupported_fields)}. NeMo-RL owns async "
            "save finalization, checkpoint metric selection, and checkpoint "
            "retention; use metric_name, keep_top_k, and ft_keep_latest_k for "
            "the corresponding NeMo-RL behavior. Diffusers checkpoint export is "
            "not supported by NeMo-RL. Automodel's legacy pickle restore only "
            "applies to training-state loaders that NeMo-RL does not use."
        )

    return {
        key: value
        for key, value in config_updates.items()
        if key in _AUTOMODEL_CONFIG_FIELDS
    }


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
    logger.warning(
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

        # Let Automodel own defaults, validation, and normalization. All
        # resource-owning settings must be present before build() creates async
        # stagers and dedicated process groups on every rank.
        automodel_config_updates = _extract_automodel_config_updates(config_updates)
        base_cfg = AutomodelCheckpointingConfig(
            enabled=True,
            checkpoint_dir=checkpoint_root or "",
            **automodel_config_updates,
        )
        self.checkpointer = base_cfg.build(
            dp_rank=self._get_dp_rank(),
            tp_rank=self._get_tp_rank(),
            pp_rank=0,
            moe_mesh=self.moe_mesh,
        )

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
        *,
        is_final_checkpoint: bool,
        peft_config: Optional[PeftConfig] = None,
    ) -> None:
        """Save a checkpoint of the model.

        The optimizer states are saved only if `optimizer` and `optimizer_path` are provided.
        Any previous async save is completed before a new one starts.
        When async saving is enabled, this method returns after model and optimizer
        staging is complete; upload and consolidation may continue in the background.

        Args:
            model: The model to save.
            weights_path: Path to save model weights.
            optimizer: Optional optimizer to save.
            optimizer_path: Optional path to save optimizer state.
            scheduler: Optional learning rate scheduler.
            tokenizer: Optional tokenizer to save with the checkpoint.
            tokenizer_path: Optional path to save tokenizer separately.
            is_final_checkpoint: Whether this checkpoint completes the training
                run, either at the configured final step or after a deliberate
                early stop. Automodel's ``save_consolidated="final"`` mode
                consolidates these checkpoints. Timeout recovery checkpoints are
                resumable and are not considered final.
            peft_config: Optional PEFT configuration.
        """
        print(f"Saving checkpoint to {weights_path}")
        assert self.checkpointer is not None, (
            "Checkpointer must be initialized before saving checkpoint. "
            "Call init_checkpointer() first."
        )

        configured_root = self.checkpointer.config.checkpoint_dir
        if is_cloud_path(configured_root) != is_cloud_path(weights_path):
            raise ValueError(
                "Automodel checkpoint storage cannot change between local and "
                "cloud paths after initialization: "
                f"{configured_root!s} -> {weights_path}. Initialize the "
                "Checkpointer with the target checkpoint root."
            )

        # Automodel keeps one future each for model and optimizer state. Finish
        # the previous save before those future handles can be replaced.
        self.checkpointer.async_wait()

        self.checkpointer.save_model(
            model=model,
            weights_path=weights_path,
            peft_config=peft_config,
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

        # Async DCP staging reads from the live model and optimizer state. Wait
        # for those copies before callers can update or offload the source tensors;
        # disk upload and deferred consolidation remain asynchronous.
        self.checkpointer.maybe_wait_for_staging()

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
