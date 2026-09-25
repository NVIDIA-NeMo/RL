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
import tempfile
from collections.abc import Mapping
from typing import Any, Optional

import torch
import torch.distributed.checkpoint as dcp
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.checkpoint import (
    CheckpointingConfig as AutomodelCheckpointingConfig,
)
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
)
from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState
from torch import nn
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.device_mesh import DeviceMesh
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


def _resolve_lora_adapter_dir(restore_from: str) -> str:
    """Resolve a ``lora_cfg.restore_from`` path to the adapter directory.

    Accepts a NeMo RL checkpoint weights directory (``step_*/policy/weights``),
    its ``model`` subdirectory, or any directory directly containing
    ``adapter_model.safetensors`` + ``adapter_config.json`` (HF PEFT layout).
    """
    for candidate in (restore_from, os.path.join(restore_from, "model")):
        if os.path.isfile(os.path.join(candidate, "adapter_model.safetensors")):
            return candidate
    raise FileNotFoundError(
        f"dtensor_cfg.lora_cfg.restore_from={restore_from!r}: no "
        "adapter_model.safetensors found there or in its 'model' subdirectory. "
        "restore_from must point to a PEFT adapter checkpoint (a directory "
        "containing adapter_model.safetensors + adapter_config.json, e.g. a "
        "previous run's step_*/policy/weights directory)."
    )


def build_checkpoint_config(
    dtensor_cfg: Mapping[str, Any],
    *,
    model_repo_id: str,
    dequantize_base_checkpoint: bool,
    is_peft: bool,
    is_async: bool,
    skip_task_head_prefixes_for_base_model: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Build the Automodel checkpoint config for a DTensor v2 worker.

    Read NeMo-RL settings from the YAML-backed checkpoint block and forward
    them to Automodel. This integration boundary is shared by the two DTensor
    v2 workers; defaults belong in the exemplar configs.

    Args:
        dtensor_cfg: The worker's ``policy.dtensor_cfg`` / ``value.dtensor_cfg``
            mapping. Automodel checkpoint settings are read from its nested
            ``checkpoint`` block; all other keys are ignored.
        model_repo_id: Forwarded to Automodel's ``CheckpointingConfig.model_repo_id``.
        dequantize_base_checkpoint: Forwarded to
            ``CheckpointingConfig.dequantize_base_checkpoint``; only takes effect
            on a base-checkpoint init load, not on a resume load.
        is_peft: Forwarded to ``CheckpointingConfig.is_peft``.
        is_async: Forwarded to ``CheckpointingConfig.is_async``.
        skip_task_head_prefixes_for_base_model: Optional parameter-prefix list
            forwarded to ``CheckpointingConfig.skip_task_head_prefixes_for_base_model``
            (only set on the returned dict when not None); the value worker
            passes ``["score."]`` to skip the reward head on base-model init loads.

    Returns:
        A dict of only the fields Automodel's ``CheckpointingConfig`` dataclass
        accepts, meant to be splatted into
        ``AutomodelCheckpointingConfig(enabled=True, checkpoint_dir="", **result)``.
    """
    raw_checkpoint_config = dtensor_cfg["checkpoint"]
    model_save_format = raw_checkpoint_config["model_save_format"]
    if model_save_format not in ("torch_save", "safetensors"):
        raise ValueError(
            "dtensor_cfg.checkpoint.model_save_format must be 'torch_save' or "
            "'safetensors' when using DTensor v2."
        )

    checkpoint_config = {
        "model_save_format": model_save_format,
        "save_consolidated": raw_checkpoint_config["save_consolidated"],
        "model_repo_id": model_repo_id,
        "dequantize_base_checkpoint": dequantize_base_checkpoint,
        "is_peft": is_peft,
        "is_async": is_async,
    }
    for field in (
        "single_rank_consolidation",
        "consolidation_timeout_minutes",
    ):
        if field in raw_checkpoint_config:
            checkpoint_config[field] = raw_checkpoint_config[field]

    if skip_task_head_prefixes_for_base_model is not None:
        checkpoint_config["skip_task_head_prefixes_for_base_model"] = (
            skip_task_head_prefixes_for_base_model
        )

    return checkpoint_config


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
    ) -> None:
        """Initialize the Automodel Checkpointer if not already created.

        This method creates a new Checkpointer instance with the provided configuration.
        If a checkpointer already exists, this method does nothing.

        Args:
            config_updates: Automodel checkpoint fields to set during initialization.
        """
        if self.checkpointer is not None:
            return

        if config_updates is None:
            config_updates = {}

        # Let Automodel own validation and normalization. All resource-owning
        # settings are supplied before build() creates async stagers and process
        # groups. NeMo-RL passes explicit paths to every save/load operation, so
        # the configured root is intentionally unused.
        config_updates.setdefault("save_consolidated", "false")
        base_cfg = AutomodelCheckpointingConfig(
            enabled=True,
            checkpoint_dir="",
            **config_updates,
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

    @staticmethod
    def _save_tokenizer_on_rank0(tokenizer: Any, tokenizer_path: str) -> None:
        """Save a tokenizer (or processor) to ``tokenizer_path`` from rank 0 only.

        Unlike model/optimizer state, the tokenizer is *replicated* rather than
        sharded: every rank holds an identical copy, and ``save_pretrained`` writes
        to the same rank-independent filenames (``tokenizer_config.json``, ...).
        Letting all ranks write means N-1 redundant writes racing on the same file.
        ``save_pretrained`` is not atomic (it opens with ``O_TRUNC``, writes, and may
        read files back), so concurrent writers can deadlock on the inode lock. On a
        ``hard``-mounted NFS share this manifests as ranks stuck indefinitely in
        uninterruptible disk sleep, hanging the whole job at checkpoint time.

        This mirrors the rank-0 guard nemo_automodel applies to the same artifacts in
        ``nemo_automodel.components.checkpoint.addons.ConsolidatedHFAddon.pre_save``.

        Note this must NOT be applied to ``dcp.save``-based model/optimizer saves:
        those are collective calls whose output is already rank-disjoint, so guarding
        them would deadlock and drop every non-zero rank's shard.

        Args:
            tokenizer: The tokenizer or processor to save.
            tokenizer_path: Directory to save the tokenizer into.
        """
        is_distributed = torch.distributed.is_initialized()
        if not is_distributed or torch.distributed.get_rank() == 0:
            print(f"Saving tokenizer (or processor) to {tokenizer_path}")
            tokenizer.save_pretrained(tokenizer_path)
        if is_distributed:
            # Keep non-zero ranks from returning while rank 0 is still writing, so
            # that the tokenizer directory is complete for any caller that reads it
            # right after save_checkpoint() returns.
            torch.distributed.barrier()

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
            self._save_tokenizer_on_rank0(tokenizer, tokenizer_path)

        # Async DCP staging reads from the live model and optimizer state. Wait
        # for those copies before callers can update or offload the source tensors;
        # disk upload and deferred consolidation remain asynchronous.
        self.checkpointer.maybe_wait_for_staging()

    def load_lora_adapter(self, model: nn.Module, adapter_dir: str) -> None:
        """Load validated donor adapter weights through the PEFT checkpointer path.

        Args:
            model: Model whose adapters will be initialized.
            adapter_dir: Adapter directory or checkpoint weights directory.
        """
        assert self.checkpointer is not None, (
            "Checkpointer must be initialized before warm starting LoRA adapters."
        )
        if not self.checkpointer.config.is_peft:
            raise RuntimeError(
                "The checkpointer must be initialized with is_peft=True before "
                "warm starting LoRA adapters."
            )

        adapter_dir = os.path.abspath(_resolve_lora_adapter_dir(adapter_dir))
        with tempfile.TemporaryDirectory(prefix="nrl_lora_warm_start_") as staging_dir:
            load_dir = adapter_dir
            if os.path.basename(adapter_dir) != "model":
                # Automodel selects PEFT loading by basename: the path must
                # end in "model" (_is_model_checkpoint_path).
                load_dir = os.path.join(staging_dir, "model")
                os.symlink(adapter_dir, load_dir)
            self.checkpointer.load_model(model=model, model_path=load_dir)
        print(f"Warm-started LoRA adapters from {adapter_dir}")

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

        # load model
        model_dir = os.path.join(weights_path, "model")
        self.checkpointer.load_model(
            model=model,
            model_path=model_dir,
        )

        # load optimizer
        if optimizer_path and optimizer is not None:
            if getattr(optimizer, "master_weights", False):
                # Check the on-disk dtype before DCP copies into current buffers:
                # legacy FP32 masters must not be cast into BF16 int16 remainders.
                metadata = dcp.FileSystemReader(
                    os.path.join(optimizer_path, "optim")
                ).read_metadata()
                optimizer_state = OptimizerState(
                    model,
                    optimizer,
                    is_peft=self.checkpointer.config.is_peft,
                    has_expert_parallelism=self.moe_mesh is not None,
                )
                expected_state, _ = flatten_state_dict(optimizer_state.state_dict())
                for key, value in expected_state.items():
                    if not key.endswith(".master_param"):
                        continue
                    saved = metadata.state_dict_metadata.get(key)
                    if saved is not None and saved.properties.dtype != value.dtype:
                        raise ValueError(
                            f"Cannot resume optimizer master weights: {key} has "
                            f"checkpoint dtype {saved.properties.dtype}, but the "
                            f"current optimizer expects {value.dtype}. "
                            "Check policy.precision and policy.optimizer.kwargs "
                            "(especially store_param_remainders and master_weight_dtype) "
                            "against the configuration and code used to save the checkpoint, "
                            "or start a fresh run without restoring optimizer state."
                        )
                del expected_state, optimizer_state
            self.checkpointer.load_optimizer(
                optimizer=optimizer,
                model=model,
                weights_path=optimizer_path,
                scheduler=scheduler,
            )
