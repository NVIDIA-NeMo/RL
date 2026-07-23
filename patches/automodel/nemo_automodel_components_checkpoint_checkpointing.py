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

import glob
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
import yaml
from packaging.version import parse
from safetensors.torch import load_file, save_file
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers.utils import TRANSFORMERS_CACHE

from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from nemo_automodel.components.checkpoint._backports.filesystem import SerializationFormat
from nemo_automodel.components.checkpoint._backports.hf_storage import (
    _HuggingFaceStorageReader,
    _HuggingFaceStorageWriter,
    get_fqn_to_file_index_mapping,
)
from nemo_automodel.components.checkpoint.addons import ConsolidatedHFAddon, PeftAddon
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.components.checkpoint.utils import is_tied_word_embeddings

if TYPE_CHECKING:
    from peft import PeftConfig
    from transformers.tokenization_utils import PreTrainedTokenizerBase


def _is_geq_torch_2_9() -> bool:
    """
    Check if the current torch version is greater than or equal to 2.9.0.
    """
    return parse(torch.__version__).base_version >= "2.9.0"


if _is_geq_torch_2_9():
    from torch.distributed.checkpoint.staging import DefaultStager
    from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType, AsyncSaveResponse


@dataclass
class _AsyncSaveContext:
    """
    Internal container for async checkpointing state.

    One instance is maintained for the model save and one for the optimizer save
    to keep staging/upload futures and the associated process group and stager
    together in a single place.
    """

    stager: Any | None
    process_group: Any | None  # torch.distributed.ProcessGroup
    future: Any | None  # AsyncSaveResponse
    staging_active: bool = False


@dataclass
class CheckpointingConfig:
    """
    Configuration for checkpointing.
    """

    enabled: bool
    checkpoint_dir: str | Path
    model_save_format: str
    model_cache_dir: str | Path
    model_repo_id: str
    save_consolidated: bool
    is_peft: bool
    model_state_dict_keys: list[str] = None  # copy of the model state dict keys before any parallelization
    is_async: bool = False
    dequantize_base_checkpoint: bool | None = None
    original_model_root_dir: str | None = None
    skip_task_head_prefixes_for_base_model: list[str] | None = (
        None  # Parameter prefixes to skip when loading base model
    )

    def __post_init__(self):
        """
        Convert a raw string such as "safetensors" into the right Enum.
        """
        assert self.model_save_format in [v.value for v in SerializationFormat], (
            f"Unsupported model save format: {self.model_save_format}"
        )
        self.model_save_format = SerializationFormat[self.model_save_format.upper()]

        # Async is only enabled for torch >= 2.9.0 currently because of large API changes in async DCP from 2.8.0 to 2.9.0
        if self.is_async and not _is_geq_torch_2_9():
            logging.error("Async mode is only supported for torch >= 2.9.0, disabling async mode")
            self.is_async = False


class Checkpointer:
    """
    High-level checkpoint manager built on torch.distributed.checkpoint (DCP).

    Supports:
    - HF sharded safetensors via custom storage reader/writer
    - Optional consolidated export (config, generation config, tokenizer)
    - PEFT adapter save/load handling
    - Async save for torch >= 2.9.0

    Also provides DP-aware helpers for saving/loading auxiliary state and
    utilities to initialize from a base HF checkpoint.
    """

    def __init__(
        self,
        config: CheckpointingConfig,
        dp_rank: int,
        tp_rank: int,
        pp_rank: int,
        moe_mesh: Optional[DeviceMesh] = None,
    ) -> None:
        """
        Initialize the checkpointer.

        Args:
            config: Checkpointing configuration.
            dp_rank: Data parallel rank for the current process.
            tp_rank: Tensor parallel rank for the current process.
            pp_rank: Pipeline parallel rank for the current process.
            moe_mesh: Optional device mesh used for MoE when adapting state dicts.
        """
        self.config = config
        self.moe_mesh = moe_mesh
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank

        # async specific variables
        self._model_ctx = _AsyncSaveContext(stager=None, process_group=None, future=None, staging_active=False)
        self._optim_ctx = _AsyncSaveContext(stager=None, process_group=None, future=None, staging_active=False)
        if self.config.is_async:
            self._model_ctx.stager = DefaultStager()
            self._optim_ctx.stager = DefaultStager()
            self._model_ctx.process_group = torch.distributed.new_group(backend="gloo")
            self._optim_ctx.process_group = torch.distributed.new_group(backend="gloo")

        self._addons = []
        if self._should_write_hf_metadata():
            self._addons.append(ConsolidatedHFAddon())
        if self.config.is_peft:
            self._addons.append(PeftAddon())

    def save_model(
        self,
        model: nn.Module,
        weights_path: str,
        peft_config: Optional["PeftConfig"] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
    ) -> None:
        """
        Save model weights to `weights_path/model`.

        Behavior:
        - PEFT: write `adapter_model.safetensors` and metadata on rank 0.
        - Safetensors + consolidation: emit HF artifacts under
          `weights_path/model/consolidated` and build a consolidated index.
        - Otherwise: use DCP with a Hugging Face or default storage writer to save shards.

        Args:
            model: Model to checkpoint.
            weights_path: Base directory for checkpoints.
            peft_config: Optional PEFT configuration when saving adapters.
            tokenizer: Optional tokenizer to save with consolidated artifacts.
        """
        # Create the model directories
        model_dir = os.path.join(weights_path, "model")
        consolidated_dir = (
            os.path.join(model_dir, "consolidated") if self._should_write_consolidated_safetensors() else None
        )
        hf_metadata_dir = os.path.join(model_dir, ".hf_metadata") if self._should_write_hf_metadata() else None
        _ensure_dirs(model_dir, consolidated_dir, hf_metadata_dir)

        # Because this call lies outside of the dcp save call, we need to consolidate on all ranks on the main process
        # of all ranks, which lies on the critical path. Therefore, we can only do this outside of async mode.
        consolidate_on_all_ranks = self._should_write_consolidated_safetensors() and not self.config.is_async

        model_state = ModelState(model, self.config.is_peft)
        state_dict = model_state.state_dict()

        # Convert to HF format if using custom model implementations
        state_dict = _maybe_adapt_state_dict_to_hf(model_state.model[0], state_dict, quantization=False)
        # Build the consolidated model.safetensors.index.json if needed
        fqn_to_file_index_mapping = self._maybe_build_consolidated_index(model_state, state_dict)

        # Run pre-saves for addons e.g., PEFT or consolidated HF safetensors
        for addon in self._addons:
            addon.pre_save(
                model_state=model_state,
                model_path=model_dir,
                consolidated_path=consolidated_dir,
                hf_metadata_dir=hf_metadata_dir,
                tokenizer=tokenizer,
                peft_config=peft_config,
                fqn_to_file_index_mapping=fqn_to_file_index_mapping,
                original_model_path=self._get_original_model_path(model_state),
            )

        storage_writer = self._get_storage_writer(
            consolidated_dir, fqn_to_file_index_mapping, model_dir, consolidate_on_all_ranks
        )
        self._model_ctx.future = self._do_save(state_dict, model_dir, storage_writer)

        for addon in self._addons:
            addon.post_save(consolidated_path=consolidated_dir, hf_metadata_path=hf_metadata_dir)

        if consolidate_on_all_ranks:
            consolidate_safetensors_files_on_every_rank(
                input_dir=model_dir,
                output_dir=consolidated_dir,
                fqn_to_index_mapping=fqn_to_file_index_mapping,
                num_threads=5,
            )

    def save_optimizer(
        self, optimizer: torch.optim.Optimizer, model: nn.Module, weights_path: str, scheduler: Optional[Any] = None
    ) -> None:
        """
        Save optimizer (and optional scheduler) state to `weights_path/optim` using DCP.

        Args:
            optimizer: Optimizer whose state will be saved.
            model: Model providing partitioning context for the optimizer wrapper.
            weights_path: Base directory for checkpoints.
            scheduler: Optional LR scheduler to include.
        """
        optimizer_path = os.path.join(weights_path, "optim")
        _ensure_dirs(optimizer_path)
        optimizer_state = OptimizerState(model, optimizer, scheduler)
        state_dict = optimizer_state.state_dict()
        self._optim_ctx.future = self._do_save(state_dict, optimizer_path)

    def load_optimizer(
        self, optimizer: torch.optim.Optimizer, model: nn.Module, weights_path: str, scheduler: Optional[Any] = None
    ) -> None:
        """
        Load optimizer (and optional scheduler) state from `weights_path/optim` using DCP.

        Args:
            optimizer: Optimizer to populate.
            model: Model providing partitioning context for the optimizer wrapper.
            weights_path: Base directory for checkpoints.
            scheduler: Optional LR scheduler to populate.
        """
        optimizer_state = OptimizerState(model, optimizer, scheduler)
        state_dict = optimizer_state.state_dict()
        # Params with no pre-save gradient (e.g. multi-LoRA adapters on
        # Mamba2 out_proj, bypassed by the fused scan kernel) have NO
        # optim.state.* keys in the checkpoint: Adam creates state lazily.
        # The fresh optimizer's template (zero-initialized by
        # get_optimizer_state_dict via _init_optim_state) demands them and
        # DCP's strict planner raises "Missing key in checkpoint state_dict".
        # Stash those template entries during dcp.load (so the planner only
        # sees keys the checkpoint has), then restore them before
        # set_optimizer_state_dict — which requires EVERY param's flat key
        # (_unflatten_optim_state_dict KeyErrors otherwise). The stashed
        # zero-init state matches those params' pre-preemption reality.
        optim_path = os.path.join(weights_path, "optim")
        try:
            ckpt_keys = set(
                FileSystemReader(optim_path).read_metadata().state_dict_metadata.keys()
            )
        except Exception:
            ckpt_keys = None
        stash: dict[str, Any] = {}
        if ckpt_keys is not None and "optim" in state_dict:
            # OptimizerState.state_dict() returns {"optim": {<flat key>: v}}
            # (flatten_optimizer_state_dict=True); DCP metadata keys are
            # "optim." + <flat key>.
            missing = [
                k for k in state_dict["optim"]
                if k.startswith("state.") and f"optim.{k}" not in ckpt_keys
            ]
            if missing:
                logging.warning(
                    "load_optimizer: %d optimizer-state template keys absent "
                    "from checkpoint (params with no pre-save gradient) — "
                    "keeping fresh state for them, e.g. %s",
                    len(missing),
                    missing[0],
                )
                for k in missing:
                    stash[k] = state_dict["optim"].pop(k)
        self._do_load(state_dict, optim_path)
        if stash:
            state_dict["optim"].update(stash)
        optimizer_state.load_state_dict(state_dict)

    def load_model(
        self,
        model: nn.Module,
        model_path: str,
        is_init_step: bool = False,
        use_checkpoint_id: bool = True,
        key_mapping: Optional[dict[str, str]] = None,
    ) -> None:
        """
        Load model weights from `model_path`.

        Behavior:
        - For PEFT (non-init): rank 0 reads `adapter_model.safetensors`, then broadcasts.
        - Otherwise: use DCP with a Hugging Face or default storage reader to populate the state dict.
        - If the model exposes a `state_dict_adapter`, convert to/from HF format as needed.

        Args:
            model: Model or parallelized model parts to load into.
            model_path: Path to the model checkpoint directory or HF snapshot.
            is_init_step: If True, treat load as initialization from a base checkpoint.
            use_checkpoint_id: Pass `checkpoint_id` to DCP if True; disable when using direct HF paths.
            key_mapping: Optional key remapping when reading from HF checkpoints.
        """
        # Validate checkpoint directory
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        model_state = ModelState(
            model,
            is_peft=self.config.is_peft,
            is_init_step=is_init_step,
            skip_task_head_prefixes=getattr(self.config, "skip_task_head_prefixes_for_base_model", None),
        )
        state_dict = model_state.state_dict()
        storage_reader = self._get_storage_reader(model_path, key_mapping, is_init_step=is_init_step)

        state_dict = _maybe_adapt_state_dict_to_hf(
            model_state.model[0], state_dict, quantization=self.config.dequantize_base_checkpoint
        )

        state_dict = self._do_load(state_dict, model_path, storage_reader, is_init_step=is_init_step)

        has_state_dict_adapter = hasattr(model_state.model[0], "state_dict_adapter")
        state_dict = _maybe_adapt_state_dict_from_hf(model_state.model[0], state_dict, moe_mesh=self.moe_mesh)
        model_state.load_state_dict(state_dict, strict=not (len(model_state.model) > 1 or has_state_dict_adapter))

    def load_base_model(
        self,
        model: torch.nn.Module,
        device: torch.device,
        root_dir: str,
        model_name: str | None,
        peft_init_method: str,
        load_base_model: bool = True,
    ) -> None:
        """
        Load a model from the base Hugging Face checkpoint in parallel.

        Args:
            model: Model to load state into
            device: Device to load model onto
            root_dir: Root directory of the model cache or snapshots
            model_name: Name of the model or an absolute path to a snapshot
            peft_init_method: Initialization method used for PEFT adapters
            load_base_model: If True, restore from HF base checkpoint
        """
        to_empty_parameters_only(model, device=device)

        # HF models set _is_hf_initialized to True after initialization.
        # But because we initialize on meta device, these are erroneously set to True.
        # We need to set them to False and call initialize_weights to re-initialize the weights.

        # Gemma3ForConditionalGeneration cannot be pretrained currently. The pinned torch version
        # doesn't support initialize_weights when the model is sharded. This is because Gemma's
        # initialize_weights method requires setting a row to zeros in the embedding matrix.
        # This index selection op is not supported for DTensors in the pinned torch version.
        try:
            model_class = model.config.architectures[0]
        except:
            model_class = ""
        if model_class not in ["Gemma3ForConditionalGeneration", "NemotronHForCausalLM"]:
            for _, module in model.named_modules():
                if hasattr(module, "_is_hf_initialized"):
                    module._is_hf_initialized = False

            # init model weights
            if hasattr(model, "initialize_weights"):
                model.initialize_weights()
            else:
                logging.warning(
                    "Warning: Model does not have initialize_weights method. Requires custom initialization to be implemented."
                )

        # init peft adapters with the scaled weights
        _init_peft_adapters(model, peft_init_method)

        if load_base_model:
            assert model_name is not None, "model_name is required when loading base model"
            self.load_model(
                model,
                model_path=model_name
                if os.path.exists(model_name)
                else get_safetensors_index_path(root_dir, model_name),
                is_init_step=True,
                key_mapping=getattr(model, "_checkpoint_conversion_mapping", None),
            )

        is_tied_lm_head = is_tied_word_embeddings(model)
        self.config.original_model_root_dir = root_dir
        if hasattr(model, "tie_weights") and is_tied_lm_head:
            model.tie_weights()

    def maybe_wait_for_staging(self) -> None:
        """
        Wait for the staging to finish if it is enabled.
        """
        if self._model_ctx.staging_active and self._model_ctx.future is not None:
            self._model_ctx.future.staging_completion.result()
            self._model_ctx.staging_active = False
        if self._optim_ctx.staging_active and self._optim_ctx.future is not None:
            self._optim_ctx.future.staging_completion.result()
            self._optim_ctx.staging_active = False

    def async_wait(self) -> None:
        """
        Wait for the async save to finish.
        """
        if self._model_ctx.future is not None:
            self._model_ctx.future.upload_completion.result()
            self._model_ctx.future = None
        if self._optim_ctx.future is not None:
            self._optim_ctx.future.upload_completion.result()
            self._optim_ctx.future = None

    def save_on_dp_ranks(self, state: Any, state_name: str, path: str) -> None:
        """
        Save the stateful object.

        This function is a helper function currently used to save the dataloader and rng state.

        Args:
            state: Stateful object to save
            state_name: Name of the stateful object
            path: Path to save stateful object
        """
        state_dir = os.path.join(path, state_name)
        _ensure_dirs(state_dir)
        if self.tp_rank == 0 and self.pp_rank == 0:
            torch.save(state.state_dict(), os.path.join(state_dir, f"{state_name}_dp_rank_{self.dp_rank}.pt"))

    def load_on_dp_ranks(self, state: Any, state_name: str, path: str) -> None:
        """
        Load the stateful object.

        This function is a helper function currently used to load the dataloader and rng state.

        Args:
            state: Stateful object to load
            state_name: Name of the stateful object
            path: Path to load stateful object
        """
        state_dir = os.path.join(path, state_name)
        state.load_state_dict(
            torch.load(os.path.join(state_dir, f"{state_name}_dp_rank_{self.dp_rank}.pt"), weights_only=False)
        )

    def close(self) -> None:
        """
        Close the checkpointer.
        """
        self.maybe_wait_for_staging()
        self.async_wait()
        if self._model_ctx.stager is not None:
            self._model_ctx.stager.close()
        if self._optim_ctx.stager is not None:
            self._optim_ctx.stager.close()

    def _do_load(
        self,
        state_dict: dict[str, torch.Tensor],
        path: str,
        storage_reader: Optional[_HuggingFaceStorageReader] = None,
        is_init_step: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Load a state dictionary from `path` using DCP or PEFT special-case logic.

        Args:
            state_dict: Mutable state dict to populate with tensors.
            path: Checkpoint directory path.
            storage_reader: Optional HF storage reader for safetensors.
            is_init_step: True if loading from a base checkpoint during initialization.

        Returns:
            The populated state dictionary (may be replaced for PEFT).
        """
        # Both model and optimizer saving is done in this function
        is_model = True if "/model" in path else False
        # PEFT loading is broadcasted from rank0 so it is a special case
        if self.config.is_peft and is_model and (not is_init_step):
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                state_dict = load_file(os.path.join(path, "adapter_model.safetensors"))
        else:
            dcp.load(state_dict, checkpoint_id=path, storage_reader=storage_reader)
        return state_dict

    def _do_save(
        self, state_dict: dict[str, torch.Tensor], path: str, storage_writer: Optional[_HuggingFaceStorageWriter] = None
    ) -> Optional["AsyncSaveResponse"]:
        """
        Save a state dictionary to `path` using DCP or PEFT special-case logic.

        - For PEFT model saves: only rank 0 writes `adapter_model.safetensors`.
        - If async mode is enabled, schedule an asynchronous save.

        Args:
            state_dict: State dict to be serialized.
            path: Checkpoint directory path.
            storage_writer: Optional HF storage writer for safetensors sharding.

        Returns:
            Optional Future object if async mode is enabled.
        """
        # Both model and optimizer saving is done in this function
        is_model = True if "/model" in path else False
        # PEFT saving is done on rank0 so it is a special case
        if self.config.is_peft and is_model:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                save_file(state_dict, os.path.join(path, "adapter_model.safetensors"))
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        ret = None
        planner = dcp.DefaultSavePlanner(enable_plan_caching=True)
        if self.config.is_async:
            ctx = self._model_ctx if is_model else self._optim_ctx
            ret = dcp.async_save(
                state_dict,
                checkpoint_id=path,
                storage_writer=storage_writer,
                process_group=ctx.process_group,
                async_stager=ctx.stager,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                planner=planner,
            )
            ctx.staging_active = True
        else:
            dcp.save(state_dict, checkpoint_id=path, storage_writer=storage_writer, planner=planner)
        return ret

    def _should_write_consolidated_safetensors(self) -> bool:
        """
        Whether to output consolidated HF weights along with sharded weights.

        Returns True only for non-PEFT safetensors when consolidation is enabled.
        """
        return self.config.save_consolidated and self._should_write_hf_metadata()

    def _should_write_hf_metadata(self) -> bool:
        """
        Whether to write the HF artifacts.
        """
        return self.config.model_save_format == SerializationFormat.SAFETENSORS and not self.config.is_peft

    def _maybe_build_consolidated_index(
        self, model_state: ModelState, state_dict: dict[str, torch.Tensor]
    ) -> Optional[dict[str, int]]:
        """
        Build FQN to shard index mapping for consolidated HF export.

        Uses the base checkpoint index (if present), removes non-persistent keys,
        and assigns new keys to the last shard by default.

        Args:
            model_state: Wrapper exposing the primary model part.
            state_dict: The state dict that will be saved.

        Returns:
            Mapping from FQN to shard index, or None when not consolidating.
        """
        if not self._should_write_hf_metadata():
            return None
        model = model_state.model[0]
        # we first need to find the FQN -> .safetensors mapping
        index_path = get_safetensors_index_path(
            self.config.model_cache_dir,
            self.config.model_repo_id,
        )
        if index_path:
            # HF VLM models may contain a special checkpoint mapping attribute
            fqn_to_file_index_mapping = get_fqn_to_file_index_mapping(
                index_path, getattr(model, "_checkpoint_conversion_mapping", None)
            )
            # some HF models like Moonlight-16B have non-persistent buffers in the base checkpoint
            # however, HF initializes buffers with persistent=False, so we need to make sure these
            # buffer keys are not saved during checkpointing
            keys_to_remove = list(set(fqn_to_file_index_mapping.keys()) - set(self.config.model_state_dict_keys))
            if model_state.is_tied_lm_head:
                keys_to_remove.append(model_state.lm_head_param_name)
            for key in keys_to_remove:
                fqn_to_file_index_mapping.pop(key, None)
        else:
            fqn_to_file_index_mapping = {k: 1 for k in state_dict.keys()}

        # Add any missing keys from the model_state_dict
        # These will go to the same file as the last file (or file 1 for single-file models)
        default_index = max(fqn_to_file_index_mapping.values())

        # add any additional keys that are not in the base checkpoint
        for fqn in list(state_dict.keys()):
            fqn_to_file_index_mapping[fqn] = fqn_to_file_index_mapping.get(fqn, default_index)
        return fqn_to_file_index_mapping

    def _get_storage_writer(
        self,
        consolidated_output_path: Optional[str],
        fqn_to_index_mapping: Optional[dict[str, int]],
        model_path: str,
        consolidate_on_all_ranks: bool = False,
    ) -> Optional[_HuggingFaceStorageWriter]:
        """
        Construct a Hugging Face storage writer for sharded safetensors.

        Args:
            consolidated_output_path: Optional path for consolidated artifacts.
            fqn_to_index_mapping: Optional mapping from FQN to shard index.
            model_path: Path where the model checkpoint is saved.
            consolidate_on_all_ranks: If True, consolidate on all ranks on the main process.

        Returns:
            Configured `_HuggingFaceStorageWriter` or None for non-safetensors.
        """
        if self.config.model_save_format == SerializationFormat.SAFETENSORS:
            return _HuggingFaceStorageWriter(
                path=model_path,
                save_sharded=True,
                consolidated_output_path=consolidated_output_path if not consolidate_on_all_ranks else None,
                fqn_to_index_mapping=fqn_to_index_mapping,
            )

    def _get_storage_reader(
        self, model_path: str, key_mapping: Optional[dict[str, str]], is_init_step: bool = False
    ) -> Optional[_HuggingFaceStorageReader]:
        """
        Construct a Hugging Face storage reader when loading safetensors or during init.

        Args:
            model_path: Path to the model checkpoint directory or HF snapshot.
            key_mapping: Optional key remapping for conversion.
            is_init_step: If True, always produce a reader for base HF load.

        Returns:
            Configured `_HuggingFaceStorageReader` or None for other formats.
        """
        # If loading the model from the base checkpoint, we need to read the base model from the Hugging Face checkpoint
        if self.config.model_save_format == SerializationFormat.SAFETENSORS or is_init_step:
            return _HuggingFaceStorageReader(path=model_path, key_mapping=key_mapping)

    def _get_original_model_path(self, model_state: ModelState) -> str | None:
        """
        Get the path to the original model from the Hugging Face checkpoint.
        """
        if not hasattr(model_state.model[0], "name_or_path"):
            return None
        pretrained_model_name_or_path = getattr(model_state.model[0], "name_or_path")
        return get_safetensors_index_path(
            getattr(self.config, "original_model_root_dir", None) or TRANSFORMERS_CACHE, pretrained_model_name_or_path
        )


def get_safetensors_index_path(cache_dir: str, repo_id: str | None) -> str | None:
    """
    Return the directory containing the first `model.safetensors.index.json` found for given model.

    If no `model.safetensors.index.json` is found then it returns None.

    For example, if the file located is

        /opt/models/models--meta-llama--Llama-3.2-3B/snapshots/13afe.../model.safetensors.index.json

    this function will return the directory path

        /opt/models/models--meta-llama--Llama-3.2-3B/snapshots/13afe...

    This will error if the model hasn't been downloaded or if the cache directory is incorrect.

    Args:
        cache_dir: Path to cache directory
        repo_id: Hugging Face repository ID

    Returns:
        Path to the directory containing the index file.

    Raises:
        FileNotFoundError: If the index file is not found.
    """
    # repo_id can be None if the model is not Hugging Face Hub yet
    if repo_id is None:
        return None

    if os.path.exists(repo_id):
        return repo_id

    repo_dir = f"models--{repo_id.replace('/', '--')}"
    snapshots_root = Path(cache_dir) / repo_dir / "snapshots"

    # Look for an index file inside any snapshot directory.
    pattern = snapshots_root / "*" / "model.safetensors.index.json"
    matches = glob.glob(str(pattern))
    if matches:
        # Return the directory path that contains the index file.
        return str(Path(matches[0]).parent)

    # Fall back: if no index file, return the first available snapshot directory (if any).
    # This is the case for single-file models.
    snapshot_dirs = [p for p in glob.glob(str(snapshots_root / "*")) if Path(p).is_dir()]
    if snapshot_dirs:
        try:
            return snapshot_dirs[0]
        except IndexError:
            raise FileNotFoundError(f"No snapshot directories found in {snapshots_root}")


def to_empty_parameters_only(
    model: nn.Module, *, device: torch.device, recurse: bool = True, dtype: torch.dtype | None = None
) -> nn.Module:
    """
    Move parameters to the specified device without copying storage, skipping buffers.

    Mirrors torch.nn.Module.to_empty but applies only to parameters, not buffers.

    Args:
        model: The module to transform
        device: Target device
        recurse: Whether to recurse into child modules

    Returns:
        The same module instance
    """
    return _apply(model, lambda t: torch.empty_like(t, device=device, dtype=dtype), recurse=recurse)


def save_config(config: dict[str, Any], weights_path: str) -> None:
    """
    Save a config to a weights path.

    Args:
        config: Config to save
        weights_path: Path to save config
    """
    with open(os.path.join(weights_path, "config.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)


def _ensure_dirs(*dirs: Optional[str]) -> None:
    """
    Create directories on all ranks and synchronize across ranks.

    Args:
        *dirs: One or more directory paths that should exist.
    """
    for d in dirs:
        if d:
            os.makedirs(d, exist_ok=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _init_peft_adapters(model: nn.Module, peft_init_method: str) -> None:
    """
    Initialize the PEFT adapters with the scaled weights.

    Args:
        model: Model to initialize PEFT adapters for
        peft_init_method: Method to initialize PEFT adapters e.g. "xavier". See `LinearLoRA` for more details.
    """
    for module in model.modules():
        if hasattr(module, "init_lora_weights"):
            try:
                module.init_lora_weights(peft_init_method)
            except Exception as e:
                logging.warning(f"Failed to initialize weights for PEFT adapter `{module.__class__.__name__}`: {e}")


def _apply(module, fn, recurse=True) -> nn.Module:
    """
    Apply a transformation function to parameters (and gradients) only.

    Mirrors `nn.Module.to_empty` for parameters while skipping buffers. Respects
    future flags controlling in-place vs swap behavior and safely handles
    wrapper subclasses.

    Args:
        module: Module whose parameters are to be transformed.
        fn: Callable applied to each parameter (and its gradient).
        recurse: Whether to recurse into child modules.

    Returns:
        The same module instance after transformation.
    """
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    if recurse:
        for child in module.children():
            _apply(child, fn, recurse=recurse)

    def compute_should_use_set_data(tensor, tensor_applied):
        if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
            # If the new tensor has compatible tensor type as the existing tensor,
            # the current behavior is to change the tensor in-place using `.data =`,
            # and the future behavior is to overwrite the existing tensor. However,
            # changing the current behavior is a BC-breaking change, and we want it
            # to happen in future releases. So for now we introduce the
            # `torch.__future__.get_overwrite_module_params_on_conversion()`
            # global flag to let the user control whether they want the future
            # behavior of overwriting the existing tensor or not.
            return not torch.__future__.get_overwrite_module_params_on_conversion()
        else:
            return False

    should_use_swap_tensors = torch.__future__.get_swap_module_params_on_conversion()
    for key, param in module._parameters.items():
        if param is None:
            continue
        # Tensors stored in modules are graph leaves, and we don't want to
        # track autograd history of `param_applied`, so we have to use
        # `with torch.no_grad():`
        with torch.no_grad():
            param_applied = fn(param)
        p_should_use_set_data = compute_should_use_set_data(param, param_applied)

        # subclasses may have multiple child tensors so we need to use swap_tensors
        p_should_use_swap_tensors = should_use_swap_tensors or is_traceable_wrapper_subclass(param_applied)

        param_grad = param.grad
        if p_should_use_swap_tensors:
            try:
                if param_grad is not None:
                    # Accessing param.grad makes its at::Tensor's use_count 2, which will prevent swapping.
                    # Decrement use count of the gradient by setting to None
                    param.grad = None
                param_applied = torch.nn.Parameter(param_applied, requires_grad=param.requires_grad)
                torch.utils.swap_tensors(param, param_applied)
            except Exception as e:
                if param_grad is not None:
                    param.grad = param_grad
                raise RuntimeError(f"_apply(): Couldn't swap {module._get_name()}.{key}") from e
            out_param = param
        elif p_should_use_set_data:
            param.data = param_applied
            out_param = param
        else:
            assert isinstance(param, torch.nn.Parameter)
            assert param.is_leaf
            out_param = torch.nn.Parameter(param_applied, param.requires_grad)
            module._parameters[key] = out_param

        if param_grad is not None:
            with torch.no_grad():
                grad_applied = fn(param_grad)
            g_should_use_set_data = compute_should_use_set_data(param_grad, grad_applied)
            if p_should_use_swap_tensors:
                grad_applied.requires_grad_(param_grad.requires_grad)
                try:
                    torch.utils.swap_tensors(param_grad, grad_applied)
                except Exception as e:
                    raise RuntimeError(f"_apply(): Couldn't swap {module._get_name()}.{key}.grad") from e
                out_param.grad = param_grad
            elif g_should_use_set_data:
                assert out_param.grad is not None
                out_param.grad.data = grad_applied
            else:
                assert param_grad.is_leaf
                out_param.grad = grad_applied.requires_grad_(param_grad.requires_grad)

    return module


def _maybe_adapt_state_dict_to_hf(
    model_part: nn.Module, state_dict: dict[str, torch.Tensor], quantization: bool = False
) -> dict[str, torch.Tensor]:
    """
    Custom models use state dict adapters to convert the state dict to the Hugging Face format.
    """
    adapter = getattr(model_part, "state_dict_adapter", None)
    if adapter:
        return adapter.to_hf(state_dict, exclude_key_regex=r".*_extra_state.*", quantization=quantization)
    return state_dict


def _maybe_adapt_state_dict_from_hf(
    model_part: nn.Module, state_dict: dict[str, torch.Tensor], moe_mesh: Optional[DeviceMesh] = None
) -> dict[str, torch.Tensor]:
    """
    Custom models use state dict adapters to convert the state dict from the Hugging Face format to the native format.
    """
    adapter = getattr(model_part, "state_dict_adapter", None)
    if adapter:
        ep_mesh_dims = [dim for dim in moe_mesh.mesh_dim_names if dim != "pp"] if moe_mesh is not None else []
        ep_mesh = moe_mesh[tuple(ep_mesh_dims)] if ep_mesh_dims else moe_mesh
        return adapter.from_hf(state_dict, device_mesh=ep_mesh)
    return state_dict
