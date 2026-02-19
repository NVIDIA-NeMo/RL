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

import contextlib
import gc
import warnings
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Generator, Optional

import ray
import torch
from nemo_automodel.components.distributed.cp_utils import (
    create_context_parallel_ctx,
)
from nemo_automodel.components.distributed.cp_utils import (
    get_train_context as get_train_context_automodel,
)
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from torch import nn
from transformers import (
    AutoTokenizer,
)

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.data import (
    check_sequence_dim,
    get_microbatch_iterator,
    process_global_batch,
)
from nemo_rl.models.automodel.setup import (
    setup_distributed,
    setup_model_and_optimizer,
    validate_and_prepare_config,
)
from nemo_rl.models.automodel.train import (
    aggregate_training_statistics,
    automodel_forward_backward,
    forward_with_post_processing_fn,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import (
    apply_torch_aten_alias_tensor_patch,
    apply_transformer_engine_patch,
)
from nemo_rl.models.value.config import ValueConfig
from nemo_rl.models.value.interfaces import ValueOutputSpec
from nemo_rl.utils.automodel_checkpoint import AutomodelCheckpointManager
from nemo_rl.utils.checkpoint import CheckpointingConfig
from nemo_rl.utils.nsys import wrap_with_nvtx_name


@contextlib.contextmanager
def get_train_context(
    cp_size: int,
    cp_mesh: Any,
    cp_buffers: list,
    sequence_dim: int,
    dtype: torch.dtype,
    autocast_enabled: bool = True,
) -> Generator[None, None, None]:
    """Create combined context manager for training with context parallel and autocast."""
    with contextlib.ExitStack() as stack:
        context_parallel_ctx = None
        if cp_size > 1:
            # Create context parallel context
            context_parallel_ctx = create_context_parallel_ctx(
                cp_mesh=cp_mesh,
                cp_buffers=cp_buffers,
                cp_seq_dims=[sequence_dim] * len(cp_buffers),
                cp_no_restore_buffers=set(cp_buffers),
            )

        stack.enter_context(
            get_train_context_automodel(False, False, context_parallel_ctx)()
        )
        if autocast_enabled:
            stack.enter_context(torch.autocast(device_type="cuda", dtype=dtype))
        yield


class ValuePostProcessor:
    """Post-processor for value function forward pass."""

    def __init__(
        self,
        cfg: dict,
        device_mesh: Any,
        cp_mesh: Any,
        tp_mesh: Any,
        cp_size: int,
        enable_seq_packing: bool,
    ):
        self.cfg = cfg
        self.device_mesh = device_mesh
        self.cp_mesh = cp_mesh
        self.tp_mesh = tp_mesh
        self.cp_size = cp_size
        self.enable_seq_packing = enable_seq_packing

    def __call__(
        self,
        logits: torch.Tensor,
        processed_inputs: Any,
    ) -> torch.Tensor:
        """Process value model outputs.

        Args:
            logits: Raw output from value model [batch_size, seq_len, 1] or [batch_size, seq_len]
            processed_inputs: Processed input batch

        Returns:
            Value predictions [batch_size, seq_len]
        """
        # Value model outputs a single scalar per token
        # Squeeze the last dimension if present
        if logits.dim() == 3 and logits.size(-1) == 1:
            values = logits.squeeze(-1)
        else:
            values = logits

        # Handle context parallelism if enabled
        if self.cp_size > 1:
            # Gather values across context parallel dimension
            from torch.distributed._tensor import Replicate

            values = values.redistribute(
                placements=[Replicate()] * len(self.device_mesh.mesh_dim_names)
            )
            values = values.to_local()

        return values


class ValueLossPostProcessor:
    """Post-processor for value function training with loss computation."""

    def __init__(
        self,
        loss_fn: LossFunction,
        cfg: dict,
        device_mesh: Any,
        cp_mesh: Any,
        tp_mesh: Any,
        cp_size: int,
        dp_size: int,
        enable_seq_packing: bool,
    ):
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.device_mesh = device_mesh
        self.cp_mesh = cp_mesh
        self.tp_mesh = tp_mesh
        self.cp_size = cp_size
        self.dp_size = dp_size
        self.enable_seq_packing = enable_seq_packing

    def __call__(
        self,
        logits: torch.Tensor,
        processed_inputs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Process value model outputs and compute loss.

        Args:
            logits: Raw output from value model [batch_size, seq_len, 1] or [batch_size, seq_len]
            processed_inputs: Processed input batch containing target values

        Returns:
            Tuple of (loss, loss_metrics)
            The loss is used both for logging and for backward() call in automodel_forward_backward.
        """
        # Extract values (squeeze if needed)
        if logits.dim() == 3 and logits.size(-1) == 1:
            values = logits.squeeze(-1)
        else:
            values = logits

        # Handle context parallelism if enabled
        if self.cp_size > 1:
            from torch.distributed._tensor import Replicate

            values = values.redistribute(
                placements=[Replicate()] * len(self.device_mesh.mesh_dim_names)
            )
            values = values.to_local()

        # Get original batch data
        batch_data = processed_inputs.orig_batch

        # Compute loss using the provided loss function
        loss, loss_metrics = self.loss_fn.compute_loss(values=values, **batch_data)

        # Track number of valid samples
        mask = batch_data.get("mask", torch.ones_like(values))
        num_valid_samples = mask.sum().item()
        loss_metrics["num_valid_samples"] = num_valid_samples

        return loss, loss_metrics


def get_runtime_env_for_value_worker(worker_type: str) -> dict:
    """Get runtime environment for value worker."""
    from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker

    # Reuse policy worker runtime env
    return get_runtime_env_for_policy_worker("dtensor_policy_worker_v2")


@ray.remote(
    runtime_env=get_runtime_env_for_value_worker("dtensor_value_worker_v2")
)  # pragma: no cover
class DTensorValueWorkerV2(AbstractPolicyWorker):
    def __repr__(self) -> str:
        """Customizes the actor's prefix in the Ray logs."""
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: ValueConfig,
        tokenizer: AutoTokenizer,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        **kwargs: Any,
    ):
        """Initialize the DTensorValueWorkerV2.

        Note: Value models don't need a reference model since they don't compute KL divergence.
        """
        # Apply patches
        apply_transformer_engine_patch()
        apply_torch_aten_alias_tensor_patch()

        # Store configuration and tokenizer
        self.cfg = config
        self.tokenizer = tokenizer
        self.lora_enabled = (
            config["dtensor_cfg"].get("lora_cfg", {}).get("enabled", False)
        )

        # Ensure reward model config is set for value models
        if (
            "reward_model_cfg" not in config
            or not config["reward_model_cfg"]["enabled"]
        ):
            # Value models use the reward model architecture but predict values instead
            config["reward_model_cfg"] = {
                "enabled": True,
                "reward_model_type": "regression",  # Value is a regression task
            }

        print(f"Initializing DTensorValueWorkerV2")

        # Initialize checkpoint manager
        self.checkpoint_manager: Optional[AutomodelCheckpointManager] = None

        # Validate configuration and prepare runtime settings
        runtime_config = validate_and_prepare_config(
            config=config,
            processor=None,  # Value models don't use vision processors
            rank=0,  # Temporary, will be updated after distributed init
        )

        # Set up distributed environment
        distributed_manager = setup_distributed(
            config=config,
            runtime_config=runtime_config,
        )

        # Set instance attributes from distributed manager
        self.rank = torch.distributed.get_rank()
        self.device_mesh = distributed_manager.device_mesh
        self.dp_cp_mesh = self.device_mesh["dp_cp"]
        self.dp_mesh = self.device_mesh["dp"]
        self.tp_mesh = self.device_mesh["tp"]
        self.cp_mesh = self.device_mesh["cp"]
        self.moe_mesh = distributed_manager.moe_mesh
        self.dp_size = distributed_manager.dp_size
        self.tp_size = distributed_manager.tp_size
        self.cp_size = distributed_manager.cp_size

        # Initialize checkpoint manager
        self._init_checkpoint_manager(
            config_updates={
                "model_repo_id": config["model_name"],
                "dequantize_base_checkpoint": config.get(
                    "dequantize_base_checkpoint", False
                ),
                "is_peft": self.lora_enabled,
            },
        )

        # Set up model and optimizer
        model_and_optimizer_state = setup_model_and_optimizer(
            config=config,
            tokenizer=tokenizer,
            runtime_config=runtime_config,
            distributed_manager=distributed_manager,
            checkpoint_manager=self.checkpoint_manager,
            is_vlm=False,  # Value models don't use vision
            init_optimizer=init_optimizer,
            weights_path=weights_path,
            optimizer_path=optimizer_path,
        )

        # Set instance attributes from model and optimizer state
        (
            self.model,
            self.model_state_dict_keys,
            self.optimizer,
            self.scheduler,
            self.is_hf_model,
            self.is_moe_model,
            self._is_reward_model,
            self.model_class,
            self.model_config,
            self.peft_config,
            self.autocast_enabled,
        ) = model_and_optimizer_state

        # Set instance attributes from runtime config
        (
            self.model_class,
            self.model_config,
            self.hf_config_overrides,
            self.allow_flash_attn_args,
            self.attn_impl,
            self.dtype,
            self.enable_seq_packing,
            self.max_grad_norm,
            self.cpu_offload,
            self.offload_optimizer_for_logprob,
            self.is_generation_colocated,
            _runtime_is_reward_model,
        ) = runtime_config

    @wrap_with_nvtx_name("dtensor_value_worker_v2/train")
    def train(
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the value function on a batch of data with a given loss function."""
        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=self.dp_mesh.get_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        # Validate sequence dimension
        sequence_dim, _ = check_sequence_dim(data)

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            self.model.train()

        # Create loss post-processor
        loss_post_processor = ValueLossPostProcessor(
            loss_fn=loss_fn,
            cfg=self.cfg,
            device_mesh=self.device_mesh,
            cp_mesh=self.cp_mesh,
            tp_mesh=self.tp_mesh,
            cp_size=self.cp_size,
            dp_size=self.dp_size,
            enable_seq_packing=self.enable_seq_packing,
        )

        # Create train context factory
        def train_context_fn(processed_inputs):
            return get_train_context(
                cp_size=self.cp_size,
                cp_mesh=self.cp_mesh,
                cp_buffers=processed_inputs.cp_buffers,
                sequence_dim=sequence_dim,
                dtype=self.dtype,
                autocast_enabled=self.autocast_enabled,
            )

        # Setup cache clearing callback if configured
        empty_cache_steps = self.cfg.get("dtensor_cfg", {}).get(
            "clear_cache_every_n_steps"
        )
        if empty_cache_steps:
            warnings.warn(
                f"Emptying cache every {empty_cache_steps} microbatches; doing so unnecessarily would incur a large performance overhead.",
            )

        def on_microbatch_start(mb_idx):
            if empty_cache_steps and mb_idx % empty_cache_steps == 0:
                torch.cuda.empty_cache()

        with ctx:
            data = data.to("cuda")

            losses = []
            all_mb_metrics = []
            for gb_idx in range(num_global_batches):
                # Process global batch
                gb_result = process_global_batch(
                    data,
                    loss_fn,
                    self.dp_mesh.get_group(),
                    batch_idx=gb_idx,
                    batch_size=local_gbs,
                )
                batch = gb_result["batch"]
                global_valid_seqs = gb_result["global_valid_seqs"]
                global_valid_toks = gb_result["global_valid_toks"]

                self.optimizer.zero_grad()

                # Get microbatch iterator
                processed_iterator, iterator_len = get_microbatch_iterator(
                    batch,
                    self.cfg,
                    mbs,
                    self.dp_mesh,
                    tokenizer=self.tokenizer,
                    cp_size=self.cp_size,
                )

                # Use automodel_forward_backward for the training loop
                mb_results = automodel_forward_backward(
                    model=self.model,
                    cfg=self.cfg,
                    data_iterator=processed_iterator,
                    post_processing_fn=loss_post_processor,
                    forward_only=eval_mode,
                    is_reward_model=True,  # Value models use reward model architecture
                    allow_flash_attn_args=False,  # Typically False for value models
                    global_valid_seqs=global_valid_seqs,
                    global_valid_toks=global_valid_toks,
                    sequence_dim=sequence_dim,
                    dp_size=self.dp_size,
                    cp_size=self.cp_size,
                    num_global_batches=num_global_batches,
                    train_context_fn=train_context_fn,
                    num_valid_microbatches=iterator_len,
                    on_microbatch_start=on_microbatch_start,
                )

                # Extract losses and metrics from results
                mb_losses = []
                for mb_idx, (loss, loss_metrics) in enumerate(mb_results):
                    if mb_idx < iterator_len:
                        num_valid_samples = loss_metrics["num_valid_samples"]
                        loss_metrics["lr"] = self.optimizer.param_groups[0]["lr"]
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()

                        if num_valid_samples > 0:
                            mb_losses.append(loss.item())
                            all_mb_metrics.append(loss_metrics)

                grad_norm: Optional[float | torch.Tensor] = None
                if not eval_mode:
                    grad_norm = scale_grads_and_clip_grad_norm(
                        self.max_grad_norm,
                        [self.model],
                        norm_type=2.0,
                        pp_enabled=False,
                        device_mesh=self.device_mesh,
                        moe_mesh=self.moe_mesh,
                        ep_axis_name="ep"
                        if self.moe_mesh is not None
                        and "ep" in self.moe_mesh.mesh_dim_names
                        else None,
                        pp_axis_name=None,
                        foreach=True,
                        num_label_tokens=1,
                        dp_group_size=self.dp_size * self.cp_size,
                    )
                    grad_norm = torch.tensor(
                        grad_norm, device="cpu", dtype=torch.float32
                    )

                    # Update parameters
                    self.optimizer.step()

                losses.append(torch.tensor(mb_losses).sum().item())

            # Release gradient memory
            self.optimizer.zero_grad()
            # Increment scheduler
            if not eval_mode:
                self.scheduler.step()
            # Clear cache
            torch.cuda.empty_cache()

            # Aggregate training statistics
            metrics = aggregate_training_statistics(
                losses=losses,
                all_mb_metrics=all_mb_metrics,
                grad_norm=grad_norm,
                dp_group=self.dp_mesh.get_group(),
                dtype=self.dtype,
            )

            return metrics

    @wrap_with_nvtx_name("dtensor_value_worker_v2/get_values")
    def get_values(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[ValueOutputSpec]:
        """Get value predictions for a batch of data."""
        value_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg.get("logprob_batch_size", self.cfg["train_micro_batch_size"])
        )

        # Validate sequence dimension
        sequence_dim, seq_dim_size = check_sequence_dim(data)

        all_values = []
        self.model.eval()

        # Create value post-processor
        value_post_processor = ValuePostProcessor(
            cfg=self.cfg,
            device_mesh=self.device_mesh,
            cp_mesh=self.cp_mesh,
            tp_mesh=self.tp_mesh,
            cp_size=self.cp_size,
            enable_seq_packing=self.enable_seq_packing,
        )

        with torch.no_grad():
            data.to("cuda")
            # Get microbatch iterator
            processed_iterator, iterator_len = get_microbatch_iterator(
                data,
                self.cfg,
                value_batch_size,
                self.dp_mesh,
                tokenizer=self.tokenizer,
                cp_size=self.cp_size,
            )

            for batch_idx, processed_mb in enumerate(processed_iterator):
                processed_inputs = processed_mb.processed_inputs

                with get_train_context(
                    cp_size=self.cp_size,
                    cp_mesh=self.cp_mesh,
                    cp_buffers=processed_inputs.cp_buffers,
                    sequence_dim=sequence_dim,
                    dtype=self.dtype,
                    autocast_enabled=self.autocast_enabled,
                ):
                    # Use forward_with_post_processing_fn for forward pass
                    values, _metrics, _ = forward_with_post_processing_fn(
                        model=self.model,
                        cfg=self.cfg,
                        post_processing_fn=value_post_processor,
                        processed_mb=processed_mb,
                        is_reward_model=True,  # Value models use reward model architecture
                        allow_flash_attn_args=False,
                        sequence_dim=sequence_dim,
                    )

                # Skip dummy batches
                if batch_idx >= iterator_len:
                    continue

                all_values.append(values)

        # Concatenate all batches
        return_data = BatchedDataDict[ValueOutputSpec]()

        all_values_padded = []
        for val in all_values:
            padding_needed = seq_dim_size - val.shape[1]
            if padding_needed > 0:
                val = torch.nn.functional.pad(
                    val, (0, padding_needed), mode="constant", value=0.0
                )
            all_values_padded.append(val)
        return_data["values"] = torch.cat(all_values_padded, dim=0).cpu()

        return return_data

    @wrap_with_nvtx_name("dtensor_value_worker_v2/prepare_for_training")
    def prepare_for_training(self, *args, **kwargs) -> None:
        """Prepare for training by loading model and optimizer to GPU."""
        if not self.cpu_offload:
            self.move_to_cuda(self.model)
        else:
            self.model = self.move_buffer_to_device(self.model, "cuda")

        self.model.train()
        if self.optimizer is not None and not self.cpu_offload:
            self.move_optimizer_to_device("cuda")

        torch.cuda.empty_cache()

    def move_optimizer_to_device(self, device: str | torch.device) -> None:
        """Move optimizer state to specified device."""
        from torch.distributed.tensor import DTensor

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, (DTensor, torch.Tensor)):
                    state[k] = v.to(device)

    def move_to_device(self, model: nn.Module, device: str | torch.device) -> nn.Module:
        """Move model to specified device."""
        model = self.move_buffer_to_device(model, device)
        return model.to(device)

    def move_buffer_to_device(
        self, model: nn.Module, device: str | torch.device
    ) -> nn.Module:
        """Move model buffers to specified device."""
        for v in model.buffers():
            torch.utils.swap_tensors(v, v.to(device))
        return model

    def move_to_cuda(self, model: torch.nn.Module) -> torch.nn.Module:
        """Move model to CUDA."""
        model = self.move_to_device(model, "cuda")
        gc.collect()
        torch.cuda.empty_cache()
        return model

    def move_to_cpu(self, model: torch.nn.Module) -> torch.nn.Module:
        """Move model to CPU."""
        model = self.move_to_device(model, "cpu")
        gc.collect()
        torch.cuda.empty_cache()
        return model

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        checkpointing_cfg: Optional[CheckpointingConfig] = None,
    ) -> None:
        """Save a checkpoint of the value model."""
        self.checkpoint_manager.save_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer,
            optimizer_path=optimizer_path,
            scheduler=self.scheduler,
            tokenizer=self.tokenizer if tokenizer_path else None,
            tokenizer_path=tokenizer_path,
            checkpointing_cfg=checkpointing_cfg,
            lora_enabled=self.lora_enabled,
            peft_config=self.peft_config,
        )

    def load_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
    ) -> None:
        """Load a checkpoint into the value model."""
        self.checkpoint_manager.load_checkpoint(
            model=self.model,
            weights_path=weights_path,
            optimizer=self.optimizer,
            optimizer_path=optimizer_path,
            scheduler=self.scheduler,
        )

    def _init_checkpoint_manager(
        self,
        config_updates: Optional[dict[str, Any]] = None,
        checkpoint_root: Optional[str] = None,
    ) -> None:
        """Initialize the AutomodelCheckpointManager for this worker."""
        if self.checkpoint_manager is None:
            self.checkpoint_manager = AutomodelCheckpointManager(
                dp_mesh=self.dp_mesh,
                tp_mesh=self.tp_mesh,
                model_state_dict_keys=getattr(self, "model_state_dict_keys", None),
                moe_mesh=self.moe_mesh,
            )
            self.checkpoint_manager.init_checkpointer(
                config_updates=config_updates,
                checkpoint_root=checkpoint_root,
            )
