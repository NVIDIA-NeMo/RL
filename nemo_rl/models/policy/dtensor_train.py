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

from contextlib import nullcontext
from typing import Any, Optional

import torch
from nemo_automodel.components.distributed.cp_utils import (
    create_context_parallel_ctx,
    get_train_context,
)
from torch import nn
from torch.distributed.tensor import DTensor, Shard

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import SequencePackingLossWrapper
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def setup_train_loop(
    data: BatchedDataDict[Any],
    gbs: int,
    dp_size: int,
    dp_mesh: Any,
) -> dict[str, Any]:
    """Setup training loop parameters and validate data.

    Args:
        data: Batched data dictionary
        gbs: Global batch size
        dp_size: Data parallel size
        dp_mesh: Data parallel mesh for reductions

    Returns:
        Dictionary containing setup parameters
    """
    local_gbs = gbs // dp_size
    total_dataset_size = torch.tensor(data.size, device="cuda")
    torch.distributed.all_reduce(
        total_dataset_size,
        op=torch.distributed.ReduceOp.SUM,
        group=dp_mesh.get_group(),
    )
    num_global_batches = int(total_dataset_size.item()) // gbs

    # dim 1 is always assumed to be the sequence dim, sanity check this here
    sequence_dim = 1
    seq_dim_size = data.get("input_ids").shape[sequence_dim]
    for k, v in data.items():
        if torch.is_tensor(v) and len(v.shape) > 1:
            assert v.shape[sequence_dim] == seq_dim_size, (
                f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
            )

    return {
        "local_gbs": local_gbs,
        "num_global_batches": num_global_batches,
        "sequence_dim": sequence_dim,
    }


def forward_backward(
    model: nn.Module,
    mb: BatchedDataDict[Any],
    loss_fn: LossFunction,
    global_valid_seqs: torch.Tensor,
    global_valid_toks: torch.Tensor,
    processed_inputs: dict[str, Any],
    dtype: torch.dtype,
    cp_size: int,
    cp_mesh: Any,
    device_mesh: Any,
    enable_seq_packing: bool,
    is_reward_model: bool,
    allow_flash_attn_args: bool,
    eval_mode: bool,
    apply_temperature_fn,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Perform forward pass, loss computation, and backward pass.

    Args:
        model: The model to train
        mb: Microbatch data
        loss_fn: Loss function to use
        global_valid_seqs: Number of valid sequences across all ranks
        global_valid_toks: Number of valid tokens across all ranks
        processed_inputs: Processed inputs from process_microbatch
        dtype: Model dtype for autocast
        cp_size: Context parallel size
        cp_mesh: Context parallel mesh
        device_mesh: Full device mesh
        enable_seq_packing: Whether sequence packing is enabled
        is_reward_model: Whether this is a reward model
        allow_flash_attn_args: Whether model supports flash_attn_kwargs
        eval_mode: Whether in evaluation mode
        apply_temperature_fn: Function to apply temperature scaling to logits

    Returns:
        Tuple of (loss, loss_metrics)
    """
    sequence_dim = 1

    input_ids = processed_inputs["input_ids"]
    attention_mask = processed_inputs["attention_mask"]
    position_ids = processed_inputs["position_ids"]
    flash_attn_kwargs = processed_inputs["flash_attn_kwargs"]
    vlm_kwargs = processed_inputs["vlm_kwargs"]
    cp_buffers = processed_inputs["cp_buffers"]
    seq_index = processed_inputs["seq_index"]
    seq_len = processed_inputs["seq_len"]

    context_parallel_ctx = None
    if cp_size > 1:
        # Create context parallel context
        context_parallel_ctx = create_context_parallel_ctx(
            cp_mesh=cp_mesh,
            cp_buffers=cp_buffers,
            cp_seq_dims=[sequence_dim] * len(cp_buffers),
            cp_no_restore_buffers=set(cp_buffers),
        )

    with get_train_context(False, False, context_parallel_ctx)():
        with nullcontext():
            model_args = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                padding_mask=~attention_mask if attention_mask is not None else None,
                position_ids=position_ids,
                use_cache=False,
                flash_attn_kwargs=flash_attn_kwargs,
                **vlm_kwargs,
            )

            if is_reward_model:
                # `flash_attn_kwarg` is not supported for `LlamaForSequenceClassification`.
                # Note that it should be empty anyway since sequence packing
                # is not supported for reward models.
                assert not flash_attn_kwargs
                del model_args["flash_attn_kwargs"]
            # remove flash_attn_kwargs if there are multimodal kwargs
            if len(vlm_kwargs) > 0:
                del model_args["flash_attn_kwargs"]

            if not allow_flash_attn_args and "flash_attn_kwargs" in model_args:
                del model_args["flash_attn_kwargs"]

            # Remove None attention_mask padding_mask if present
            if model_args.get("attention_mask") is None:
                del model_args["attention_mask"]
            if model_args.get("padding_mask") is None:
                del model_args["padding_mask"]

            outputs = model(**model_args)

        # Get logprobs
        if isinstance(outputs, (torch.Tensor, DTensor)):
            # custom models (e.g., those coming from AutoModel) can output logits directly
            logits = outputs
        else:
            if not hasattr(outputs, "logits"):
                logits = model.lm_head(outputs.last_hidden_state)
            else:
                logits = outputs.logits
        del outputs

        # Apply temperature scaling
        logits = apply_temperature_fn(logits)

        if cp_size > 1:
            seq_index_dtensor = (
                DTensor.from_local(
                    seq_index,
                    device_mesh=cp_mesh,
                    placements=[Shard(1)],
                )
                .full_tensor()
                .squeeze(0)
            )

            mb["seq_index"] = seq_index_dtensor

            for tensor_name in mb:
                current_tensor = mb[tensor_name]
                for buffer in cp_buffers:
                    if current_tensor is buffer:
                        assert type(current_tensor) == torch.Tensor, (
                            f"tensor {tensor_name} is not a tensor"
                        )
                        mb[tensor_name] = DTensor.from_local(
                            current_tensor,
                            device_mesh=cp_mesh,
                            placements=[Shard(sequence_dim)],
                        )
                        break

            if isinstance(logits, DTensor):
                # Must be tp sharded
                assert (
                    logits.device_mesh.ndim == 1
                    and logits.device_mesh.mesh_dim_names[0] == "tp"
                ), "logits must be tp sharded"

                # CP is implicitly sharded on the seq dim, so we need to redistribute to the tp dim
                logits = DTensor.from_local(
                    logits.to_local(),
                    device_mesh=device_mesh[("cp", "tp")],
                    placements=[Shard(sequence_dim), Shard(-1)],
                )
            else:
                logits = DTensor.from_local(
                    logits,
                    device_mesh=device_mesh[("cp", "tp")],
                    placements=[Shard(sequence_dim), Shard(-1)],
                )

        if enable_seq_packing:
            loss_fn_ = SequencePackingLossWrapper(
                loss_fn=loss_fn,
                cu_seqlens_q=flash_attn_kwargs.cu_seqlens_q,
                cu_seqlens_q_padded=flash_attn_kwargs.cu_seqlens_q,
            )
        else:
            loss_fn_ = loss_fn

        loss, loss_metrics = loss_fn_(
            logits,
            mb,
            global_valid_seqs,
            global_valid_toks,
        )
        del logits

        # Backward pass
        if not eval_mode:
            ## NOTE: invalid samples should be multiplied
            ## by zero in the loss function to prevent them
            ## from affecting the gradient calculation

            # when FSDP reduces the gradients over the DP dim, they're automatically averaged
            # but we want to sum them so we cancel out the average here
            # loss *= self.dp_size * self.cp_size
            loss.backward()

    return loss, loss_metrics


def optimizer_step(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    max_grad_norm: Optional[float],
    device_mesh: Any,
    moe_mesh: Any,
    dp_size: int,
    cp_size: int,
) -> Optional[torch.Tensor]:
    """Perform gradient clipping and optimizer step.

    Args:
        optimizer: Optimizer to step
        model: Model being trained
        max_grad_norm: Maximum gradient norm for clipping
        device_mesh: Full device mesh
        moe_mesh: MoE mesh (if applicable)
        dp_size: Data parallel size
        cp_size: Context parallel size

    Returns:
        Gradient norm (or None if not computed)
    """
    from nemo_automodel.components.training.utils import (
        scale_grads_and_clip_grad_norm,
    )

    grad_norm = scale_grads_and_clip_grad_norm(
        max_grad_norm,
        [model],
        norm_type=2.0,
        pp_enabled=False,
        device_mesh=device_mesh,
        moe_mesh=moe_mesh,
        ep_axis_name="ep"
        if moe_mesh is not None and "ep" in moe_mesh.mesh_dim_names
        else None,
        pp_axis_name=None,
        foreach=True,
        num_label_tokens=1,
        dp_group_size=dp_size * cp_size,
    )
    grad_norm = grad_norm.detach().cpu().float()

    if not torch.isfinite(grad_norm):
        print(f"WARN: grad_norm is not finite: {grad_norm}")
        optimizer.zero_grad()
    else:
        # Update parameters
        optimizer.step()

    return grad_norm


def cleanup_after_training(
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    eval_mode: bool,
) -> None:
    """Cleanup after all training batches are processed.

    Args:
        optimizer: Optimizer to zero gradients
        scheduler: Learning rate scheduler to step (if not in eval mode)
        eval_mode: Whether in evaluation mode
    """
    # Release gradient memory before rollouts
    optimizer.zero_grad()

    # Increment scheduler after all batches in rollout are processed
    if not eval_mode and scheduler is not None:
        scheduler.step()

    # Dynamic batch and sequence dims causes a lot of fragmentation, so clear
    # the memory allocator before moving on
    torch.cuda.empty_cache()
