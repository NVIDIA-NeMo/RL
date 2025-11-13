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

from typing import Any, Optional

import torch
from nemo_automodel.components.distributed.cp_utils import (
    create_context_parallel_ctx,
    get_train_context,
)
from nemo_automodel.components.training.utils import (
    scale_grads_and_clip_grad_norm,
)
from torch import nn
from torch.distributed.tensor import DTensor, Shard

from nemo_rl.algorithms.interfaces import LossFunction
from nemo_rl.algorithms.loss_functions import SequencePackingLossWrapper
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import get_logprobs_from_vocab_parallel_logits


def setup_train_loop(
    data: BatchedDataDict[Any],
    gbs: int,
    dp_size: int,
    dp_mesh: Any,
) -> dict[str, Any]:
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
    is_hf_model: bool,
    is_moe_model: bool,
    eval_mode: bool,
    apply_temperature_fn,
) -> tuple[torch.Tensor, dict[str, Any]]:
    outputs = model_forward(
        model,
        processed_inputs,
        cp_size,
        cp_mesh,
        is_reward_model,
        allow_flash_attn_args,
        is_hf_model,
        is_moe_model,
        dtype,
    )

    loss, loss_metrics = process_output_for_train(
        outputs,
        model,
        mb,
        loss_fn,
        global_valid_seqs,
        global_valid_toks,
        processed_inputs,
        cp_size,
        cp_mesh,
        device_mesh,
        enable_seq_packing,
        eval_mode,
        apply_temperature_fn,
    )

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
    grad_norm = grad_norm.full_tensor() if isinstance(grad_norm, DTensor) else grad_norm
    grad_norm = (
        torch.tensor(grad_norm)
        if not isinstance(grad_norm, torch.Tensor)
        else grad_norm
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
    # Release gradient memory before rollouts
    optimizer.zero_grad()

    # Increment scheduler after all batches in rollout are processed
    if not eval_mode and scheduler is not None:
        scheduler.step()

    # Dynamic batch and sequence dims causes a lot of fragmentation, so clear
    # the memory allocator before moving on
    torch.cuda.empty_cache()


def model_forward(
    model: nn.Module,
    processed_inputs: dict[str, Any],
    cp_size: int,
    cp_mesh: Any,
    is_reward_model: bool,
    allow_flash_attn_args: bool,
    is_hf_model: bool,
    is_moe_model: bool,
    dtype: torch.dtype,
) -> Any:
    sequence_dim = 1

    input_ids = processed_inputs["input_ids"]
    attention_mask = processed_inputs["attention_mask"]
    position_ids = processed_inputs["position_ids"]
    flash_attn_kwargs = processed_inputs["flash_attn_kwargs"]
    vlm_kwargs = processed_inputs["vlm_kwargs"]
    cp_buffers = processed_inputs["cp_buffers"]

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
        with torch.autocast(device_type="cuda", dtype=dtype):
            model_args = dict(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                flash_attn_kwargs=flash_attn_kwargs,
                **vlm_kwargs,
            )
            if is_moe_model and not is_hf_model:
                padding_mask = ~attention_mask if attention_mask is not None else None
                model_args["padding_mask"] = padding_mask

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
            if "padding_mask" in model_args and model_args.get("padding_mask") is None:
                del model_args["padding_mask"]

            outputs = model(**model_args)

    return outputs


def _process_logits(
    outputs: Any,
    model: nn.Module,
    apply_temperature_fn,
) -> torch.Tensor:
    # Get logits
    if isinstance(outputs, (torch.Tensor, DTensor)):
        # custom models (e.g., those coming from AutoModel) can output logits directly
        logits = outputs
    else:
        if not hasattr(outputs, "logits"):
            logits = model.lm_head(outputs.last_hidden_state)
        else:
            logits = outputs.logits

    # Apply temperature scaling
    logits = apply_temperature_fn(logits)

    return logits


def _handle_context_parallel_sharding(
    logits: torch.Tensor,
    seq_index: torch.Tensor,
    cp_mesh: Any,
    device_mesh: Any,
    sequence_dim: int = 1,
    mb: Optional[BatchedDataDict[Any]] = None,
    cp_buffers: Optional[list] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    seq_index_dtensor = (
        DTensor.from_local(
            seq_index,
            device_mesh=cp_mesh,
            placements=[Shard(1)],
        )
        .full_tensor()
        .squeeze(0)
    )

    # Optionally shard microbatch tensors (for training)
    if mb is not None and cp_buffers is not None:
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

    # Shard logits for CP+TP
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

    return logits, seq_index_dtensor


def process_output_for_train(
    outputs: Any,
    model: nn.Module,
    mb: BatchedDataDict[Any],
    loss_fn: LossFunction,
    global_valid_seqs: torch.Tensor,
    global_valid_toks: torch.Tensor,
    processed_inputs: dict[str, Any],
    cp_size: int,
    cp_mesh: Any,
    device_mesh: Any,
    enable_seq_packing: bool,
    eval_mode: bool,
    apply_temperature_fn,
) -> tuple[torch.Tensor, dict[str, Any]]:
    sequence_dim = 1
    flash_attn_kwargs = processed_inputs["flash_attn_kwargs"]
    cp_buffers = processed_inputs["cp_buffers"]
    seq_index = processed_inputs["seq_index"]

    with get_train_context(False, False, None)():
        # Process logits from model outputs
        logits = _process_logits(outputs, model, apply_temperature_fn)
        del outputs

        # Handle context parallel sharding if needed
        if cp_size > 1:
            logits, _ = _handle_context_parallel_sharding(
                logits,
                seq_index,
                cp_mesh,
                device_mesh,
                sequence_dim,
                mb=mb,
                cp_buffers=cp_buffers,
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


def process_outputs_for_logprobs(
    outputs: Any,
    model: nn.Module,
    mb: BatchedDataDict[Any],
    processed_inputs: dict[str, Any],
    input_ids: torch.Tensor,
    cp_size: int,
    cp_mesh: Any,
    device_mesh: Any,
    enable_seq_packing: bool,
    apply_temperature_fn,
    logprob_chunk_size: Optional[int] = None,
) -> torch.Tensor:
    sequence_dim = 1
    cp_buffers = processed_inputs["cp_buffers"]
    seq_index = processed_inputs["seq_index"]
    seq_len = processed_inputs["seq_len"]

    with get_train_context(False, False, None)():
        # Process logits from model outputs
        logits = _process_logits(outputs, model, apply_temperature_fn)
        del outputs

        if cp_size > 1:
            # Shard logits for CP (without modifying mb)
            logits, seq_index_tensor = _handle_context_parallel_sharding(
                logits,
                seq_index,
                cp_mesh,
                device_mesh,
                sequence_dim,
            )

            # For logprob extraction, we need input_ids as DTensor
            input_ids_dtensor = DTensor.from_local(
                input_ids,
                device_mesh=cp_mesh,
                placements=[Shard(sequence_dim)],
            )

            token_logprobs = get_logprobs_from_vocab_parallel_logits(
                logits,
                input_ids_dtensor,
                seq_index_tensor,
                chunk_size=logprob_chunk_size,
            )

            assert token_logprobs.shape[1] == seq_len - 1
        else:
            if isinstance(logits, DTensor):
                token_logprobs = get_logprobs_from_vocab_parallel_logits(
                    logits,
                    input_ids,
                    chunk_size=logprob_chunk_size,
                )
            else:
                if logprob_chunk_size is not None:
                    logits_seq_len = int(logits.shape[1])
                    num_chunks = (
                        logits_seq_len + logprob_chunk_size - 1
                    ) // logprob_chunk_size
                    chunked_log_probs = []
                    for chunk_idx in range(num_chunks):
                        chunk_start = chunk_idx * logprob_chunk_size
                        chunk_end = min(
                            logits_seq_len,
                            (chunk_idx + 1) * logprob_chunk_size,
                        )
                        chunk_logits = logits[:, chunk_start:chunk_end, :].to(
                            torch.float32
                        )
                        log_probs = torch.nn.functional.log_softmax(
                            chunk_logits, dim=-1
                        )
                        chunked_log_probs.append(log_probs)
                    log_probs = torch.cat(chunked_log_probs, dim=1)
                    del chunked_log_probs
                else:
                    logits = logits.to(torch.float32)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

                # Extract logprobs for each token in the sequence by gathering the logprob
                # corresponding to the next token at each position
                next_tokens = input_ids[:, 1:]
                log_probs = log_probs[:, :-1]
                token_logprobs = log_probs.gather(
                    dim=-1, index=next_tokens.unsqueeze(-1)
                ).squeeze(-1)
                del log_probs

        del logits

    # Prepend zeros to maintain sequence length
    token_logprobs = torch.cat(
        [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
    )

    return token_logprobs


def process_outputs_for_topk(
    outputs: Any,
    model: nn.Module,
    mb: BatchedDataDict[Any],
    processed_inputs: dict[str, Any],
    k: int,
    cp_size: int,
    cp_mesh: Any,
    device_mesh: Any,
    tp_mesh: Any,
    enable_seq_packing: bool,
    apply_temperature_fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    from nemo_rl.distributed.model_utils import (
        allgather_cp_sharded_tensor,
        distributed_vocab_topk,
    )

    sequence_dim = 1
    seq_index = processed_inputs["seq_index"]

    with get_train_context(False, False, None)():
        # Process logits from model outputs
        logits = _process_logits(outputs, model, apply_temperature_fn)
        del outputs

        if cp_size > 1:
            # Shard logits for CP (without modifying mb)
            logits, seq_index_tensor = _handle_context_parallel_sharding(
                logits,
                seq_index,
                cp_mesh,
                device_mesh,
                sequence_dim,
            )

            # Deal with TP first
            local_logits = logits.to_local()  # [B, S_cp, V_tp]

            tp_group = tp_mesh.get_group()
            tp_rank = torch.distributed.get_rank(tp_group)
            V_local = int(local_logits.shape[-1])
            vocab_start_index = tp_rank * V_local
            vocab_end_index = (tp_rank + 1) * V_local

            vals, idx = distributed_vocab_topk(
                local_logits,
                k=k,
                tp_group=tp_group,
                vocab_start_index=vocab_start_index,
                vocab_end_index=vocab_end_index,
            )
            # [B, S_cp, k]

            cp_group = cp_mesh.get_group()

            vals = allgather_cp_sharded_tensor(vals, cp_group, seq_dim=sequence_dim)
            idx = allgather_cp_sharded_tensor(idx, cp_group, seq_dim=sequence_dim)
            # [B, S, k]
        else:
            # Compute top-k over full sequence length (do not drop last position)
            if isinstance(logits, DTensor):
                local_logits = logits.to_local()  # [B, S, V_local]
                tp_group = tp_mesh.get_group()
                tp_rank = torch.distributed.get_rank(tp_group)
                V_local = int(local_logits.shape[-1])
                vocab_start_index = tp_rank * V_local
                vocab_end_index = (tp_rank + 1) * V_local

                vals, idx = distributed_vocab_topk(
                    local_logits,
                    k=k,
                    tp_group=tp_group,
                    vocab_start_index=vocab_start_index,
                    vocab_end_index=vocab_end_index,
                )
            else:
                full_logits = logits.to(torch.float32)
                vals, idx = torch.topk(full_logits, k=k, dim=-1)

        del logits

    return vals, idx
