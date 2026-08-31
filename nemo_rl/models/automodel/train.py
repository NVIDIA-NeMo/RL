# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Training utilities for automodel (DTensor-based) policy workers.

This module provides post-processor classes and forward/backward functions
that follow the same pattern as nemo_rl/models/megatron/train.py.

Key differences from megatron approach:
- Post-processors compute results directly (no callable return pattern)
- forward_with_post_processing_fn calls post-processor directly
- automodel_forward_backward uses PyTorch autograd instead of Megatron's pipeline
"""

from collections import defaultdict
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterator, Optional, Tuple, Union

import torch
from nemo_automodel.components.distributed.context_parallel import (
    ContextParallelSharder,
)
from nemo_automodel.components.distributed.tensor_utils import to_local_if_dtensor
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from transformers.models.gemma3.modeling_gemma3 import (
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
)

from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    apply_top_k_top_p,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.algorithms.loss import SequencePackingLossWrapper, prepare_loss_input
from nemo_rl.algorithms.loss.interfaces import LossFunction, LossInputType
from nemo_rl.algorithms.utils import mask_out_neg_inf_logprobs
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    distributed_vocab_topk,
    get_cp_sharded_next_token_logprobs,
    get_logprobs_from_vocab_parallel_logits,
)
from nemo_rl.models.automodel.data import (
    ProcessedInputs,
    ProcessedMicrobatch,
    filter_multimodal_kwargs_for_model,
)
from nemo_rl.models.policy import PolicyConfig

# Union type for any post-processing function
PostProcessingFunction = Union[
    "LossPostProcessor",
    "LogprobsPostProcessor",
    "TopkLogitsPostProcessor",
    "FullLogitsPostProcessor",
    "ScorePostProcessor",
]


@dataclass
class PreparedModelForward:
    """Model inputs and optional Automodel CP state for one microbatch."""

    model_batch: dict[str, Any]
    cp_size: int
    cp_sharder: Optional[ContextParallelSharder]
    model_context_factory: Callable[[], AbstractContextManager[Any]]


def _build_model_batch(
    model: nn.Module,
    processed_inputs: ProcessedInputs,
    *,
    is_reward_model: bool,
    allow_flash_attn_args: bool,
    clone_model_tensors: bool,
) -> dict[str, Any]:
    """Build a model-facing batch from canonical inputs."""
    model_batch: dict[str, Any] = {
        "input_ids": processed_inputs.input_ids,
        "use_cache": False,
    }
    if processed_inputs.attention_mask is not None:
        model_batch["attention_mask"] = processed_inputs.attention_mask
    if processed_inputs.position_ids is not None:
        model_batch["position_ids"] = processed_inputs.position_ids
    if processed_inputs.has_flash_attention:
        model_batch["flash_attn_kwargs"] = processed_inputs.flash_attn_kwargs

    if processed_inputs.is_multimodal:
        model_batch.update(
            filter_multimodal_kwargs_for_model(model, processed_inputs.vlm_kwargs)
        )
        model_batch.pop("flash_attn_kwargs", None)

    is_gemma3 = isinstance(model, Gemma3ForCausalLM) or isinstance(
        model, Gemma3ForConditionalGeneration
    )
    if is_gemma3 and "token_type_ids" not in model_batch:
        model_batch["token_type_ids"] = torch.zeros_like(processed_inputs.input_ids)

    if getattr(getattr(model, "config", None), "model_type", None) == "gemma4":
        if "mm_token_type_ids" not in model_batch:
            model_batch["mm_token_type_ids"] = torch.zeros_like(
                processed_inputs.input_ids
            )

    if is_reward_model or not allow_flash_attn_args:
        model_batch.pop("flash_attn_kwargs", None)

    if clone_model_tensors:
        # Automodel may pad or shard these tensors in place. Keep the loss-side
        # canonical tensors in ProcessedInputs/data_dict untouched under CP.
        for key in (
            "input_ids",
            "position_ids",
            "attention_mask",
            "token_type_ids",
            "mm_token_type_ids",
        ):
            value = model_batch.get(key)
            if isinstance(value, torch.Tensor):
                model_batch[key] = value.clone()
    return model_batch


def prepare_model_forward(
    model: nn.Module,
    processed_inputs: ProcessedInputs,
    *,
    device_mesh: Optional[DeviceMesh],
    cp_size: int,
    padding_token_id: int,
    is_reward_model: bool,
    allow_flash_attn_args: bool,
) -> PreparedModelForward:
    """Prepare the model batch and resolve Automodel CP state when active."""
    model_batch = _build_model_batch(
        model,
        processed_inputs,
        is_reward_model=is_reward_model,
        allow_flash_attn_args=allow_flash_attn_args,
        clone_model_tensors=cp_size > 1,
    )

    if cp_size <= 1:
        return PreparedModelForward(
            model_batch=model_batch,
            cp_size=cp_size,
            cp_sharder=None,
            model_context_factory=nullcontext,
        )

    # Automodel's generic CP prep reads ``batch["labels"]`` unconditionally and
    # shards it alongside input_ids/position_ids (see ``shard_batch_load_balanced``).
    # NeMo-RL computes its own masked losses from the returned logits, so pass an
    # all-ignore placeholder purely to satisfy that contract and drop it again
    # before the forward. ``pop`` tolerates a strategy that consumed it itself.
    model_batch["labels"] = torch.full_like(processed_inputs.input_ids, -100)

    cp_sharder = ContextParallelSharder(
        model,
        device_mesh,
        model_batch,
        padding_token_id=padding_token_id,
        num_chunks=1,
    )
    model_context_factory, model_batch = cp_sharder.shard(model_batch)
    model_batch.pop("labels", None)
    return PreparedModelForward(
        model_batch=model_batch,
        cp_size=cp_size,
        cp_sharder=cp_sharder,
        model_context_factory=model_context_factory,
    )


def model_forward(
    model: nn.Module,
    model_batch: dict[str, Any],
) -> Any:
    """Run a model on an Automodel-prepared batch.

    Args:
        model: The model to run.
        model_batch: Private batch returned by the CP sharder.

    Returns:
        Model-specific forward output.
    """
    return model(**model_batch)


def extract_logits(
    model: nn.Module,
    outputs: Any,
) -> torch.Tensor:
    """Extract logits from model outputs.

    Args:
        model: The model (used for lm_head if needed)
        outputs: Model outputs (can be tensor, DTensor, or object with logits attribute)

    Returns:
        torch.Tensor: Logits tensor
    """
    if isinstance(outputs, (torch.Tensor, DTensor)):
        # Custom models can output logits directly
        return outputs
    elif not hasattr(outputs, "logits"):
        return model.lm_head(outputs.last_hidden_state)
    else:
        return outputs.logits


def apply_temperature_scaling(
    logits: torch.Tensor, sampling_params: Optional[TrainingSamplingParams]
) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: Logits tensor to scale
        sampling_params: Sampling parameters

    Returns:
        torch.Tensor: Temperature-scaled logits
    """
    if sampling_params is not None and sampling_params.temperature != 1.0:
        logits.div_(sampling_params.temperature)
    return logits


def apply_top_k_top_p_filtering_for_local_logits(
    logits: torch.Tensor, sampling_params: Optional[TrainingSamplingParams]
) -> torch.Tensor:
    """Apply top-k and top-p filtering to the non-distributed logits.

    Args:
        logits: Logits tensor to filter
        sampling_params: Sampling parameters

    Returns:
        torch.Tensor: Filtered logits
    """
    if need_top_k_or_top_p_filtering(sampling_params):
        logits, _ = apply_top_k_top_p(
            logits,
            top_k=sampling_params.top_k,
            top_p=sampling_params.top_p,
        )
    return logits


def _cp_gather_logits(
    logits: torch.Tensor | DTensor,
    cp_sharder: ContextParallelSharder,
    seq_dim: int = 1,
) -> torch.Tensor | DTensor:
    """Restore CP-local logits to canonical full-sequence order.

    Keeps a tensor-parallel ``DTensor`` a ``DTensor`` on the same vocabulary
    mesh: only the sequence dimension is reassembled.
    """
    if isinstance(logits, DTensor):
        gathered = cp_sharder.gather_token_tensor(
            logits.to_local(), seq_dim=seq_dim, trim=True
        )
        return DTensor.from_local(
            gathered,
            device_mesh=logits.device_mesh,
            placements=logits.placements,
        )
    return cp_sharder.gather_token_tensor(logits, seq_dim=seq_dim, trim=True)


def forward_with_post_processing_fn(
    model: nn.Module,
    prepared: PreparedModelForward,
    post_processing_fn: PostProcessingFunction,
    processed_mb: ProcessedMicrobatch,
    global_valid_seqs: Optional[torch.Tensor] = None,
    global_valid_toks: Optional[torch.Tensor] = None,
    sampling_params: Optional[TrainingSamplingParams] = None,
    sequence_dim: int = 1,
) -> Tuple[Any, dict[str, Any], ProcessedMicrobatch]:
    """Perform forward pass with pre-processed microbatch and apply post-processing.

    This function takes a pre-processed microbatch (with sequence packing already handled),
    runs the forward step through the model, and applies the post-processing function
    to compute the result.

    Unlike the megatron approach which returns a callable, this directly computes
    and returns the result since automodel uses PyTorch autograd.

    Args:
        model: The model to run forward pass on
        prepared: Per-microbatch model batch, CP layout, and forward context.
        post_processing_fn: Post-processing function to apply to the logits
        processed_mb: Pre-fetched ProcessedMicrobatch containing data and processed inputs
        global_valid_seqs: Global valid sequence count for loss normalization
        global_valid_toks: Global valid token count for loss normalization
        sampling_params: Sampling parameters (top-k, top-p, temperature)
        sequence_dim: Sequence dimension

    Returns:
        tuple: (result, metrics, processed_microbatch)
            - result: Output from post-processing (loss, logprobs, topk, or scores)
            - metrics: Dictionary of metrics from post-processing
            - processed_microbatch: The ProcessedMicrobatch that was processed
    """
    # Extract the processed components
    data_dict = processed_mb.data_dict
    processed_inputs = processed_mb.processed_inputs
    cp_sharder = prepared.cp_sharder
    if prepared.cp_size > 1 and cp_sharder is None:
        raise RuntimeError(
            "ContextParallelSharder is required when context_parallel_size > 1"
        )
    if prepared.cp_size > 1 and isinstance(post_processing_fn, ScorePostProcessor):
        raise NotImplementedError(
            "ScorePostProcessor does not support context_parallel_size > 1 "
            "on the automodel backend. Set context_parallel_size=1."
        )

    # Model forward pass
    outputs = model_forward(model, prepared.model_batch)

    # Extract logits from model outputs
    logits = extract_logits(model, outputs)
    del outputs

    # Apply temperature scaling only for sampling-oriented post-processors
    # Score computations should use unscaled logits
    if isinstance(
        post_processing_fn,
        (
            LossPostProcessor,
            LogprobsPostProcessor,
            TopkLogitsPostProcessor,
            FullLogitsPostProcessor,
        ),
    ):
        # Temperature scaling is element-wise, directly applying it here.
        # Other sampling parameters like top-k and top-p need the logits from whole vocabulary,
        # so applying them when gathering logits from vocab parallel (called in LossPostProcessor and LogprobsPostProcessor).
        logits = apply_temperature_scaling(logits, sampling_params)

    # Apply the post-processing function directly based on type
    if isinstance(post_processing_fn, LossPostProcessor):
        result, metrics = post_processing_fn(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            cp_sharder=cp_sharder,
            sequence_dim=sequence_dim,
        )
    elif isinstance(
        post_processing_fn,
        (LogprobsPostProcessor, TopkLogitsPostProcessor),
    ):
        result = post_processing_fn(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=processed_mb.original_batch_size,
            original_seq_len=processed_mb.original_seq_len,
            cp_sharder=cp_sharder,
            sequence_dim=sequence_dim,
        )
        if isinstance(post_processing_fn, LogprobsPostProcessor):
            metrics = {"logprobs": result}
        else:
            vals, idx = result
            metrics = {"topk_logits": vals, "topk_indices": idx}
    elif isinstance(post_processing_fn, FullLogitsPostProcessor):
        result = post_processing_fn(
            logits=logits,
            data_dict=data_dict,
            processed_inputs=processed_inputs,
            original_batch_size=processed_mb.original_batch_size,
            original_seq_len=processed_mb.original_seq_len,
            cp_sharder=cp_sharder,
            sequence_dim=sequence_dim,
        )
        metrics = {"full_logits": result}
    elif isinstance(post_processing_fn, ScorePostProcessor):
        result = post_processing_fn(logits=logits)
        metrics = {"scores": result}
    else:
        raise TypeError(
            f"Unknown post-processing function type: {type(post_processing_fn)}"
        )

    del logits
    return result, metrics, processed_mb


def automodel_forward_backward(
    model: nn.Module,
    data_iterator: Iterator[ProcessedMicrobatch],
    post_processing_fn: PostProcessingFunction,
    device_mesh: Optional[DeviceMesh],
    padding_token_id: int,
    autocast_context_factory: Callable[[], AbstractContextManager[Any]],
    forward_only: bool = False,
    is_reward_model: bool = False,
    allow_flash_attn_args: bool = True,
    global_valid_seqs: Optional[torch.Tensor] = None,
    global_valid_toks: Optional[torch.Tensor] = None,
    sampling_params: Optional[TrainingSamplingParams] = None,
    sequence_dim: int = 1,
    dp_size: int = 1,
    cp_size: int = 1,
    num_global_batches: int = 1,
    num_valid_microbatches: Optional[int] = None,
    on_microbatch_start: Optional[Callable[[int], None]] = None,
) -> list[Tuple[Any, dict[str, Any]]]:
    """Execute forward and backward passes for automodel.

    This is the main training loop function that coordinates forward and backward
    passes across multiple microbatches using PyTorch autograd.

    Unlike megatron_forward_backward which uses Megatron's pipeline parallel
    framework, this uses standard PyTorch operations.

    Args:
        model: The model to train
        data_iterator: Iterator yielding ProcessedMicrobatch objects (already processed)
        post_processing_fn: Post-processing function to apply to the logits
        device_mesh: Worker device mesh used by Automodel CP resolution.
        padding_token_id: Token ID used for Automodel sequence padding.
        autocast_context_factory: Worker-owned precision context factory.
        forward_only: If True, skip backward pass
        is_reward_model: Whether this is a reward model
        allow_flash_attn_args: Whether to pass flash_attn_kwargs to model
        global_valid_seqs: Global valid sequence count for loss normalization
        global_valid_toks: Global valid token count for loss normalization
        sampling_params: Sampling parameters (top-k, top-p, temperature)
        sequence_dim: Sequence dimension
        dp_size: Data parallel size
        cp_size: Context parallel size
        num_global_batches: Number of global batches (for metric scaling)
        num_valid_microbatches: Number of valid (non-dummy) microbatches. If provided,
            microbatches beyond this index are treated as dummy batches (loss *= 0).
            If None, all microbatches are considered valid.
        on_microbatch_start: Optional callback called at the start of each microbatch
            with the microbatch index. Useful for cache clearing, etc.

    Returns:
        List of (result, metrics) tuples from each microbatch
    """
    results = []

    for mb_idx, processed_mb in enumerate(data_iterator):
        # Call optional callback at start of microbatch
        if on_microbatch_start is not None:
            on_microbatch_start(mb_idx)

        prepared = prepare_model_forward(
            model,
            processed_mb.processed_inputs,
            device_mesh=device_mesh,
            cp_size=cp_size,
            padding_token_id=padding_token_id,
            is_reward_model=is_reward_model,
            allow_flash_attn_args=allow_flash_attn_args,
        )

        with prepared.model_context_factory(), autocast_context_factory():
            # Forward pass with post-processing
            result, metrics, _ = forward_with_post_processing_fn(
                model=model,
                prepared=prepared,
                post_processing_fn=post_processing_fn,
                processed_mb=processed_mb,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                sampling_params=sampling_params,
                sequence_dim=sequence_dim,
            )

            # Check if this is a dummy batch
            is_dummy = (
                num_valid_microbatches is not None and mb_idx >= num_valid_microbatches
            )

            # Scale metrics for aggregation (only for loss)
            if isinstance(post_processing_fn, LossPostProcessor):
                # skip the update for dummy batches
                if not is_dummy:
                    ## scale by the number of global batches so we get the correct
                    ## value when summing metrics across all microbatches
                    for k in metrics.keys():
                        if "_min" in k or "_max" in k:
                            continue

                        metrics[k] /= num_global_batches
                else:
                    # Zero out loss for dummy batches
                    result = result * 0

                # Backward pass if training
                if not forward_only:
                    ## NOTE: invalid samples should be multiplied
                    ## by zero in the loss function to prevent them
                    ## from affecting the gradient calculation

                    # FSDP averages gradients over its DP mesh (dp_size * cp_size),
                    # while loss normalization expects their sum, so cancel that
                    # average here. Replicated CP losses send each local model
                    # contribution to cp_size loss consumers; divide by that fanout
                    # to avoid overcounting. Partitioned losses have a fanout of 1.
                    loss = (
                        result
                        * dp_size
                        * cp_size
                        / post_processing_fn.cp_gradient_fanout
                    )
                    loss.backward()

        results.append((result, metrics))

    return results


class LossPostProcessor:
    """Post-processor for computing training loss from model outputs."""

    def __init__(
        self,
        loss_fn: LossFunction,
        cfg: PolicyConfig,
        cp_mesh: Any,
        cp_size: int,
        dp_size: int,
        enable_seq_packing: bool = False,
        sampling_params: Optional[TrainingSamplingParams] = None,
    ):
        """Initialize LossPostProcessor.

        Args:
            loss_fn: Loss function to compute loss
            cfg: Configuration dictionary
            cp_mesh: Context parallel mesh, used only to resolve the CP process
                group handed to the loss. The sequence layout itself belongs to
                the per-microbatch ``cp_sharder``.
            cp_size: Context parallel size
            dp_size: Data parallel size
            enable_seq_packing: Whether sequence packing is enabled
            sampling_params: Sampling parameters
        """
        self.loss_fn: LossFunction = loss_fn
        self.cfg: PolicyConfig = cfg
        self.cp_mesh = cp_mesh
        self.cp_size = cp_size
        self.dp_size = dp_size
        self.enable_seq_packing = enable_seq_packing
        self.sampling_params = sampling_params
        self._cp_gradient_fanout = (
            cp_size
            if cp_size > 1
            and loss_fn.input_type
            in (
                LossInputType.LOGIT,
                LossInputType.LOGPROB,
                LossInputType.DISTILLATION,
            )
            else 1
        )

    @property
    def cp_gradient_fanout(self) -> int:
        """Number of CP loss consumers for each local model contribution."""
        return self._cp_gradient_fanout

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],
        processed_inputs: ProcessedInputs,
        global_valid_seqs: torch.Tensor,
        global_valid_toks: torch.Tensor,
        *,
        cp_sharder: Optional[ContextParallelSharder],
        sequence_dim: int = 1,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Compute loss from logits.

        Args:
            logits: Model output logits
            data_dict: Microbatch data
            processed_inputs: Processed inputs
            global_valid_seqs: Global valid sequence count
            global_valid_toks: Global valid token count
            cp_sharder: Per-microbatch Automodel sequence-layout owner, or None
                when context parallelism is inactive.
            sequence_dim: Sequence dimension

        Returns:
            Tuple of (loss, metrics)
        """
        # Under CP, ``logits`` is this rank's local shard while ``data_dict``
        # stays canonical; the sharder maps between the two.
        token_layout = cp_sharder
        if token_layout is not None:
            input_type = self.loss_fn.input_type
            if input_type == LossInputType.LOGIT:
                # Logit losses such as value-head MSE consume full-sequence logits
                # against the canonical data_dict, so restore canonical order here
                # while keeping any vocabulary (TP) sharding intact.
                logits = _cp_gather_logits(logits, token_layout, sequence_dim)
            elif input_type not in (
                LossInputType.LOGPROB,
                LossInputType.DISTILLATION,
                LossInputType.DISTILLATION_CROSS_TOKENIZER,
            ):
                raise NotImplementedError(
                    f"Loss input type {input_type} is not supported with "
                    "context_parallel_size > 1 on the automodel policy worker."
                )

        # Wrap prepare_loss_input with sampling_params
        prepare_loss_input_wrapped = partial(
            prepare_loss_input,
            sampling_params=self.sampling_params,
            context_parallel_group=(
                self.cp_mesh.get_group() if self.cp_size > 1 else None
            ),
            cp_sharder=token_layout,
        )
        # Wrap loss function for sequence packing if needed
        if self.enable_seq_packing:
            loss_fn = SequencePackingLossWrapper(
                loss_fn=self.loss_fn,
                prepare_fn=prepare_loss_input_wrapped,
                cu_seqlens_q=processed_inputs.flash_attn_kwargs.cu_seqlens_q,
                cu_seqlens_q_padded=processed_inputs.flash_attn_kwargs.cu_seqlens_q,
            )
            loss, loss_metrics = loss_fn(
                logits,
                data_dict,
                global_valid_seqs,
                global_valid_toks,
            )
        else:
            loss_input, data_dict = prepare_loss_input_wrapped(
                logits, data_dict, self.loss_fn
            )
            loss, loss_metrics = self.loss_fn(
                data=data_dict,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                **loss_input,
            )

        return loss, loss_metrics


class LogprobsPostProcessor:
    """Post-processor for computing log probabilities from model outputs."""

    def __init__(
        self,
        cfg: PolicyConfig,
        enable_seq_packing: bool = False,
        sampling_params: Optional[TrainingSamplingParams] = None,
    ):
        """Initialize LogprobsPostProcessor.

        Args:
            cfg: Configuration dictionary
            enable_seq_packing: Whether sequence packing is enabled
            sampling_params: Sampling parameters
        """
        self.cfg = cfg
        self.enable_seq_packing = enable_seq_packing
        self.sampling_params = sampling_params
        self.logprob_chunk_size = cfg.get("logprob_chunk_size", None)

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],
        processed_inputs: ProcessedInputs,
        original_batch_size: int,
        original_seq_len: int,
        *,
        cp_sharder: Optional[ContextParallelSharder],
        sequence_dim: int = 1,
    ) -> torch.Tensor:
        """Compute token log probabilities from logits.

        Args:
            logits: Model output logits
            data_dict: Microbatch data
            processed_inputs: Processed inputs
            original_batch_size: Original batch size before packing
            original_seq_len: Original sequence length before packing
            cp_sharder: Per-microbatch Automodel sequence-layout owner, or None
                when context parallelism is inactive.
            sequence_dim: Sequence dimension

        Returns:
            Token log probabilities tensor [batch_size, seq_length]
        """
        input_lengths = data_dict["input_lengths"]

        if cp_sharder is not None:
            # ``data_dict`` stays canonical under CP: ``_build_model_batch`` clones
            # the model-facing tensors so Automodel's in-place buffer sharding
            # cannot reach the loss-side ones. Shift and gather against that
            # canonical sequence and let the sharder own the local layout.
            canonical_input_ids = data_dict["input_ids"]
            seq_len = canonical_input_ids.shape[1]
            token_logprobs = get_cp_sharded_next_token_logprobs(
                logits,
                canonical_input_ids,
                cp_sharder,
                chunk_size=self.logprob_chunk_size,
                sampling_params=self.sampling_params,  # top-k and top-p filtering
            )

            assert token_logprobs.shape[1] == seq_len - 1
        else:
            seq_len = processed_inputs.seq_len
            if isinstance(logits, DTensor):
                # DTensor path with TP sharding
                token_logprobs = get_logprobs_from_vocab_parallel_logits(
                    logits,
                    processed_inputs.input_ids,
                    chunk_size=self.logprob_chunk_size,
                    sampling_params=self.sampling_params,  # top-k and top-p filtering
                )
            else:
                # Non-DTensor path (no TP sharding)
                token_logprobs = self._compute_local_logprobs(
                    logits, processed_inputs.input_ids
                )

        # Prepend 0 for first token to maintain sequence length
        token_logprobs = torch.cat(
            [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
        )

        # Handle sequence packing unpacking or mask application
        if self.enable_seq_packing:
            unpacked_logprobs = torch.zeros(
                (original_batch_size, original_seq_len),
                dtype=token_logprobs.dtype,
                device=token_logprobs.device,
            )
            cu_seqlens = processed_inputs.flash_attn_kwargs.cu_seqlens_q
            for i in range(original_batch_size):
                start = cu_seqlens[i].item() + 1
                end = cu_seqlens[i + 1].item()
                seq_len_actual = input_lengths[i].item()
                unpacked_logprobs[i, 1:seq_len_actual] = token_logprobs[0, start:end]
            token_logprobs = unpacked_logprobs
        else:
            # Apply mask to zero out padding tokens logprobs
            batch_size = processed_inputs.input_ids.shape[0]
            post_attention_mask = torch.zeros(
                (batch_size, seq_len),
                dtype=torch.bool,
                device=token_logprobs.device,
            )
            for i, length in enumerate(input_lengths):
                # For right-padded sequence, set 1s at the beginning of the sequence
                post_attention_mask[i, :length] = 1
            token_logprobs = token_logprobs * post_attention_mask

        # handle top-k/top-p filtering for logprobs, only used for ClippedPGLossFn now
        if need_top_k_or_top_p_filtering(self.sampling_params):
            mask = data_dict["token_mask"] * data_dict["sample_mask"].unsqueeze(-1)
            token_logprobs = mask_out_neg_inf_logprobs(
                token_logprobs, mask, "prev_logprobs"
            )

        return token_logprobs

    def _compute_local_logprobs(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logprobs locally without distributed processing.

        Args:
            logits: Model output logits
            input_ids: Input token IDs

        Returns:
            Token log probabilities

        When ``logprob_chunk_size`` is set, log-softmax and gather run per
        sequence chunk to bound peak memory for long-context runs.
        """
        # Extract logprobs for each token in the sequence by gathering the logprob
        # corresponding to the next token at each position
        # Input shapes:
        #   logits: [batch_size, sequence_length, vocab_size] - logits for each position
        #   token_ids: [batch_size, sequence_length] - actual tokens
        next_tokens = input_ids[:, 1:].to(logits.device)
        target_seq_len = int(next_tokens.shape[1])

        if target_seq_len == 0:
            return logits.new_empty(
                (input_ids.shape[0], 0),
                dtype=torch.float32,
            )

        logits = logits[:, :target_seq_len, :]

        if self.logprob_chunk_size is None:
            logits = logits.to(torch.float32).contiguous()
            logits = apply_top_k_top_p_filtering_for_local_logits(
                logits, self.sampling_params
            )
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_logprobs = log_probs.gather(
                dim=-1, index=next_tokens.unsqueeze(-1)
            ).squeeze(-1)
            del log_probs
            return token_logprobs

        chunked_token_logprobs = []
        for chunk_start in range(0, target_seq_len, self.logprob_chunk_size):
            chunk_end = min(chunk_start + self.logprob_chunk_size, target_seq_len)
            chunk_logits = logits[:, chunk_start:chunk_end, :].to(torch.float32)
            chunk_logits = chunk_logits.contiguous()
            chunk_logits = apply_top_k_top_p_filtering_for_local_logits(
                chunk_logits, self.sampling_params
            )
            chunk_log_probs = torch.nn.functional.log_softmax(chunk_logits, dim=-1)
            chunk_token_logprobs = chunk_log_probs.gather(
                dim=-1,
                index=next_tokens[:, chunk_start:chunk_end].unsqueeze(-1),
            ).squeeze(-1)
            chunked_token_logprobs.append(chunk_token_logprobs)
            del chunk_log_probs, chunk_logits

        token_logprobs = torch.cat(chunked_token_logprobs, dim=1)
        del chunked_token_logprobs

        return token_logprobs


class TopkLogitsPostProcessor:
    """Post-processor for computing top-k logits from model outputs."""

    def __init__(
        self,
        cfg: PolicyConfig,
        tp_mesh: Any,
        k: int,
        enable_seq_packing: bool = False,
    ):
        """Initialize TopkLogitsPostProcessor.

        Args:
            cfg: Configuration dictionary
            tp_mesh: Tensor parallel mesh, used for the vocabulary-parallel
                top-k. The sequence layout belongs to the per-microbatch
                ``cp_sharder``.
            k: Number of top logits to return
            enable_seq_packing: Whether sequence packing is enabled
        """
        self.cfg = cfg
        self.tp_mesh = tp_mesh
        self.k = k
        self.enable_seq_packing = enable_seq_packing

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],
        processed_inputs: ProcessedInputs,
        original_batch_size: int,
        original_seq_len: int,
        *,
        cp_sharder: Optional[ContextParallelSharder],
        sequence_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k logits and indices from model outputs.

        Args:
            logits: Model output logits
            data_dict: Microbatch data
            processed_inputs: Processed inputs
            original_batch_size: Original batch size before packing
            original_seq_len: Original sequence length before packing
            cp_sharder: Per-microbatch Automodel sequence-layout owner, or None
                when context parallelism is inactive.
            sequence_dim: Sequence dimension

        Returns:
            Tuple of (top-k values, top-k indices) tensors
        """
        input_lengths = data_dict["input_lengths"]

        if cp_sharder is not None:
            # Deal with TP first; the logits are already this rank's CP shard.
            local_logits = to_local_if_dtensor(logits)  # [B, S_cp, V_tp]

            tp_group = self.tp_mesh.get_group()
            tp_rank = torch.distributed.get_rank(tp_group)
            V_local = int(local_logits.shape[-1])
            vocab_start_index = tp_rank * V_local
            vocab_end_index = (tp_rank + 1) * V_local

            vals, idx = distributed_vocab_topk(
                local_logits,
                k=self.k,
                tp_group=tp_group,
                vocab_start_index=vocab_start_index,
                vocab_end_index=vocab_end_index,
            )
            # [B, S_cp, k]

            # Automodel's layout restores canonical order and trims its CP padding.
            vals = cp_sharder.gather_token_tensor(vals, seq_dim=sequence_dim, trim=True)
            idx = cp_sharder.gather_token_tensor(idx, seq_dim=sequence_dim, trim=True)
            # [B, S, k]
        else:
            # Compute top-k over full sequence length
            if isinstance(logits, DTensor):
                local_logits = logits.to_local()  # [B, S, V_local]
                tp_group = self.tp_mesh.get_group()
                tp_rank = torch.distributed.get_rank(tp_group)
                V_local = int(local_logits.shape[-1])
                vocab_start_index = tp_rank * V_local
                vocab_end_index = (tp_rank + 1) * V_local

                vals, idx = distributed_vocab_topk(
                    local_logits,
                    k=self.k,
                    tp_group=tp_group,
                    vocab_start_index=vocab_start_index,
                    vocab_end_index=vocab_end_index,
                )
            else:
                full_logits = logits.to(torch.float32)
                vals, idx = torch.topk(full_logits, k=self.k, dim=-1)

        # Handle sequence packing unpacking
        if self.enable_seq_packing:
            # Unpack top-k results from packed format back to original batch format
            # vals: [1, packed_seq_len, k] -> [original_batch_size, original_seq_len, k]
            # idx: [1, packed_seq_len, k] -> [original_batch_size, original_seq_len, k]
            unpacked_vals = torch.zeros(
                (original_batch_size, original_seq_len, self.k),
                dtype=vals.dtype,
                device=vals.device,
            )
            unpacked_idx = torch.zeros(
                (original_batch_size, original_seq_len, self.k),
                dtype=idx.dtype,
                device=idx.device,
            )

            cu_seqlens = processed_inputs.flash_attn_kwargs.cu_seqlens_q

            for i in range(original_batch_size):
                start = cu_seqlens[i].item()
                end = cu_seqlens[i + 1].item()
                seq_len_actual = input_lengths[i].item()

                # Extract the corresponding portion from packed results
                # Note: vals and idx are [1, packed_seq_len, k] due to packing
                unpacked_vals[i, :seq_len_actual, :] = vals[0, start:end, :]
                unpacked_idx[i, :seq_len_actual, :] = idx[0, start:end, :]

            vals = unpacked_vals
            idx = unpacked_idx

        return vals, idx


class FullLogitsPostProcessor:
    """Export this rank's raw teacher logits (full vocab, no reduction at the worker).

    Used by cross-tokenizer distillation; the loss fn does all vocab
    reduction (none at the worker) so the distributed result matches the
    single-GPU PyTorch reference. Teacher TP/CP
    are supported and may differ from the student's: under TP each rank emits
    its vocab shard, under CP it allgathers and re-emits its contiguous seq
    slice; the IPC consumer reassembles the global ``[B, T_t, V_t]``.
    Sequence packing raises ``NotImplementedError``.
    """

    def __init__(
        self,
        cfg: PolicyConfig,
        cp_mesh: Any,
        cp_size: int,
        enable_seq_packing: bool = False,
    ):
        """Initialize FullLogitsPostProcessor.

        Args:
            cfg: Configuration dictionary
            cp_mesh: Context parallel mesh, used to pick this rank's contiguous
                IPC window. The sequence layout of the logits themselves belongs
                to the per-microbatch ``cp_sharder``.
            cp_size: Context parallel size
            enable_seq_packing: Whether sequence packing is enabled
        """
        self.cfg = cfg
        self.cp_mesh = cp_mesh
        self.cp_size = cp_size
        self.enable_seq_packing = enable_seq_packing

    def __call__(
        self,
        logits: torch.Tensor,
        data_dict: BatchedDataDict[Any],
        processed_inputs: Any,
        original_batch_size: int,
        original_seq_len: int,
        *,
        cp_sharder: Optional[ContextParallelSharder],
        sequence_dim: int = 1,
    ) -> torch.Tensor:
        if self.enable_seq_packing:
            raise NotImplementedError(
                "FullLogitsPostProcessor: sequence packing is not supported in v0."
            )
        if isinstance(logits, DTensor):
            logits = logits.to_local()
        # fp32 for the consumer's precision-sensitive log_softmax / top-k /
        # projection (KL math), and a dtype-consistent IPC buffer producer<->consumer.
        logits = logits.to(torch.float32)

        # Automodel's CP layout is not contiguous per rank, but the IPC consumer
        # routes by contiguous ``global_seq_start`` over the teacher CP group.
        # Restore canonical order (trimming Automodel's CP padding) and emit this
        # rank's contiguous slice, else heterogeneous teacher_cp != student_cp
        # lands teacher data at the wrong seq positions in the consumer's dest
        # tensor.
        if cp_sharder is not None and self.cp_mesh is not None:
            full = cp_sharder.gather_token_tensor(
                logits, seq_dim=sequence_dim, trim=True
            )
            full_seq_len = full.shape[sequence_dim]
            if full_seq_len % self.cp_size != 0:
                raise ValueError(
                    "X-token teacher sequence length must be divisible by the "
                    "teacher context parallel size, but got "
                    f"sequence_length={full_seq_len}, cp_size={self.cp_size}. "
                    "Set the teacher's make_sequence_length_divisible_by to a "
                    "multiple of its dtensor_cfg.context_parallel_size."
                )
            local_len = full_seq_len // self.cp_size
            cp_rank = torch.distributed.get_rank(self.cp_mesh.get_group())
            logits = full.narrow(
                sequence_dim, cp_rank * local_len, local_len
            ).contiguous()
        return logits  # [B, S_local_contiguous, V_t]


class ScorePostProcessor:
    """Post-processor for computing reward model scores from model outputs."""

    def __init__(
        self,
        cfg: PolicyConfig,
    ):
        """Initialize ScorePostProcessor.

        Args:
            cfg: Configuration dictionary
        """
        self.cfg = cfg

    def __call__(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Extract scores from reward model outputs.

        Args:
            logits: Model output logits

        Returns:
            Scores tensor
        """
        logits = logits.to(torch.float32)
        rm_scores = to_local_if_dtensor(logits)
        rm_scores = rm_scores.squeeze(-1)

        return rm_scores


def aggregate_training_statistics(
    losses: list[float],
    all_mb_metrics: list[dict[str, Any]],
    grad_norm: Optional[torch.Tensor],
    dp_group: Any,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Aggregate training statistics across microbatches and ranks.

    Args:
        losses: List of loss values from each microbatch
        all_mb_metrics: List of metrics dictionaries from each microbatch
        grad_norm: Gradient norm tensor (or None if eval mode)
        dp_group: Data parallel process group for all-reduce
        dtype: Model dtype for metrics

    Returns:
        Dictionary containing aggregated metrics including global_loss, grad_norm, etc.
    """
    # Compute global loss across all ranks
    with torch.no_grad():
        global_loss = torch.tensor(losses, device="cuda")
        torch.distributed.all_reduce(global_loss, group=dp_group)

    # Aggregate metrics across all microbatches
    mb_metrics = defaultdict(list)
    for m in all_mb_metrics:
        for k, v in m.items():
            mb_metrics[k].append(v)

    metrics = {
        "global_loss": global_loss.cpu(),
        "grad_norm": grad_norm,
        "rank": torch.distributed.get_rank(),
        "gpu_name": torch.cuda.get_device_name(),
        "model_dtype": dtype,
        "all_mb_metrics": dict(mb_metrics),
    }

    return metrics
