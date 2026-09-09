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

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import partial
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    PipelineOffloadManager,
)
from megatron.core.utils import StragglerDetector, get_model_config

from nemo_rl.algorithms.logits_sampling_utils import (
    TrainingSamplingParams,
    need_top_k_or_top_p_filtering,
)
from nemo_rl.algorithms.loss import (
    DraftLossWrapper,
    SequencePackingFusionLossWrapper,
    SequencePackingLossWrapper,
    prepare_loss_input,
    prepare_packed_loss_input,
    resolve_block_draft_slot_weights,
    wrap_loss_fn_with_input_preparation,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.utils import mask_out_neg_inf_logprobs
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
)
from nemo_rl.models.megatron.config import MegatronModule
from nemo_rl.models.megatron.data import ProcessedMicrobatch
from nemo_rl.models.megatron.draft.hidden_capture import (
    TapChannel,
    get_capture_context,
)
from nemo_rl.models.megatron.router_replay import (
    clear_router_replay,
    set_router_replay_backward,
    set_router_replay_forward,
)
from nemo_rl.models.policy import PolicyConfig

# Union type for any post-processing function (defined after classes below)
PostProcessingFunction = Union[
    "LossPostProcessor",
    "LogprobsPostProcessor",
    "TopkLogitsPostProcessor",
]


@contextmanager
def suspend_activation_offload_for_forward_only(
    model: Union[GPTModel, List[GPTModel]], forward_only: bool
) -> Iterator[None]:
    """Keep inference-only RL phases from consuming MCore's training warmup."""
    if not forward_only:
        yield
        return

    model_chunks = model if isinstance(model, list) else [model]
    original_values: List[Tuple[Any, bool]] = []
    seen_configs: set[int] = set()
    for model_chunk in model_chunks:
        model_config = get_model_config(model_chunk)
        if id(model_config) in seen_configs:
            continue
        seen_configs.add(id(model_config))
        original_value = bool(
            getattr(model_config, "fine_grained_activation_offloading", False)
        )
        if original_value:
            original_values.append((model_config, original_value))

    offload_manager = PipelineOffloadManager.OFFLOAD_MGR
    suspend_manager = bool(
        original_values and offload_manager is not None and offload_manager.do_offload
    )

    try:
        for model_config, _ in original_values:
            model_config.fine_grained_activation_offloading = False
        if suspend_manager and offload_manager is not None:
            offload_manager.disable_offload()
        yield
    finally:
        try:
            if suspend_manager and offload_manager is not None:
                offload_manager.enable_offload()
        finally:
            for model_config, original_value in original_values:
                model_config.fine_grained_activation_offloading = original_value


def model_forward(
    model: GPTModel,
    data_dict: BatchedDataDict[Any],
    input_ids_cp_sharded: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    packed_seq_params: Optional[PackedSeqParams] = None,
    defer_fp32_logits: Optional[bool] = False,
    mtp_loss_mask: Optional[torch.Tensor] = None,
    straggler_timer: Optional[StragglerDetector] = None,
    use_fused_linear_logprobs: bool = False,
    media_token_validity_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Perform a single forward pass through the model.

    Args:
        model: The model to run forward pass on
        data_dict: Dictionary containing batch data
        input_ids_cp_sharded: Model-forward token IDs. Usually CP-sharded; models
            that insert media before CP selection receive the full packed THD row.
        position_ids: Position IDs for tokens
        attention_mask: Attention mask for the sequence
        packed_seq_params: Parameters for packed sequences (optional)
        defer_fp32_logits: Whether to skip the conversion of logits to fp32
        mtp_loss_mask: MTP loss mask to exclude prompt tokens from MTP loss (optional)
        straggler_timer: Straggler detector for profiling the forward pass
        use_fused_linear_logprobs: Whether to compute logprobs with the fused
            chunked linear cross-entropy kernel (directly from hidden states)
        media_token_validity_mask: Which media-token positions actually anchor a
            projected feature, already in this model's token layout. Only passed
            when the model accepts it; otherwise the model derives its own.

    Returns:
        torch.Tensor: Output tensor from the model (logits)
    """
    multimodal_data = data_dict.get_multimodal_dict(
        as_tensors=True, device=input_ids_cp_sharded.device
    )
    if len(multimodal_data) > 0:
        position_ids = None

    additional_kwargs = {}
    # Mamba models currently do not support packed_seq_params
    if packed_seq_params is not None:
        additional_kwargs["packed_seq_params"] = packed_seq_params

    # Pass MTP loss mask to exclude prompt tokens from MTP loss
    if mtp_loss_mask is not None:
        additional_kwargs["loss_mask"] = mtp_loss_mask

    # Only sent when the model advertises the parameter, so it never reaches a
    # forward that would swallow it into **kwargs and quietly ignore it.
    if media_token_validity_mask is not None:
        additional_kwargs["media_token_validity_mask"] = media_token_validity_mask

    if defer_fp32_logits:
        additional_kwargs["fp32_output"] = False
    if use_fused_linear_logprobs:
        additional_kwargs["labels"] = input_ids_cp_sharded
        # Only pass this kwarg when linear CE fusion is enabled. Older Megatron-LM
        # GPTModel.forward signatures do not accept it.
        additional_kwargs["return_logprobs_for_linear_ce_fusion"] = True

    with straggler_timer() if straggler_timer is not None else nullcontext():
        output_tensor = model(
            input_ids=input_ids_cp_sharded,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **additional_kwargs,
            **multimodal_data,
        )

    # A model that slices context parallelism itself returns (output,
    # sliced_loss_mask) when it was handed a full-sequence loss_mask, so the
    # caller can see the mask in the model's own CP-local token order. The MTP
    # loss is computed inside the model against that mask, so only the logits
    # are needed here. Without this the tuple reaches the loss wrapper, which
    # calls .narrow() on it. See modeling_nemotron_omni.py return_sliced_loss_mask.
    if isinstance(output_tensor, tuple):
        output_tensor = output_tensor[0]

    return output_tensor


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


def _run_block_draft_forward(
    *,
    model: MegatronModule,
    draft_model: MegatronModule,
    captured_states: Any,
    data_dict: BatchedDataDict[Any],
    packed_seq_params: Optional[PackedSeqParams] = None,
    tap_channel: Optional[TapChannel] = None,
) -> torch.Tensor:
    """Run the DFlash/DSpark block-draft forward for one microbatch.

    Returns prediction-slot logits aligned with labels ``x_{p+1} ..
    x_{p+gamma}`` per anchor ``p`` (DFlash's bonus anchor slot is dropped;
    DSpark's teacher-forced Markov bias is added): ``[B, N, gamma, V_local]``
    unpacked, ``[NB, gamma, V_local]`` packed — under packing each block is
    owned by the CP rank holding its anchor's zigzag chunk (which balances
    the staircase load the way zigzag balances causal, and keeps the anchor
    embedding local); the flat owned coords are stashed for the loss.

    Following the official DFlash contract the draft owns neither an LM head
    nor a mask embedding: logits are projected through the policy's LIVE head
    and mask slots embed via the policy's LIVE ``embed_tokens[mask_token_id]``
    row — both passed DETACHED (the draft never trains them; serving matches
    because vLLM shares the target's lm_head and embed_tokens with a
    head-less/embedding-less drafter).
    """
    # Deferred imports mirror the worker: block-draft-path-only dependencies.
    from nemo_rl.models.megatron.draft.dflash import count_map_to_anchors
    from nemo_rl.models.megatron.draft.utils import (
        get_policy_embedding_row,
        get_policy_lm_head_weight,
    )

    # Anchors travel through the batch as a [B, S] count map (the only layout
    # that survives dynamic batching's sequence-dim validation/truncation and
    # length-bucket reorder); rebuild this microbatch's block list and stash
    # it for the loss (slot mask + teacher gather read these keys).
    anchors, anchor_valid = count_map_to_anchors(data_dict["draft_anchor_count_map"])

    method_kwargs: dict[str, Any] = {}
    if packed_seq_params is not None:
        cp_group = get_context_parallel_group()
        cp_size = torch.distributed.get_world_size(cp_group)
        cp_rank = torch.distributed.get_rank(cp_group)
        cu_global = packed_seq_params.cu_seqlens_q_padded
        if cu_global is None:
            cu_global = packed_seq_params.cu_seqlens_q
        cu_local = torch.div(cu_global, cp_size, rounding_mode="floor").to(torch.long)
        half = ((cu_local[1:] - cu_local[:-1]) // 2).clamp(min=1)

        batch_size, num_anchors = anchors.shape
        seq_flat = (
            torch.arange(batch_size, device=anchors.device)
            .unsqueeze(1)
            .expand(-1, num_anchors)
            .reshape(-1)
        )
        anchors_flat = anchors.reshape(-1)
        valid_flat = anchor_valid.reshape(-1)
        chunk_idx = anchors_flat // half[seq_flat]
        owner = torch.minimum(chunk_idx, 2 * cp_size - 1 - chunk_idx)
        mine = owner == cp_rank
        seq_flat = seq_flat[mine]
        anchors_flat = anchors_flat[mine]
        valid_flat = valid_flat[mine]
        if seq_flat.numel() == 0:
            # Every rank must field >= 1 block: the trunk ring is a CP
            # collective and the decoder cannot run an empty stream. A dummy
            # invalid block at this rank's front-chunk start is loss-masked.
            seq_flat = torch.zeros(1, dtype=torch.long, device=anchors.device)
            anchors_flat = (cp_rank * half[0]).reshape(1)
            valid_flat = torch.zeros(1, dtype=torch.bool, device=anchors.device)

        data_dict["draft_anchor_positions"] = anchors_flat
        data_dict["draft_anchor_valid"] = valid_flat
        data_dict["draft_block_seq_idx"] = seq_flat
        data_dict["draft_packed_local_cu_seqlens"] = cu_local
        method_kwargs["packed_seq_params"] = packed_seq_params
        method_kwargs["block_seq_idx"] = seq_flat
        anchors, anchor_valid = anchors_flat, valid_flat
    else:
        data_dict["draft_anchor_positions"] = anchors
        data_dict["draft_anchor_valid"] = anchor_valid

    if draft_model.speculator_type == "dspark":
        # Teacher-forces the Markov/confidence heads inside the forward.
        method_kwargs["input_ids"] = data_dict["input_ids"]
    draft_out = draft_model(
        taps=captured_states.hidden_states,
        input_embeds=captured_states.inputs_embeds,
        anchors=anchors,
        anchor_valid=anchor_valid,
        lm_head_weight=get_policy_lm_head_weight(model).detach(),
        # Under PP > 1 the embedding lives on the first stage;
        # TapChannel.begin_pass broadcast this pass's live row here.
        mask_embedding=(
            tap_channel.mask_row
            if tap_channel is not None
            else get_policy_embedding_row(model, draft_model.mask_token_id)
        ).detach(),
        **method_kwargs,
    )
    if draft_model.speculator_type == "dflash":
        # Slot 0 is the anchor bonus slot (condition only); the gamma mask
        # slots align with labels x_{p+1} .. x_{p+gamma}.
        if packed_seq_params is not None:
            return draft_out[:, 1:, :]
        return draft_out[:, :, 1:, :]
    elif draft_model.speculator_type == "dspark":
        block_logits, confidence_pred = draft_out
        data_dict["draft_confidence_pred"] = confidence_pred
        return block_logits
    else:
        raise ValueError(
            f"Unknown block-draft speculator_type '{draft_model.speculator_type}'."
        )


def forward_with_post_processing_fn(
    data_iterator: Iterator[ProcessedMicrobatch],
    model: GPTModel,
    post_processing_fn: PostProcessingFunction,
    defer_fp32_logits: Optional[bool] = False,
    global_valid_seqs: Optional[torch.Tensor] = None,
    global_valid_toks: Optional[torch.Tensor] = None,
    global_draft_pass_counts: Optional[torch.Tensor] = None,
    sampling_params: Optional[TrainingSamplingParams] = None,
    straggler_timer: Optional[StragglerDetector] = None,
    draft_model: Optional[MegatronModule] = None,
    enable_hidden_capture: Optional[bool] = False,
    tap_channel: Optional[TapChannel] = None,
    draft_ttt_steps: int = 1,
    draft_aux_layer_indices: Optional[Tuple[int, ...]] = None,
    use_fused_linear_logprobs: bool = False,
    use_router_replay: bool = False,
    router_replay_train: bool = False,
) -> Tuple[torch.Tensor, Callable]:
    """Perform forward pass with pre-processed microbatch and return output tensor and post-processing function.

    This function takes a pre-processed microbatch (with sequence packing already handled),
    runs the forward step through the model, and prepares a post-processing function for
    post-processing the outputs.

    Args:
        data_iterator: Iterator yielding ProcessedMicrobatch objects (already processed)
        model: The model to run forward pass on
        post_processing_fn: Post-processing function to post-process the logits
        defer_fp32_logits: Whether to defer FP32 conversion of logits
        global_valid_seqs: Global valid sequence count for loss normalization
        global_valid_toks: Global valid token count for loss normalization
        sampling_params: Sampling parameters (top-k, top-p, temperature)
        straggler_timer: Straggler detector for profiling the forward pass
        draft_model: Draft model for online draft model training
        enable_hidden_capture: Whether to enable hidden state capture for draft model training
        draft_aux_layer_indices: Aux-layer ids to capture, resolved
            rank-independently by the worker (every PP stage must agree; see
            resolve_draft_aux_layer_ids)

    Returns:
        tuple: (output_tensor, post_processing_fn_wrapped)
            - output_tensor: Raw model outputs (logits)
            - post_processing_fn_wrapped: Function to create output post-processing function when called
    """
    # Get the pre-processed microbatch from the iterator
    processed_mb = next(data_iterator)

    # Extract the processed components
    data_dict = processed_mb.data_dict
    input_ids = processed_mb.input_ids
    input_ids_cp_sharded = processed_mb.input_ids_cp_sharded
    attention_mask = processed_mb.attention_mask
    position_ids = processed_mb.position_ids
    packed_seq_params = processed_mb.packed_seq_params
    cu_seqlens_padded = processed_mb.cu_seqlens_padded
    mtp_loss_mask = processed_mb.mtp_loss_mask
    routed_experts_cp_sharded = processed_mb.routed_experts_cp_sharded
    original_seq_length = processed_mb.original_seq_length
    media_token_validity_mask = processed_mb.media_token_validity_mask

    if use_router_replay:
        if routed_experts_cp_sharded is None:
            raise RuntimeError(
                "Router replay is enabled but routed_experts is missing from the microbatch."
            )
        set_router_replay_forward(model, routed_experts_cp_sharded)

    # Insert hook to capture hidden states and embeddings for draft model
    # training. Capture the aux layers the DRAFT was built for (checkpoint
    # target_layer_ids / policy.draft.aux_layer_indices): the serving-side
    # drafter taps exactly these policy layers, so capturing the hard-coded
    # defaults instead would silently train on different features (or break
    # the fc width when the counts differ). The worker resolves the list
    # rank-independently (resolve_draft_aux_layer_ids) and threads it in:
    # under PP only the last stage owns a draft model to read it from, and
    # the capture posts one P2P send/recv per id, so stages disagreeing about
    # the list would desync the pipeline.
    aux_layer_indices = draft_aux_layer_indices
    if aux_layer_indices is None and draft_model is not None:
        configured_aux_layers = draft_model.config.eagle_aux_hidden_state_layer_ids
        if configured_aux_layers:
            aux_layer_indices = tuple(int(i) for i in configured_aux_layers)
    capture_context, capture = get_capture_context(
        model,
        enable_hidden_capture,
        aux_layer_indices=aux_layer_indices,
        tap_channel=tap_channel,
    )
    try:
        with capture_context:
            output_tensor = model_forward(
                model=model,
                data_dict=data_dict,
                input_ids_cp_sharded=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
                defer_fp32_logits=defer_fp32_logits,
                mtp_loss_mask=mtp_loss_mask,
                straggler_timer=straggler_timer,
                use_fused_linear_logprobs=use_fused_linear_logprobs,
                media_token_validity_mask=media_token_validity_mask,
            )
    except Exception:
        # The forward above armed the router-replay action (set_router_replay_forward);
        # if it raised, clear that armed state so stale replay action/indices do not
        # leak into the next microbatch, then re-raise the original error unchanged.
        if use_router_replay:
            clear_router_replay(model)
        raise

    if use_router_replay:
        if router_replay_train:
            set_router_replay_backward(model)
        else:
            clear_router_replay(model)

    if capture is not None:
        if use_fused_linear_logprobs:
            # The fused path never materializes the policy's full logits, so
            # there is no soft-CE teacher for the draft; the DRAFT loss prep
            # would silently treat fused logprobs as logits.
            raise RuntimeError(
                "Draft-model training requires the policy's full logits; "
                "disable megatron_cfg.use_fused_linear_logprobs."
            )

        # PP > 1 source stages push their taps to the draft stage inside
        # get_captured_states and come back empty; only the draft owner rank
        # (the one with draft_model attached) runs the draft forward below.
        captured_states = capture.get_captured_states()

    if capture is not None and draft_model is not None:
        if getattr(draft_model, "speculator_type", "eagle3") in (
            "dflash",
            "dspark",
        ):
            data_dict["draft_block_logits"] = _run_block_draft_forward(
                model=model,
                draft_model=draft_model,
                captured_states=captured_states,
                data_dict=data_dict,
                packed_seq_params=packed_seq_params,
                tap_channel=tap_channel,
            )
        else:
            from megatron.core.transformer.multi_token_prediction import roll_tensor

            from nemo_rl.algorithms.loss.utils import (
                packed_zigzag_token_coords,
                roll_packed_left_cp,
            )

            cp_group = get_context_parallel_group()
            if packed_seq_params is not None:
                # Packed (THD) input: the pass-1 shift must stop at
                # subsequence boundaries and, under CP, exchange the zigzag
                # chunk-boundary elements. Also stash the local->(subseq,
                # position) coords the packed draft loss uses to gather its
                # per-pass masks from the unpacked [B, S] token_mask.
                cp_size = torch.distributed.get_world_size(cp_group)
                cp_rank = torch.distributed.get_rank(cp_group)
                cu_global = packed_seq_params.cu_seqlens_q_padded
                if cu_global is None:
                    cu_global = packed_seq_params.cu_seqlens_q
                cu_local = torch.div(cu_global, cp_size, rounding_mode="floor").to(
                    torch.int32
                )
                seq_index, pos_in_seq = packed_zigzag_token_coords(
                    cu_global, cp_rank, cp_size
                )
                data_dict["draft_packed_seq_index"] = seq_index
                data_dict["draft_packed_pos_in_seq"] = pos_in_seq
                data_dict["draft_packed_local_cu_seqlens"] = cu_local
                shifted_input_embeds = roll_packed_left_cp(
                    captured_states.inputs_embeds,
                    cu_local,
                    cp_group if cp_size > 1 else None,
                )
            else:
                shifted_input_embeds = roll_tensor(
                    captured_states.inputs_embeds,
                    shifts=-1,
                    dims=0,
                    cp_group=cp_group,
                )[0]
            if draft_ttt_steps > 1:
                data_dict["student_logits_by_pass"] = draft_model.forward_ttt(
                    hidden_states=captured_states.hidden_states,
                    input_embeds=shifted_input_embeds,
                    packed_seq_params=packed_seq_params,
                )
            else:
                data_dict["student_logits"] = draft_model(
                    hidden_states=captured_states.hidden_states,
                    input_embeds=shifted_input_embeds,
                    # The draft decoder is forced onto the fused causal path
                    # (see EagleModel); an explicit mask tensor would route TE
                    # back to the unfused O(seq^2) backend.
                    attention_mask=None,
                    packed_seq_params=packed_seq_params,
                )

    # Apply temperature scaling only for sampling-oriented post-processors.
    # Loss computation should use unscaled logits.
    if isinstance(
        post_processing_fn,
        (LossPostProcessor, LogprobsPostProcessor, TopkLogitsPostProcessor),
    ):
        # Temperature scaling is element-wise, directly applying it here.
        # Other sampling parameters like top-k and top-p need the logits from whole vocabulary,
        # so applying them when gathering logits from vocab parallel (called in LossPostProcessor and LogprobsPostProcessor).
        apply_temperature_scaling(output_tensor, sampling_params)

    # Use type checking to dispatch to the correct post-processing method
    if isinstance(post_processing_fn, LossPostProcessor):
        post_processing_fn_wrapped = post_processing_fn(
            data_dict=data_dict,
            packed_seq_params=packed_seq_params,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
            global_draft_pass_counts=global_draft_pass_counts,
        )
    elif isinstance(post_processing_fn, LogprobsPostProcessor):
        assert original_seq_length is not None
        post_processing_fn_wrapped = post_processing_fn(
            data_dict=data_dict,
            input_ids=input_ids,
            cu_seqlens_padded=cu_seqlens_padded,
            original_seq_length=original_seq_length,
        )
    elif isinstance(post_processing_fn, TopkLogitsPostProcessor):
        assert original_seq_length is not None
        post_processing_fn_wrapped = post_processing_fn(
            data_dict=data_dict,
            cu_seqlens_padded=cu_seqlens_padded,
            original_seq_length=original_seq_length,
        )
    else:
        raise TypeError(
            f"Unknown post-processing function type: {type(post_processing_fn)}"
        )

    return output_tensor, post_processing_fn_wrapped


def megatron_forward_backward(
    model: GPTModel,
    data_iterator: Iterator[ProcessedMicrobatch],
    num_microbatches: int,
    seq_length: int,
    mbs: int,
    post_processing_fn: PostProcessingFunction,
    forward_only: bool = False,
    defer_fp32_logits: Optional[bool] = False,
    global_valid_seqs: Optional[torch.Tensor] = None,
    global_valid_toks: Optional[torch.Tensor] = None,
    global_draft_pass_counts: Optional[torch.Tensor] = None,
    sampling_params: Optional[TrainingSamplingParams] = None,
    straggler_timer: Optional[StragglerDetector] = None,
    draft_model: Optional[MegatronModule] = None,
    enable_hidden_capture: Optional[bool] = False,
    tap_channel: Optional[TapChannel] = None,
    draft_ttt_steps: int = 1,
    draft_aux_layer_indices: Optional[Tuple[int, ...]] = None,
    use_fused_linear_logprobs: bool = False,
    use_router_replay: bool = False,
    router_replay_train: bool = False,
) -> Any:
    """Execute forward and backward passes using Megatron's utilities.

    This is the main training loop function that coordinates forward and backward
    passes across multiple microbatches using Megatron's pipeline parallel
    execution framework.

    Args:
        model: The model to train
        data_iterator: Iterator yielding ProcessedMicrobatch objects (already processed)
        num_microbatches: Number of microbatches to process
        seq_length: Sequence length
        mbs: Micro batch size
        post_processing_fn: Post-processing function to post-process the logits
        forward_only: If True, skip backward pass
        defer_fp32_logits: Whether to skip the conversion of logits to fp32
        global_valid_seqs: Global valid sequence count for loss normalization
        global_valid_toks: Global valid token count for loss normalization
        sampling_params: Sampling parameters (top-k, top-p, temperature)
        straggler_timer: Straggler detector for profiling the forward pass
        draft_model: Draft model for online draft model training
        enable_hidden_capture: Whether to enable hidden state capture for draft model training

    Returns:
        Results from the forward/backward execution
    """
    forward_step = partial(
        forward_with_post_processing_fn,
        post_processing_fn=post_processing_fn,
        defer_fp32_logits=defer_fp32_logits,
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
        global_draft_pass_counts=global_draft_pass_counts,
        sampling_params=sampling_params,
        straggler_timer=straggler_timer,
        draft_model=draft_model,
        enable_hidden_capture=enable_hidden_capture,
        tap_channel=tap_channel,
        draft_ttt_steps=draft_ttt_steps,
        draft_aux_layer_indices=draft_aux_layer_indices,
        use_fused_linear_logprobs=use_fused_linear_logprobs,
        use_router_replay=use_router_replay,
        router_replay_train=router_replay_train,
    )
    if tap_channel is not None and enable_hidden_capture:
        # Outside the schedule: prune writer refs and broadcast this pass's
        # live mask-embedding row from the first stage to the draft stage.
        tap_channel.begin_pass(model)
    forward_backward_func = get_forward_backward_func()
    if use_router_replay:
        clear_router_replay(model)
    with suspend_activation_offload_for_forward_only(model, forward_only):
        try:
            return forward_backward_func(
                forward_step_func=forward_step,
                data_iterator=data_iterator,
                model=model,
                num_microbatches=num_microbatches,
                seq_length=seq_length,
                micro_batch_size=mbs,
                decoder_seq_length=seq_length,
                forward_only=forward_only,
            )
        finally:
            if use_router_replay:
                clear_router_replay(model)


class LossPostProcessor:
    def __init__(
        self,
        loss_fn: LossFunction,
        cfg: PolicyConfig,
        num_microbatches: int = 1,
        cp_normalize: bool = True,
        sampling_params: Optional[TrainingSamplingParams] = None,
        draft_model: Optional[MegatronModule] = None,
        prepare_fn: Optional[Callable[..., Any]] = None,
    ):
        """Build a per-microbatch loss post-processor for the Megatron train loop.

        Args:
            loss_fn: Loss function to wrap.
            cfg: Policy(-like) config; supplies sequence_packing / logprob_chunk_size.
            num_microbatches: Microbatch count, used to counteract Megatron's
                per-microbatch loss averaging.
            cp_normalize: Whether to divide the POLICY loss by the context-parallel
                size (compensates the CP logprob all-gather; the rank-local
                draft losses are exempt, see ``__call__``).
            sampling_params: Optional temperature / top-k/p for logprob losses.
            draft_model: Optional EAGLE draft model for distillation.
            prepare_fn: Optional override for the default ``prepare_loss_input``.
                Must accept ``(logits, data, loss_fn, vocab_parallel_rank,
                vocab_parallel_group, context_parallel_group)`` and return
                ``(loss_input, data)``; value models pass one that right-shifts
                and CP-all-gathers the scalar value-head output.
        """
        self.loss_fn = loss_fn
        self.cfg = cfg
        self.num_microbatches = num_microbatches
        self.cp_normalize = cp_normalize
        self.sampling_params = sampling_params
        self.prepare_fn = prepare_fn
        eagle_module = getattr(draft_model, "eagle_module", None)
        if eagle_module is not None:
            self.d2t = getattr(eagle_module, "d2t", None)
        else:
            # Block drafts (DFlash/DSpark) train full-vocab in v1 (no d2t).
            self.d2t = None

    def _wrap_with_draft_loss(
        self,
        loss_fn_wrapped: Any,
        prepare_fn: Callable[..., Any],
        data_dict: BatchedDataDict[Any],
        global_draft_pass_counts: Optional[torch.Tensor],
    ) -> "DraftLossWrapper":
        """Wrap the (possibly packing-wrapped) policy loss with the draft loss."""
        draft_cfg = self.cfg["draft"]
        raw_ttt_steps = draft_cfg.get("ttt_steps", 1)
        ttt_steps = 1 if raw_ttt_steps is None else int(raw_ttt_steps)
        if ttt_steps < 1:
            raise ValueError(
                f"policy.draft.ttt_steps must be >= 1, got {raw_ttt_steps}."
            )
        # ttt_steps reaches the forward independently (the draft_ttt_steps
        # argument threaded into megatron_forward_backward); a caller omitting
        # it with a multi-pass config would otherwise surface as a KeyError
        # far from the cause. Block drafts are dispatched on the logits key
        # the forward stashed — one config selector (speculator_type), read
        # only where the model is built.
        multi_pass = "student_logits_by_pass" in data_dict
        block_draft = "draft_block_logits" in data_dict
        if block_draft and ttt_steps > 1:
            raise ValueError(
                "policy.draft.ttt_steps > 1 applies to the eagle3 speculator "
                "only; block drafts predict a whole block per pass."
            )
        if not block_draft and multi_pass != (ttt_steps > 1):
            raise RuntimeError(
                f"draft forward produced "
                f"{'multi-pass' if multi_pass else 'single-pass'} logits "
                f"but policy.draft.ttt_steps={ttt_steps}; the "
                "draft_ttt_steps argument threaded into "
                "megatron_forward_backward disagrees with the config."
            )
        # Ctor kwargs for the draft LossFn (uniform shape; DraftLossWrapper
        # splats them into the selected class).
        if block_draft:
            draft_loss_kwargs: dict[str, Any] = {
                "slot_weights": resolve_block_draft_slot_weights(
                    draft_cfg.get("loss_weighting"), int(draft_cfg["gamma"])
                )
            }
            if "draft_confidence_pred" in data_dict:
                for key in (
                    "ce_loss_alpha",
                    "tv_loss_alpha",
                    "confidence_head_alpha",
                ):
                    if draft_cfg.get(key) is not None:
                        draft_loss_kwargs[key] = float(draft_cfg[key])
        else:
            draft_loss_kwargs = {"pass_weights": draft_cfg.get("ttt_pass_weights")}
        if draft_cfg.get("loss_seq_chunk_size") is not None:
            draft_loss_kwargs["seq_chunk_size"] = int(draft_cfg["loss_seq_chunk_size"])
        return DraftLossWrapper(
            loss_fn=loss_fn_wrapped,
            prepare_fn=prepare_fn,
            data_dict=data_dict,
            loss_weight=float(draft_cfg["loss_weight"]),
            draft_loss_kwargs=draft_loss_kwargs,
            global_draft_pass_counts=global_draft_pass_counts,
            vocab_parallel_rank=get_tensor_model_parallel_rank(),
            vocab_parallel_group=get_tensor_model_parallel_group(),
            context_parallel_group=get_context_parallel_group(),
        )

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        packed_seq_params: Optional[PackedSeqParams] = None,
        global_valid_seqs: Optional[torch.Tensor] = None,
        global_valid_toks: Optional[torch.Tensor] = None,
        global_draft_pass_counts: Optional[torch.Tensor] = None,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, Any]]]:
        """Create a loss post-processing function for training.

        This function wraps a loss function with the necessary context and parameters
        to compute loss and metrics from model outputs. It handles sequence packing
        and context parallelism normalization.

        Args:
            data_dict: Batched data dictionary for the current microbatch
            packed_seq_params: Parameters for packed sequences (optional)
            global_valid_seqs: Global valid sequence count for loss normalization
            global_valid_toks: Global valid token count for loss normalization

        Returns:
            Callable: Function that takes output tensor and returns (loss, metrics) tuple
        """
        # A custom prepare_fn (e.g. value models) overrides the default logit prep.
        logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)
        if self.prepare_fn is not None:
            prepare_loss_input_wrapped = self.prepare_fn
        else:
            prepare_loss_input_wrapped = partial(
                prepare_loss_input,
                sampling_params=self.sampling_params,
                d2t=self.d2t,
                chunk_size=logprob_chunk_size,
            )

        # wrap loss function with loss input preparation
        pack_sequences = self.cfg["sequence_packing"]["enabled"]
        has_draft_logits = (
            "student_logits" in data_dict
            or "student_logits_by_pass" in data_dict
            or "draft_block_logits" in data_dict
        )
        if pack_sequences and packed_seq_params is not None:
            fuse_loss = self.cfg.get("sequence_packing", {}).get("fuse_loss", False)
            if has_draft_logits and fuse_loss:
                # The fused policy-loss prep never materializes full logits in
                # a form the draft's soft-CE teacher prep was validated with.
                raise NotImplementedError(
                    "Draft-model training with sequence packing requires "
                    "sequence_packing.fuse_loss=false."
                )
            if fuse_loss:
                # The fused path prepares loss via prepare_packed_loss_input and
                # cannot honor a custom prepare_fn (e.g. the value model's); guard
                # rather than silently bypass it.
                assert self.prepare_fn is None, (
                    "sequence_packing.fuse_loss=true does not support a custom "
                    "prepare_fn (e.g. the value model's value-specific prep). "
                    "Disable fuse_loss for the value model."
                )
                wrapper_cls = SequencePackingFusionLossWrapper
                prepare_fn = partial(
                    prepare_packed_loss_input,
                    sampling_params=self.sampling_params,
                    chunk_size=logprob_chunk_size,
                )
            else:
                wrapper_cls = SequencePackingLossWrapper
                prepare_fn = prepare_loss_input_wrapped

            loss_fn_wrapped = wrapper_cls(
                loss_fn=self.loss_fn,
                prepare_fn=prepare_fn,
                cu_seqlens_q=packed_seq_params.cu_seqlens_q,
                cu_seqlens_q_padded=packed_seq_params.cu_seqlens_q_padded,
                vocab_parallel_rank=get_tensor_model_parallel_rank(),
                vocab_parallel_group=get_tensor_model_parallel_group(),
                context_parallel_group=get_context_parallel_group(),
            )
        else:
            loss_fn_wrapped = partial(
                wrap_loss_fn_with_input_preparation,
                loss_fn=self.loss_fn,
                prepare_fn=prepare_loss_input_wrapped,
                vocab_parallel_rank=get_tensor_model_parallel_rank(),
                vocab_parallel_group=get_tensor_model_parallel_group(),
                context_parallel_group=get_context_parallel_group(),
            )

        if self.cp_normalize:
            # Policy loss only. Under CP every rank evaluates the FULL-sequence
            # policy loss (from_parallel_logits_to_logprobs all-gathers the
            # logprobs, and that all-gather's backward hands each local logit
            # cp_size copies of its gradient), hence the 1/cp_size. The draft
            # losses are rank-local numerators over a global denominator and
            # reach the global gradient through the dp_cp grad all-reduce
            # alone, so they must be added AFTER this division: dividing them
            # too shrinks the draft gradient by cp_size while the draft
            # metrics (explicitly CP-reduced) keep looking right.
            cp_size = get_context_parallel_world_size()
            policy_loss_fn = loss_fn_wrapped

            def _div_policy_loss_by_cp_size(*args, **kwargs):
                loss, metrics = policy_loss_fn(*args, **kwargs)
                return loss / cp_size, metrics

            loss_fn_wrapped = _div_policy_loss_by_cp_size

        if has_draft_logits:
            # The draft loss wraps AROUND the (packing-wrapped, CP-normalized)
            # policy loss: with packing the policy part iterates subsequences
            # while the draft part runs once over the whole packed row
            # (per-pass teacher roll + coord-gathered masks; see
            # DraftCrossEntropyLossFn's packed mode).
            loss_fn_wrapped = self._wrap_with_draft_loss(
                loss_fn_wrapped,
                prepare_loss_input_wrapped,
                data_dict,
                global_draft_pass_counts,
            )

        loss_fn_wrapped = partial(
            loss_fn_wrapped,
            data=data_dict,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
        )

        # Counteract Megatron's default loss averaging in schedules.py,
        # which applies (* cp_size / num_microbatches) to the loss.
        cp_size = get_context_parallel_world_size()
        num_microbatches = self.num_microbatches
        loss_fn_before_mcore_scaling = loss_fn_wrapped

        def _counteract_mcore_loss_averaging(*args, **kwargs):
            loss, metrics = loss_fn_before_mcore_scaling(*args, **kwargs)
            return loss * num_microbatches / cp_size, metrics

        loss_fn_wrapped = _counteract_mcore_loss_averaging

        return loss_fn_wrapped


class LogprobsPostProcessor:
    def __init__(
        self,
        cfg: PolicyConfig,
        sampling_params: Optional[TrainingSamplingParams] = None,
        use_fused_linear_logprobs: bool = False,
    ):
        self.cfg = cfg
        self.sampling_params = sampling_params
        self.use_fused_linear_logprobs = use_fused_linear_logprobs

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        input_ids: torch.Tensor,
        cu_seqlens_padded: torch.Tensor,
        original_seq_length: int,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Create a post-processing function that computes token log probabilities.

        This function returns a processor that takes model logits and converts them
        to token-level log probabilities, handling both packed and unpacked sequences.

        Args:
            data_dict: Batched data dictionary containing input sequences
            input_ids: Processed input token IDs
            cu_seqlens_padded: Cumulative sequence lengths for packed sequences
            original_seq_length: Sequence width before dense padding was applied

        Returns:
            Callable: Function that takes output tensor and returns (dummy_loss, {"logprobs": token_logprobs})
        """
        unpacked_input_ids = data_dict["input_ids"]

        def processor_fn_inner(output_tensor):
            if self.use_fused_linear_logprobs:
                token_logprobs = output_tensor.to(torch.float32)
                token_logprobs = token_logprobs[:, : original_seq_length - 1]
            elif self.cfg["sequence_packing"]["enabled"]:
                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)
                token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                    output_tensor,
                    target=input_ids,
                    cu_seqlens_padded=cu_seqlens_padded,
                    unpacked_seqlen=original_seq_length,
                    vocab_start_index=tp_rank * output_tensor.shape[-1],
                    vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                    group=tp_grp,
                    inference_only=True,
                    cp_group=get_context_parallel_group(),
                    chunk_size=logprob_chunk_size,
                    sampling_params=self.sampling_params,
                )
            else:
                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)
                token_logprobs = from_parallel_logits_to_logprobs(
                    output_tensor,
                    target=unpacked_input_ids,
                    vocab_start_index=tp_rank * output_tensor.shape[-1],
                    vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                    tp_group=tp_grp,
                    inference_only=True,
                    chunk_size=logprob_chunk_size,
                    sampling_params=self.sampling_params,
                )

            # Prepend 0 logprob for first token to maintain same sequence length as input
            token_logprobs = torch.cat(
                [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
            )

            # handle top-k/top-p filtering for logprobs, only used for ClippedPGLossFn now
            if need_top_k_or_top_p_filtering(self.sampling_params):
                mask = data_dict["token_mask"] * data_dict["sample_mask"].unsqueeze(-1)
                token_logprobs = mask_out_neg_inf_logprobs(
                    token_logprobs, mask, "prev_logprobs"
                )

            token_logprobs = token_logprobs[:, :original_seq_length]

            return torch.tensor(0.0, device=token_logprobs.device), {
                "logprobs": token_logprobs
            }

        return processor_fn_inner


class TopkLogitsPostProcessor:
    def __init__(self, cfg: PolicyConfig, k: int):
        self.cfg = cfg
        self.k = k

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        cu_seqlens_padded: torch.Tensor,
        original_seq_length: int,
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Create a post-processing function that computes top-k logits and indices.

        This function returns a processor that extracts the top-k highest logits
        and their corresponding vocabulary indices from model outputs. It handles
        tensor parallelism, context parallelism, and sequence packing.

        Args:
            data_dict: Batched data dictionary
            cu_seqlens_padded: Cumulative sequence lengths for packed sequences
            original_seq_length: Sequence width before dense padding was applied

        Returns:
            Callable: Function that takes output tensor and returns
                      (dummy_loss, {"topk_logits": values, "topk_indices": indices})
        """
        pack = self.cfg["sequence_packing"]["enabled"]
        cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
        unpacked_seqlen = data_dict["input_ids"].shape[1]
        seq_lengths = data_dict["input_lengths"]

        def processor_fn_inner(output_tensor):
            tp_grp = get_tensor_model_parallel_group()
            tp_rank = get_tensor_model_parallel_rank()
            vocab_shard_size = output_tensor.shape[-1]
            vocab_start_index = tp_rank * vocab_shard_size

            chunk_size = None
            if "logprob_chunk_size" in self.cfg:
                chunk_size = self.cfg["logprob_chunk_size"]

            topk_vals_local, topk_idx_local = distributed_vocab_topk(
                output_tensor,
                self.k,
                tp_grp,
                vocab_start_index=vocab_start_index,
                vocab_end_index=vocab_start_index + vocab_shard_size,
                chunk_size=chunk_size,
            )

            if self.cfg["megatron_cfg"]["context_parallel_size"] > 1:
                cp_grp = get_context_parallel_group()
                if pack:
                    # Per-sequence CP allgather following packed-sequence logic
                    batch_size = data_dict["input_ids"].shape[0]
                    total_packed_len = int(cu_seqlens_padded[-1].item())

                    topk_vals_full = torch.zeros(
                        (1, total_packed_len, self.k),
                        dtype=topk_vals_local.dtype,
                        device=topk_vals_local.device,
                    )
                    topk_idx_full = torch.zeros(
                        (1, total_packed_len, self.k),
                        dtype=topk_idx_local.dtype,
                        device=topk_idx_local.device,
                    )

                    for i in range(batch_size):
                        start_idx = int(cu_seqlens_padded[i].item())
                        end_idx = int(cu_seqlens_padded[i + 1].item())
                        if end_idx > start_idx:
                            local_vals_slice = topk_vals_local[
                                :, start_idx // cp_size : end_idx // cp_size, :
                            ]
                            local_idx_slice = topk_idx_local[
                                :, start_idx // cp_size : end_idx // cp_size, :
                            ]
                            gathered_vals = allgather_cp_sharded_tensor(
                                local_vals_slice, cp_grp, seq_dim=1
                            )
                            gathered_idx = allgather_cp_sharded_tensor(
                                local_idx_slice, cp_grp, seq_dim=1
                            )
                            # Some kernels may return [X, Y, k] where X*Y = (end_idx - start_idx).
                            # Flatten leading dims and reshape to [1, expected_len, k] to match target.
                            expected_len = end_idx - start_idx
                            if (
                                gathered_vals.dim() == 3
                                and gathered_vals.shape[1] != expected_len
                            ):
                                gathered_vals = gathered_vals.reshape(
                                    1, expected_len, gathered_vals.shape[-1]
                                )
                            if (
                                gathered_idx.dim() == 3
                                and gathered_idx.shape[1] != expected_len
                            ):
                                gathered_idx = gathered_idx.reshape(
                                    1, expected_len, gathered_idx.shape[-1]
                                )
                            topk_vals_full[:, start_idx:end_idx, :] = gathered_vals
                            topk_idx_full[:, start_idx:end_idx, :] = gathered_idx
                else:
                    # Sequence packing must be enabled when CP > 1
                    raise RuntimeError(
                        "Context Parallelism (CP>1) requires sequence packing to be enabled."
                    )
            else:
                topk_vals_full = topk_vals_local
                topk_idx_full = topk_idx_local

            if pack:
                batch_size = data_dict["input_ids"].shape[0]
                out_vals = torch.zeros(
                    (batch_size, unpacked_seqlen, self.k),
                    dtype=topk_vals_full.dtype,
                    device=topk_vals_full.device,
                )
                out_idx = torch.zeros(
                    (batch_size, unpacked_seqlen, self.k),
                    dtype=topk_idx_full.dtype,
                    device=topk_idx_full.device,
                )
                for i in range(batch_size):
                    seq_len = int(seq_lengths[i].item())
                    start_idx = int(cu_seqlens_padded[i].item())
                    if seq_len > 0:
                        out_vals[i, :seq_len, :] = topk_vals_full[
                            0, start_idx : start_idx + seq_len, :
                        ]
                        out_idx[i, :seq_len, :] = topk_idx_full[
                            0, start_idx : start_idx + seq_len, :
                        ]
                return output_tensor.new_zeros(()), {
                    "topk_logits": out_vals,
                    "topk_indices": out_idx,
                }
            else:
                return output_tensor.new_zeros(()), {
                    "topk_logits": topk_vals_full[:, :original_seq_length],
                    "topk_indices": topk_idx_full[:, :original_seq_length],
                }

        return processor_fn_inner


def aggregate_training_statistics(
    all_mb_metrics: List[Dict[str, Any]],
    losses: List[float],
    data_parallel_group: torch.distributed.ProcessGroup,
) -> Tuple[Dict[str, List[Any]], torch.Tensor]:
    """Aggregate training statistics across microbatches and data-parallel ranks.

    Computes a global loss by all-reducing per-gradient-buffer losses across the
    data-parallel group, then collects per-microbatch metrics into lists keyed by
    metric name.

    Args:
        all_mb_metrics: List of metric dicts from each microbatch.
        losses: List of per-gradient-buffer scalar losses on this rank.
        data_parallel_group: The data-parallel process group for all-reduce.

    Returns:
        Tuple of:
            - mb_metrics: Dict mapping metric names to lists of values across microbatches.
            - global_loss: Tensor of losses summed across all data-parallel ranks.
    """
    # Compute global loss across all data-parallel ranks
    with torch.no_grad():
        global_loss = torch.tensor(losses, device="cuda")
        torch.distributed.all_reduce(
            global_loss,
            op=torch.distributed.ReduceOp.SUM,
            group=data_parallel_group,
        )

    # Aggregate metrics across all microbatches
    mb_metrics: Dict[str, List[Any]] = defaultdict(list)
    for m in all_mb_metrics:
        for k, v in m.items():
            mb_metrics[k].append(v)

    return dict(mb_metrics), global_loss
