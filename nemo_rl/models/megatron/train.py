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
from contextlib import nullcontext
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
from megatron.core.utils import StragglerDetector

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

    if defer_fp32_logits:
        additional_kwargs["fp32_output"] = False
    if use_fused_linear_logprobs:
        additional_kwargs["labels"] = input_ids_cp_sharded
        # Only pass this kwarg when linear CE fusion is enabled. Older Megatron-LM
        # GPTModel.forward signatures do not accept it.
        additional_kwargs["return_logprobs_for_linear_ce_fusion"] = True

    # PATCH(golden score-checksum): timeline of which theta the train model holds.
    # Hashes a marker param (layer-0 router weight) on every forward; prints only on
    # value CHANGE so the log shows exactly when weights transition across
    # gen/score/train phases. Compare against [NRL_REFIT_CHECKSUM] swap-time hashes.
    import os as _os_sc
    if _os_sc.environ.get("NRL_SCORE_CHECKSUM", "0") == "1":
        import hashlib as _hl
        _st = getattr(model_forward, "_nrl_sc", None)
        if _st is None:
            # router.weight is bitwise-frozen under routing replay (no grads) —
            # use a param guaranteed to be in the loss path.
            _mark = None
            for _n, _p in model.named_parameters():
                if "layers.0." in _n and "linear_qkv" in _n and _n.endswith("weight"):
                    _mark = (_n, _p)
                    break
            if _mark is None:
                _mark = next(iter(model.named_parameters()))
            model_forward._nrl_sc = _st = {"n": 0, "last": None, "mark": _mark}
        _st["n"] += 1
        _mn, _mp = _st["mark"]
        _h = _hl.md5(
            _mp.data.detach().float().cpu().contiguous().numpy().tobytes()
        ).hexdigest()[:12]
        if _h != _st["last"]:
            import torch.distributed as _dist_sc
            _r = _dist_sc.get_rank() if _dist_sc.is_initialized() else -1
            print(
                f"[NRL_SCORE_CHECKSUM] rank={_r} call={_st['n']} {_mn} md5={_h} (CHANGED)",
                flush=True,
            )
            _st["last"] = _h

    # PATCH(NRL_SCORE_LAYERDUMP): in-situ per-layer residual-stream dump of the
    # SCORING forward — one-shot hooks on every decoder layer's pre-attn norm input
    # (the exact residual the engine's gdump records). Faithful localization of the
    # infopt-vs-TE-scoring logit gap: diff engine gdump vs this per (position,layer).
    import os as _os_sld
    if _os_sld.environ.get("NRL_SCORE_LAYERDUMP", "") and not getattr(model_forward, "_nrl_sld", False):
        model_forward._nrl_sld = True
        _mm = model
        while hasattr(_mm, "module"):
            _mm = _mm.module
        _dec = getattr(_mm, "decoder", None)
        if _dec is not None and hasattr(_dec, "layers"):
            import torch as _t_sld
            _buf = {}
            builtins_sld = __import__("builtins")
            builtins_sld._NRL_SLD_BUF = _buf
            # capture ALL rows of the first-token dim so offline we can pick any seq;
            # residual is [S,B,H] (megatron) — keep [S,B,H] transposed to [B,S,H]
            builtins_sld._NRL_SLD_FIRES = [0, None]
            def _mk(li):
                def hook(mod, args, kwargs, out):
                    x = None
                    if args and _t_sld.is_tensor(args[0]):
                        x = args[0]
                    elif isinstance(kwargs, dict) and _t_sld.is_tensor(kwargs.get("hidden_states")):
                        x = kwargs["hidden_states"]
                    builtins_sld._NRL_SLD_FIRES[0] += 1
                    if x is None:
                        builtins_sld._NRL_SLD_FIRES[1] = f"no-tensor args={len(args)} kw={list(kwargs)[:4]}"
                        return
                    if x.dim() != 3:
                        builtins_sld._NRL_SLD_FIRES[1] = f"dim={x.dim()} shape={tuple(x.shape)}"
                        return
                    # [S,B,H] -> [B,S,H]; store first 16 microbatches (all local seqs)
                    _mb = getattr(builtins_sld, "_NRL_SLD_MB", 0)
                    if _mb < 16:
                        _buf[(_mb, li)] = x.transpose(0, 1).detach().float().cpu().clone()
                return hook
            for _li, _lyr in enumerate(_dec.layers):
                _lyr.register_forward_hook(_mk(_li), with_kwargs=True)
            # save the batch input_ids so offline we can map batch-row -> jsonl seq
            builtins_sld._NRL_SLD_IDS = []
            def _save_sld(*_a):
                try:
                    import torch.distributed as _d_sld
                    _rk = _d_sld.get_rank() if _d_sld.is_initialized() else 0
                    _t_sld.save({"layers": {k: v for k, v in _buf.items()},
                                 "ids": getattr(builtins_sld, "_NRL_SLD_IDS", None)},
                                _os_sld.environ["NRL_SCORE_LAYERDUMP"] + f"/score_layers_rank{_rk}.pt")
                    print(f"[NRL_SCORE_LAYERDUMP] saved {len(_buf)} layers rank{_rk} "
                          f"fires={getattr(builtins_sld, '_NRL_SLD_FIRES', None)}", flush=True)
                except Exception as _e:
                    print(f"[NRL_SCORE_LAYERDUMP] save failed: {_e}", flush=True)
            import atexit as _ax_sld
            _ax_sld.register(_save_sld)

    # PATCH(NRL_MOE_DUMP, K2v3): scoring-side counterpart of the engine's moeA dump —
    # per-layer MoE INPUT (post-norm hidden, = engine vllm_fused_moe `hs`) and router
    # output (probs + routing map) for layers 0..3, first 4 scoring fires, all ranks.
    # Matched-point bracket: MoE-in matches + next-layer-in differs => MoE block
    # convicted; MoE-in differs => attention path upstream.
    if _os_sld.environ.get("NRL_MOE_DUMP_DEADPATH", "") and not getattr(model_forward, "_nrl_moedump", False):
        model_forward._nrl_moedump = True
        _mm2 = model
        while hasattr(_mm2, "module"):
            _mm2 = _mm2.module
        _dec2 = getattr(_mm2, "decoder", None)
        if _dec2 is not None and hasattr(_dec2, "layers"):
            import torch as _t_md
            _mdst = {"fires": {}}
            def _mk_moein(li):
                def hook(mod, args, kwargs):
                    x = args[0] if args and _t_md.is_tensor(args[0]) else kwargs.get("hidden_states")
                    if x is None or x.dim() != 3:
                        return
                    f = _mdst["fires"].setdefault(li, 0)
                    if f < 4:
                        _mdst["fires"][li] = f + 1
                        _mdst[("in", f, li)] = x.transpose(0, 1).reshape(-1, x.shape[-1])[:512].detach().float().cpu().clone()
                return hook
            def _mk_router(li):
                def hook(mod, args, kwargs, out):
                    f = _mdst["fires"].get(li, 1) - 1
                    if f < 0 or f >= 4 or ("rt", f, li) in _mdst:
                        pass
                    try:
                        probs, rmap = out[0], out[1]
                        if ("rt", f, li) not in _mdst and 0 <= f < 4:
                            _mdst[("rt", f, li)] = (probs.reshape(-1, probs.shape[-1])[:512].detach().float().cpu().clone(),
                                                    rmap.reshape(-1, rmap.shape[-1])[:512].detach().cpu().clone())
                    except Exception:
                        pass
                return hook
            # module hooks do NOT fire here (megatron calls submodule.forward directly,
            # bypassing __call__) -> wrap the BOUND forward methods instead.
            _nmlp, _nrtr = 0, 0
            def _wrap_mlp(li, mod):
                _orig = mod.forward
                def fwd(*args, **kwargs):
                    x = args[0] if args and _t_md.is_tensor(args[0]) else kwargs.get("hidden_states")
                    _dbg = _mdst.setdefault("dbg", [])
                    if len(_dbg) < 8:
                        _dbg.append((li, None if x is None else tuple(x.shape), len(args), sorted(kwargs)[:3]))
                    if x is not None and x.dim() >= 2:
                        f = _mdst["fires"].setdefault(li, 0)
                        if f < 4:
                            _mdst["fires"][li] = f + 1
                            _mdst[("in", f, li)] = x.reshape(-1, x.shape[-1])[:512].detach().float().cpu().clone()
                    return _orig(*args, **kwargs)
                mod.forward = fwd
            def _wrap_router(li, mod):
                _orig = mod.forward
                def fwd(*args, **kwargs):
                    out = _orig(*args, **kwargs)
                    try:
                        f = max(_mdst["fires"].get(li, 1) - 1, 0)
                        if f < 4 and ("rt", f, li) not in _mdst:
                            probs, rmap = out[0], out[1]
                            _mdst[("rt", f, li)] = (probs.reshape(-1, probs.shape[-1])[:512].detach().float().cpu().clone(),
                                                    rmap.reshape(-1, rmap.shape[-1])[:512].detach().cpu().clone())
                    except Exception:
                        pass
                    return out
                mod.forward = fwd
            for _li2, _lyr2 in enumerate(_dec2.layers):
                if _li2 >= 4:
                    break
                _mlp = getattr(_lyr2, "mlp", None)
                if _mlp is not None:
                    _wrap_mlp(_li2, _mlp)
                    _nmlp += 1
                    _rtr = getattr(_mlp, "router", None)
                    if _rtr is not None:
                        _wrap_router(_li2, _rtr)
                        _nrtr += 1
            print(f"[NRL_MOE_DUMP:B] method-wrapped mlp={_nmlp} router={_nrtr} "
                  f"n_layers={len(_dec2.layers)}", flush=True)
            def _save_md(*_a):
                try:
                    import torch.distributed as _d_md
                    _rk2 = _d_md.get_rank() if _d_md.is_initialized() else 0
                    payload = {k: v for k, v in _mdst.items() if k != "fires"}
                    payload["_dbg"] = _mdst.get("dbg", [])
                    _t_md.save(payload, _os_sld.environ["NRL_MOE_DUMP"] + f"/moeB_rank{_rk2}.pt")
                    print(f"[NRL_MOE_DUMP:B] saved {len(payload)} entries rank{_rk2}", flush=True)
                except Exception as _e:
                    print(f"[NRL_MOE_DUMP:B] save failed: {_e}", flush=True)
            import atexit as _ax_md
            _ax_md.register(_save_md)
            builtins_sld._NRL_SLD_SAVE = _save_sld
            print(f"[NRL_SCORE_LAYERDUMP] hooked {len(_dec.layers)} decoder layers", flush=True)

    with straggler_timer() if straggler_timer is not None else nullcontext():
        output_tensor = model(
            input_ids=input_ids_cp_sharded,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **additional_kwargs,
            **multimodal_data,
        )

    # dump the scoring residuals after the FIRST scoring forward (one-shot) so the
    # save happens even without atexit (SIGTERM/ray teardown races)
    if _os_sld.environ.get("NRL_SCORE_LAYERDUMP", "") and getattr(model_forward, "_nrl_sld", False):
        _bi2 = __import__("builtins")
        _mb = getattr(_bi2, "_NRL_SLD_MB", 0)
        if _mb < 16:
            _bi2._NRL_SLD_IDS.append(input_ids_cp_sharded.detach().cpu().clone())
        _bi2._NRL_SLD_MB = _mb + 1
        _sv = getattr(_bi2, "_NRL_SLD_SAVE", None)
        if _sv is not None and _bi2._NRL_SLD_MB == 16 and not getattr(model_forward, "_nrl_sld_done", False):
            model_forward._nrl_sld_done = True
            _sv()

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


def forward_with_post_processing_fn(
    data_iterator: Iterator[ProcessedMicrobatch],
    model: GPTModel,
    post_processing_fn: PostProcessingFunction,
    defer_fp32_logits: Optional[bool] = False,
    global_valid_seqs: Optional[torch.Tensor] = None,
    global_valid_toks: Optional[torch.Tensor] = None,
    sampling_params: Optional[TrainingSamplingParams] = None,
    straggler_timer: Optional[StragglerDetector] = None,
    draft_model: Optional[MegatronModule] = None,
    enable_hidden_capture: Optional[bool] = False,
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

    if use_router_replay:
        if routed_experts_cp_sharded is None:
            raise RuntimeError(
                "Router replay is enabled but routed_experts is missing from the microbatch."
            )
        set_router_replay_forward(model, routed_experts_cp_sharded)

    # Insert hook to capture hidden states and embeddings for draft model training if draft_model is provided
    capture_context, capture = get_capture_context(model, enable_hidden_capture)
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
        from megatron.core.transformer.multi_token_prediction import roll_tensor

        captured_states = capture.get_captured_states()
        shifted_input_embeds = roll_tensor(
            captured_states.inputs_embeds,
            shifts=-1,
            dims=0,
            cp_group=get_context_parallel_group(),
        )[0]
        data_dict["student_logits"] = draft_model(
            hidden_states=captured_states.hidden_states,
            input_embeds=shifted_input_embeds,
            attention_mask=attention_mask,
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
        )
    elif isinstance(post_processing_fn, LogprobsPostProcessor):
        post_processing_fn_wrapped = post_processing_fn(
            data_dict=data_dict,
            input_ids=input_ids,
            cu_seqlens_padded=cu_seqlens_padded,
        )
    elif isinstance(post_processing_fn, TopkLogitsPostProcessor):
        post_processing_fn_wrapped = post_processing_fn(
            data_dict=data_dict,
            cu_seqlens_padded=cu_seqlens_padded,
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
    sampling_params: Optional[TrainingSamplingParams] = None,
    straggler_timer: Optional[StragglerDetector] = None,
    draft_model: Optional[MegatronModule] = None,
    enable_hidden_capture: Optional[bool] = False,
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
        sampling_params=sampling_params,
        straggler_timer=straggler_timer,
        draft_model=draft_model,
        enable_hidden_capture=enable_hidden_capture,
        use_fused_linear_logprobs=use_fused_linear_logprobs,
        use_router_replay=use_router_replay,
        router_replay_train=router_replay_train,
    )
    forward_backward_func = get_forward_backward_func()
    if use_router_replay:
        clear_router_replay(model)
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
            cp_normalize: Whether to divide the loss by the context-parallel size.
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
        if draft_model is not None and draft_model.eagle_module is not None:
            self.d2t = getattr(draft_model.eagle_module, "d2t", None)
        else:
            self.d2t = None

    def __call__(
        self,
        data_dict: BatchedDataDict[Any],
        packed_seq_params: Optional[PackedSeqParams] = None,
        global_valid_seqs: Optional[torch.Tensor] = None,
        global_valid_toks: Optional[torch.Tensor] = None,
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
        if pack_sequences and packed_seq_params is not None:
            fuse_loss = self.cfg.get("sequence_packing", {}).get("fuse_loss", False)
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
            if "student_logits" in data_dict:
                loss_fn_wrapped = DraftLossWrapper(
                    loss_fn=loss_fn_wrapped,
                    prepare_fn=prepare_loss_input_wrapped,
                    data_dict=data_dict,
                    loss_weight=float(self.cfg["draft"]["loss_weight"]),
                    vocab_parallel_rank=get_tensor_model_parallel_rank(),
                    vocab_parallel_group=get_tensor_model_parallel_group(),
                    context_parallel_group=get_context_parallel_group(),
                )

        loss_fn_wrapped = partial(
            loss_fn_wrapped,
            data=data_dict,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
        )

        if self.cp_normalize:
            cp_size = get_context_parallel_world_size()
            prev_loss_fn = loss_fn_wrapped

            def _div_by_cp_size(*args, **kwargs):
                loss, metrics = prev_loss_fn(*args, **kwargs)
                return loss / cp_size, metrics

            loss_fn_wrapped = _div_by_cp_size

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
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Create a post-processing function that computes token log probabilities.

        This function returns a processor that takes model logits and converts them
        to token-level log probabilities, handling both packed and unpacked sequences.

        Args:
            data_dict: Batched data dictionary containing input sequences
            input_ids: Processed input token IDs
            cu_seqlens_padded: Cumulative sequence lengths for packed sequences

        Returns:
            Callable: Function that takes output tensor and returns (dummy_loss, {"logprobs": token_logprobs})
        """
        unpacked_input_ids = data_dict["input_ids"]
        original_seq_length = unpacked_input_ids.shape[1]

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
            elif (
                __import__("os").environ.get("NRL_FUSED_TRAIN_LOGPROB", "off")
                in ("1", "fused")
                and torch.distributed.get_world_size(
                    group=get_tensor_model_parallel_group()
                )
                == 1
                and torch.distributed.get_world_size(
                    group=get_context_parallel_group()
                )
                == 1
            ):
                # PATCH(S7 fused-logprob, ported from v0.5.0 train.py): match mcore
                # generation's exact logprob op sequence (logits.float() -> fused
                # F.log_softmax -> gather) instead of the hand-rolled distributed
                # logsumexp. The two routines round differently (~1 fp32 ulp on
                # ~30% of tokens) and are the residual train/gen logprob floor at
                # TP=1. Only valid at TP=1 & CP=1 (full logit row on one rank).
                if not getattr(LogprobsPostProcessor, "_nrl_s7_banner", False):
                    LogprobsPostProcessor._nrl_s7_banner = True
                    print("[NRL_FUSED_TRAIN_LOGPROB] fused log_softmax->gather logprob path ACTIVE", flush=True)
                _full_lp = torch.nn.functional.log_softmax(
                    output_tensor.float(), dim=-1
                )
                token_logprobs = (
                    _full_lp[:, :-1]
                    .gather(
                        -1,
                        unpacked_input_ids[:, 1:]
                        .to(output_tensor.device)
                        .unsqueeze(-1),
                    )
                    .squeeze(-1)
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
    ) -> Callable[[torch.Tensor], Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Create a post-processing function that computes top-k logits and indices.

        This function returns a processor that extracts the top-k highest logits
        and their corresponding vocabulary indices from model outputs. It handles
        tensor parallelism, context parallelism, and sequence packing.

        Args:
            data_dict: Batched data dictionary
            cu_seqlens_padded: Cumulative sequence lengths for packed sequences

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
                    "topk_logits": topk_vals_full,
                    "topk_indices": topk_idx_full,
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
