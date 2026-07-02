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

import gc
import os
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from typing import Any, Iterator, Optional, TypeVar

import ray
import torch
from megatron.bridge.training.checkpointing import (
    maybe_finalize_async_save,
    save_checkpoint,
)
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.utils.train_utils import (
    LinearForLastLayer,
    create_value_head_hook,
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.core import parallel_state
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as custom_FSDP,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_world_size,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import get_rerun_state_machine
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import allgather_cp_sharded_tensor
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.megatron.common import (
    broadcast_tensor,
    get_moe_metrics,
)
from nemo_rl.models.megatron.data import (
    get_microbatch_iterator,
    process_global_batch,
)
from nemo_rl.models.megatron.setup import (
    finalize_megatron_setup,
    handle_model_import,
    make_policy_like_config,
    setup_distributed,
    setup_model_and_optimizer,
    validate_and_set_config,
    validate_model_paths,
)
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.models.value.config import ValueConfig
from nemo_rl.models.value.interfaces import ValueOutputSpec
from nemo_rl.utils.nsys import wrap_with_nvtx_name

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def forward_step_value(
    state,
    global_valid_seqs,
    global_valid_toks,
    data_iterator,
    model,
    *,
    loss_fn,
    pack_sequences=False,
    defer_fp32_logits=None,
    cp_normalize=True,
    num_microbatches=1,
    policy_cfg=None,
):
    """Forward step for the value model.

    The LM head is replaced at build time by a ``LinearForLastLayer``
    (hidden_size -> 1) value head (see ``create_value_head_hook``), so
    ``model(...)`` returns per-token values directly — the head already gathers
    sequence-parallel shards.
    """
    from nemo_rl.algorithms.loss import SequencePackingLossWrapper

    straggler_timer = state.straggler_timer

    # Get the pre-processed microbatch from the iterator
    processed_mb = next(data_iterator)

    data_dict = processed_mb.data_dict
    input_ids_cp_sharded = processed_mb.input_ids_cp_sharded
    attention_mask = processed_mb.attention_mask
    position_ids = processed_mb.position_ids
    packed_seq_params = processed_mb.packed_seq_params

    additional_kwargs = {}
    if packed_seq_params is not None:
        additional_kwargs["packed_seq_params"] = packed_seq_params
    if defer_fp32_logits:
        additional_kwargs["fp32_output"] = False

    with straggler_timer:
        output_tensor = model(
            input_ids=input_ids_cp_sharded,
            position_ids=position_ids,
            attention_mask=attention_mask,
            **additional_kwargs,
        )

    is_packed = pack_sequences and packed_seq_params is not None

    # Head-owning stage returns values; others pass hidden states through.
    if is_pipeline_last_stage(ignore_virtual=True):
        if is_packed:
            # Packed [1, T // CP, 1] -> [1, T // CP]. The per-sequence shift (and CP
            # all-gather) happens in the loss wrapper's prepare_fn below.
            output_tensor = output_tensor.squeeze(-1)
        else:
            # [B, S]: shift right by 1 so values[t] = V(state before token t).
            values = output_tensor.squeeze(-1)
            output_tensor = torch.cat(
                [torch.zeros_like(values[:, :1]), values[:, :-1]], dim=1
            )

    if is_packed:
        loss_fn = SequencePackingLossWrapper(
            loss_fn=loss_fn,
            prepare_fn=_value_packed_loss_prepare_fn,
            cu_seqlens_q=packed_seq_params.cu_seqlens_q,
            cu_seqlens_q_padded=packed_seq_params.cu_seqlens_q_padded,
            context_parallel_group=get_context_parallel_group(),
        )

    loss_data = data_dict

    # MseValueLossFn only accepts (logits, data, global_valid_seqs, global_valid_toks).
    # Unlike ClippedPGLossFn, it does not need vocab_parallel or context_parallel args.
    loss_fn_wrapped = partial(
        loss_fn,
        data=loss_data,
        global_valid_seqs=global_valid_seqs,
        global_valid_toks=global_valid_toks,
    )

    if cp_normalize:
        cp_size = parallel_state.get_context_parallel_world_size()
        orig_loss_fn_wrapped = loss_fn_wrapped

        def _div_by_cp_size(*args, **kwargs):
            loss, metrics = orig_loss_fn_wrapped(*args, **kwargs)
            return loss / cp_size, metrics

        loss_fn_wrapped = _div_by_cp_size

    # Counteract Megatron schedules.py's default (* cp_size / num_microbatches) loss
    # averaging so per-microbatch losses (each normalized by global_valid_toks) are
    # summed, not averaged. Mirrors the policy LossPostProcessor.
    cp_size = parallel_state.get_context_parallel_world_size()
    loss_fn_before_mcore_scaling = loss_fn_wrapped

    def _counteract_mcore_loss_averaging(*args, **kwargs):
        loss, metrics = loss_fn_before_mcore_scaling(*args, **kwargs)
        return loss * num_microbatches / cp_size, metrics

    loss_fn_wrapped = _counteract_mcore_loss_averaging

    return output_tensor, loss_fn_wrapped


def _install_value_head_load_skip(chunk: GPTModel) -> None:
    """Give the chunk a ``hide_loss_modules`` context manager that drops ``output_layer.*``.

    The freshly-initialized value head is never in a base checkpoint, so Megatron-Bridge
    enters this context during finetune loads (and HF->Megatron conversion) to skip it.
    Resume loads run with ``finetune=False``, so the trained head is restored normally.
    """
    chunk._skip_value_head_in_sharded_sd = False
    original_sharded_state_dict = chunk.sharded_state_dict

    def sharded_state_dict(
        prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ):
        sharded_sd = original_sharded_state_dict(
            prefix=prefix, sharded_offsets=sharded_offsets, metadata=metadata
        )
        if chunk._skip_value_head_in_sharded_sd:
            for key in list(sharded_sd.keys()):
                if key.startswith(f"{prefix}output_layer."):
                    del sharded_sd[key]
        return sharded_sd

    chunk.sharded_state_dict = sharded_state_dict

    @contextmanager
    def hide_loss_modules():
        previous = chunk._skip_value_head_in_sharded_sd
        chunk._skip_value_head_in_sharded_sd = True
        try:
            yield
        finally:
            chunk._skip_value_head_in_sharded_sd = previous

    chunk.hide_loss_modules = hide_loss_modules


def make_value_head_hook(hidden_size: int, sequence_parallel: bool):
    """Build the pre-wrap hook that installs the value head and lets it skip base loads."""
    base_hook = create_value_head_hook(
        hidden_size=hidden_size, sequence_parallel=sequence_parallel
    )

    def hook(model):
        model = base_hook(model)
        for chunk in model if isinstance(model, list) else [model]:
            if isinstance(getattr(chunk, "output_layer", None), LinearForLastLayer):
                _install_value_head_load_skip(chunk)
        return model

    return hook


def _unpack_value_sequences(
    values: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    unpacked_seqlen: int,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> torch.Tensor:
    """Unpack packed (and CP-sharded) per-token values back to ``[B, S]``.

    ``values`` is the value head output for a packed microbatch, shape ``[1, T // CP]``
    (SP/TP already gathered inside the head). For each sequence delimited by
    ``cu_seqlens_padded`` (indices into the full, non-CP-adjusted packed layout) the local
    CP shard is all-gathered to the full sequence, then shifted right by one so
    ``values[t] = V(state before token t)`` — matching the unpacked path.
    """
    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    batch_size = cu_seqlens_padded.shape[0] - 1
    values = values.squeeze(0)  # [T // CP]

    if cp_size > 1:
        full = torch.zeros(
            values.shape[0] * cp_size, dtype=values.dtype, device=values.device
        )
        for i in range(batch_size):
            start = cu_seqlens_padded[i].item()
            end = cu_seqlens_padded[i + 1].item()
            full[start:end] = allgather_cp_sharded_tensor(
                values[start // cp_size : end // cp_size], cp_group, seq_dim=0
            )
        values = full

    out = torch.zeros(
        (batch_size, unpacked_seqlen), dtype=values.dtype, device=values.device
    )
    for i in range(batch_size):
        start = cu_seqlens_padded[i].item()
        end = cu_seqlens_padded[i + 1].item()
        seq_values = values[start:end]
        seq_values = torch.cat(
            [torch.zeros_like(seq_values[:1]), seq_values[:-1]], dim=0
        )
        seq_len = min(seq_values.shape[0], unpacked_seqlen)
        out[i, :seq_len] = seq_values[:seq_len]
    return out


def _value_packed_loss_prepare_fn(
    logits: torch.Tensor,
    data: BatchedDataDict[Any],
    loss_fn: Optional[LossFunction] = None,
    vocab_parallel_rank: Optional[int] = None,
    vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
) -> tuple[dict[str, torch.Tensor], BatchedDataDict[Any]]:
    """Prepare one packed sequence's value-head output for ``MseValueLossFn``.

    ``SequencePackingLossWrapper`` hands us a single sequence's CP-sharded value
    slice ``[1, padded_len // CP]``. All-gather across CP to ``[1, padded_len]``,
    shift right by one (``values[t] = V(state before token t)``), and truncate to
    the sequence's unpadded length so it lines up with ``data["returns"]``.
    """
    values = logits
    cp_size = (
        1
        if context_parallel_group is None
        else torch.distributed.get_world_size(context_parallel_group)
    )
    if cp_size > 1:
        values = allgather_cp_sharded_tensor(values, context_parallel_group, seq_dim=1)
    values = torch.cat([torch.zeros_like(values[:, :1]), values[:, :-1]], dim=1)
    values = values[:, : data["returns"].shape[1]]
    return {"logits": values}, data


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
# This is useful when using worker extension classes.
class MegatronValueWorkerImpl(AbstractPolicyWorker):
    """Megatron-Core based value function worker for PPO.

    This worker wraps a Megatron-Core GPT model backbone with a value head
    (Linear: hidden_size -> 1) to predict per-token state values.

    Supports:
    - Tensor parallelism (TP)
    - Pipeline parallelism (PP)
    - Context parallelism (CP)
    - Sequence packing
    - Activation checkpointing
    - FP8 quantization
    - MoE models
    """

    def __repr__(self):
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    @staticmethod
    def configure_worker(
        num_gpus: int | float,
        bundle_indices: Optional[tuple[int, list[int]]] = None,
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
        """Worker-controlled Ray actor configuration.

        Mirrors `MegatronPolicyWorker` to ensure NVLS communication functions correctly.

        Args:
            num_gpus: Original GPU allocation for this worker based on the placement group
            bundle_indices: Tuple of (node_idx, local_bundle_indices) for this server

        Returns:
            tuple with complete worker configuration:
              - 'resources': Resource allocation (e.g., num_gpus)
              - 'env_vars': Environment variables for this worker
              - 'init_kwargs': Parameters to pass to __init__ of the worker
              - 'runtime_env': Additional runtime_env options (e.g., nsight config)
        """
        del bundle_indices  # one GPU per worker; no per-bundle seeding needed
        resources: dict[str, Any] = {"num_gpus": num_gpus}
        env_vars: dict[str, str] = {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
        init_kwargs: dict[str, Any] = {}
        return resources, env_vars, init_kwargs, {}

    def __init__(
        self,
        config: ValueConfig,
        tokenizer: TokenizerType,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        *,
        worker_sharding_annotations: NamedSharding,
        **kwargs: Any,
    ):
        """Initialize the MegatronValueWorker.

        Args:
            config: Value model configuration.
            tokenizer: HuggingFace tokenizer.
            weights_path: Path to load finetuned weights from (optional).
            optimizer_path: Path to load optimizer state from (optional).
            init_optimizer: Whether to initialize the optimizer.
            worker_sharding_annotations: Sharding topology for distributed training.
        """
        # Must be the first CUDA-touching call in this process.
        # With `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` (set by `configure_worker()`),
        gpu_ids = ray.get_gpu_ids()
        local_rank = int(gpu_ids[0])
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(local_rank)

        apply_transformer_engine_patch()

        from nemo_rl.distributed.numa_utils import bind_to_gpu_numa

        # Pin this value worker to its GPU's NUMA-local CPUs/memory before model
        # load (mirrors the Megatron/DTensor policy workers and vLLM workers).
        bind_to_gpu_numa()

        self.cfg = config
        self.rank = get_rank_safe()

        # Step 1: Setup distributed
        setup_distributed()

        # Step 2: Validate and setup model paths
        # Value config uses the same model_name field as policy config.
        # Adapt it once and cache it — the result is deterministic for `config`,
        # and downstream `train` / `get_values` calls reuse this attribute.
        self._policy_like_cfg = make_policy_like_config(config)

        hf_model_name, pretrained_path, pt_checkpoint_exists = validate_model_paths(
            self._policy_like_cfg
        )
        handle_model_import(
            self._policy_like_cfg,
            hf_model_name,
            pretrained_path,
            pt_checkpoint_exists,
        )

        # Store tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Step 3: Setup model configuration
        runtime_config = validate_and_set_config(
            self._policy_like_cfg,
            self.rank,
            hf_model_name,
            pretrained_path,
            weights_path,
            optimizer_path,
        )

        self.megatron_cfg = runtime_config.megatron_cfg
        self.dtype = runtime_config.dtype
        self.optimizer_cpu_offload = runtime_config.optimizer_cpu_offload
        self.offload_optimizer_for_logprob = (
            runtime_config.offload_optimizer_for_logprob
        )
        self.final_padded_vocab_size = runtime_config.final_padded_vocab_size

        self.defer_fp32_logits = config["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (runtime_config.model_cfg.fp16 or runtime_config.model_cfg.bf16)

        # Validate configuration
        self.megatron_cfg.validate()

        assert self.megatron_cfg.model.virtual_pipeline_model_parallel_size in (
            None,
            1,
        ), (
            "Virtual pipeline parallelism (VPP) is not supported for the "
            "Megatron PPO value model."
        )

        # Step 4: Setup Megatron model and components
        # The value head is an independent hidden->1 head, not tied to the input
        # embedding. Untie before building so PP>1 does not set up an embedding/
        # output-weight grad all-reduce that mismatches the [1, hidden] head and hangs.
        self.megatron_cfg.model.share_embeddings_and_output_weights = False
        # Replace output_layer with a hidden->1 value head before DDP wrapping, so grad
        # sync, the optimizer, dist-checkpoint save/load, PP, and SP are all inherited.
        value_head_hook = make_value_head_hook(
            hidden_size=self.megatron_cfg.model.hidden_size,
            sequence_parallel=self.megatron_cfg.model.sequence_parallel,
        )

        model_and_optimizer_state = setup_model_and_optimizer(
            self._policy_like_cfg,
            self.megatron_cfg,
            init_optimizer,
            additional_pre_wrap_hooks=[value_head_hook],
        )

        self.mcore_state = model_and_optimizer_state.state
        self.model = model_and_optimizer_state.model
        self.optimizer = model_and_optimizer_state.optimizer
        self.scheduler = model_and_optimizer_state.scheduler
        self.checkpointing_context = model_and_optimizer_state.checkpointing_context
        param_sync_func = model_and_optimizer_state.param_sync_func

        if param_sync_func is not None:
            self.megatron_cfg.param_sync_func = param_sync_func

        # The value head (output_layer) is now a model parameter, so it is
        # trained by the main optimizer and saved/loaded via the normal
        # dist-checkpoint — no separate value-head optimizer or sidecar needed.

        # Step 6: Finalize setup
        (
            self.megatron_tokenizer,
            self.megatron_bridge,
            self.should_disable_forward_pre_hook,
            self.dp_size,
        ) = finalize_megatron_setup(
            self._policy_like_cfg,
            self.megatron_cfg,
            hf_model_name,
            worker_sharding_annotations,
            self.model,
            self.optimizer,
        )

        print(f"MegatronValueWorker initialized on rank {self.rank}")

    def enable_forward_pre_hook(self):
        if isinstance(self.model, DistributedDataParallel):
            self.model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        if isinstance(self.model, DistributedDataParallel):
            self.model.disable_forward_pre_hook(param_sync=param_sync)

    @wrap_with_nvtx_name("megatron_value_worker/train")
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the value function on a batch of data with a given loss function.

        Args:
            data: BatchedDataDict containing training data with keys:
                - input_ids, input_lengths, token_mask, sample_mask, returns
            loss_fn: Value loss function (e.g., MseValueLossFn).
            eval_mode: If True, run forward only without parameter updates.
            gbs: Global batch size override.
            mbs: Micro batch size override.

        Returns:
            Dictionary with training metrics (global_loss, grad_norm, etc.)
        """
        self.model.zero_grad_buffer()
        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None

        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_data_parallel_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            self.model.train()

        with ctx:
            forward_step = partial(
                forward_step_value,
                loss_fn=loss_fn,
                policy_cfg=None,
            )
            all_mb_metrics = []
            losses = []
            total_num_microbatches = 0

            for gb_idx in range(num_global_batches):
                gb_result = process_global_batch(
                    data,
                    loss_fn=loss_fn,
                    dp_group=parallel_state.get_data_parallel_group(),
                    batch_idx=gb_idx,
                    batch_size=local_gbs,
                )
                batch = gb_result["batch"]
                global_valid_seqs = gb_result["global_valid_seqs"]
                global_valid_toks = gb_result["global_valid_toks"]

                (
                    data_iterator,
                    num_microbatches,
                    micro_batch_size_actual,
                    seq_length,
                    padded_seq_length,
                ) = get_microbatch_iterator(
                    batch,
                    self._policy_like_cfg,
                    mbs,
                    straggler_timer=self.mcore_state.straggler_timer,
                )
                total_num_microbatches += int(num_microbatches)

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Zero gradients
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()

                    # Forward-backward pass
                    forward_backward_func = get_forward_backward_func()
                    losses_reduced = forward_backward_func(
                        forward_step_func=partial(
                            forward_step,
                            self.mcore_state,
                            global_valid_seqs,
                            global_valid_toks,
                            pack_sequences=self.cfg.get("sequence_packing", {}).get(
                                "enabled", False
                            ),
                            defer_fp32_logits=self.defer_fp32_logits,
                            num_microbatches=num_microbatches,
                        ),
                        data_iterator=data_iterator,
                        model=self.model,
                        num_microbatches=num_microbatches,
                        seq_length=padded_seq_length,
                        micro_batch_size=micro_batch_size_actual,
                        decoder_seq_length=padded_seq_length,
                        forward_only=eval_mode,
                    )

                # Empty unused memory
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters
                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )

                    pg_collection = get_pg_collection(self.model)
                    update_successful = logical_and_across_model_parallel_group(
                        update_successful, mp_group=pg_collection.mp
                    )
                    grad_norm = reduce_max_stat_across_model_parallel_group(
                        grad_norm, mp_group=pg_collection.mp
                    )
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        True,
                        0.0,
                        0.0,
                    )

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
                    torch.cuda.empty_cache()

                if is_pipeline_last_stage(ignore_virtual=True):
                    gb_loss_metrics = []
                    mb_losses = []
                    for x in losses_reduced:
                        loss_metrics = {}
                        for k in x.keys():
                            if "_min" in k or "_max" in k:
                                loss_metrics[k] = x[k]
                            else:
                                loss_metrics[k] = x[k] / num_global_batches
                        gb_loss_metrics.append(loss_metrics)
                        curr_lr = self.scheduler.get_lr(self.optimizer.param_groups[0])
                        curr_wd = self.scheduler.get_wd()
                        loss_metrics["lr"] = curr_lr
                        loss_metrics["wd"] = curr_wd
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()
                        mb_losses.append(loss_metrics["loss"])

                    torch.distributed.broadcast_object_list(
                        [gb_loss_metrics],
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                else:
                    loss_metrics = [None]
                    torch.distributed.broadcast_object_list(
                        loss_metrics,
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                    gb_loss_metrics = loss_metrics[0]
                    mb_losses = [x["loss"] for x in gb_loss_metrics]

                all_mb_metrics.extend(gb_loss_metrics)
                losses.append(torch.tensor(mb_losses).sum().item())

        if not eval_mode:
            # Step LR scheduler once per train() call, not per global batch.
            # Megatron's OptimizerParamScheduler.step takes an `increment` in
            # samples: NeMo init scales lr_warmup_steps by gbs internally, so
            # passing increment=gbs cancels that scaling and one tick == one
            # train() call regardless of batch size.
            self.scheduler.step(increment=gbs)

        # Aggregate metrics
        mb_metrics = defaultdict(list)
        for m in all_mb_metrics:
            for k, v in m.items():
                mb_metrics[k].append(v)

        with torch.no_grad():
            global_loss = torch.tensor(losses, device="cuda")
            torch.distributed.all_reduce(
                global_loss,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_data_parallel_group(),
            )

        metrics = {
            "global_loss": global_loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "all_mb_metrics": dict(mb_metrics),
            "grad_norm": torch.tensor([grad_norm])
            if grad_norm is not None
            else torch.tensor([0.0]),
        }

        # Collect MoE aux metrics if applicable
        # Use getattr-by-string so "config" stays out of co_names; torch 2.11
        # cloudpickle otherwise matches torch.distributed.config (non-pickleable).
        model_config = getattr(self.model, "config", None)
        num_moe_experts = getattr(model_config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"].get(
                    "moe_per_layer_logging", False
                ),
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics

        return metrics

    @wrap_with_nvtx_name("megatron_value_worker/get_values")
    def get_values(
        self, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[ValueOutputSpec]:
        """Get per-token value predictions for a batch of data.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths.
            micro_batch_size: Override for inference micro batch size.

        Returns:
            BatchedDataDict with "values" key of shape [batch_size, seq_length].
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()

        value_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg.get("logprob_batch_size", self.cfg["train_micro_batch_size"])
        )

        self.model.eval()

        pp_grp = get_pipeline_model_parallel_group()

        (
            mb_iterator,
            num_microbatches,
            micro_batch_size_actual,
            seq_length,
            padded_seq_length,
        ) = get_microbatch_iterator(
            data,
            self._policy_like_cfg,
            value_batch_size,
            straggler_timer=self.mcore_state.straggler_timer,
        )

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            processed_mb = next(data_iterator)
            data_dict = processed_mb.data_dict
            input_ids_cp_sharded = processed_mb.input_ids_cp_sharded
            attention_mask = processed_mb.attention_mask
            position_ids = processed_mb.position_ids
            packed_seq_params = processed_mb.packed_seq_params

            additional_kwargs = {}
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **additional_kwargs,
            )

            def collection_fn(output_tensor):
                # The head-owning stage returns per-token values (SP/TP already
                # gathered in the head); other stages return hidden states whose
                # collection output is discarded (values are broadcast from the
                # last stage below).
                if not is_pipeline_last_stage(ignore_virtual=True):
                    values = torch.zeros(1, device=output_tensor.device)
                    return torch.tensor(0.0, device=values.device), {"values": values}

                if processed_mb.cu_seqlens_padded is not None:
                    # Packed (and possibly CP-sharded) [1, T // CP, 1] -> [B, S],
                    # per-sequence shift + CP all-gather.
                    cp_size = get_context_parallel_world_size()
                    values = _unpack_value_sequences(
                        output_tensor.squeeze(-1),
                        processed_mb.cu_seqlens_padded,
                        seq_length,
                        cp_group=get_context_parallel_group() if cp_size > 1 else None,
                    )
                else:
                    # [B, S]: shift right by 1 so values[t] = V(state before token t).
                    values = output_tensor.squeeze(-1)
                    values = torch.cat(
                        [torch.zeros_like(values[:, :1]), values[:, :-1]], dim=1
                    )
                return torch.tensor(0.0, device=values.device), {"values": values}

            return output_tensor, collection_fn

        forward_backward_func = get_forward_backward_func()
        list_of_values = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=num_microbatches,
            seq_length=padded_seq_length,
            micro_batch_size=micro_batch_size_actual,
            decoder_seq_length=padded_seq_length,
            forward_only=True,
        )

        if is_pipeline_last_stage(ignore_virtual=True):
            all_values_padded = []
            all_values = [v["values"] for v in list_of_values]
            for val in all_values:
                padding_needed = seq_length - val.shape[1]
                if padding_needed > 0:
                    # For [B, S] tensors, pad along seq dim (dim 1)
                    val = torch.nn.functional.pad(
                        val, (0, padding_needed), mode="constant", value=0.0
                    )
                all_values_padded.append(val)

            values_tensor = torch.cat(all_values_padded, dim=0)
            broadcast_tensor(values_tensor, torch.distributed.get_rank(), pp_grp)
        else:
            values_tensor = broadcast_tensor(
                None, get_pipeline_model_parallel_last_rank(), pp_grp
            )

        no_grad.__exit__(None, None, None)

        return BatchedDataDict[ValueOutputSpec](values=values_tensor.cpu())

    def prepare_for_training(self) -> None:
        """Move model and optimizer to CUDA for training."""
        self.model = self.move_model(
            self.model, "cuda", move_grads=True, move_params=True
        )
        self.model.train()

        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cuda")

        if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
            torch.cuda.empty_cache()

    def prepare_for_inference(self):
        """Prepare model for value inference."""
        self.model = self.move_model(self.model, "cuda", move_grads=False)
        self.model.eval()

        # Offload gradients
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )

        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def move_model(
        self,
        model: torch.nn.Module,
        device: str,
        move_params: bool = True,
        move_grads: bool = True,
    ) -> torch.nn.Module:
        """Move model parameters and gradient buffers to the specified device."""
        if isinstance(model, DistributedDataParallel):
            for buffers in [model.buffers, model.expert_parallel_buffers]:
                for buffer_idx in range(len(buffers)):
                    if device == "cpu":
                        buffers[buffer_idx].offload_to_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    elif device == "cuda":
                        buffers[buffer_idx].reload_from_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only 'cpu' and 'cuda' are supported."
                        )
        elif isinstance(model, custom_FSDP):
            if device == "cpu":
                model.param_and_grad_buffer.offload_to_cpu(move_params, move_grads)
            elif device == "cuda":
                model.param_and_grad_buffer.reload_from_cpu(
                    move_params=move_params, move_grads=move_grads
                )
            else:
                raise ValueError(
                    f"Invalid device: {device}. Only 'cpu' and 'cuda' are supported."
                )
        else:
            if move_params:
                new_state_dict = {}
                for name, item in model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        item = item.detach().to(
                            device=device, non_blocking=True, copy=True
                        )
                    new_state_dict[name] = item
                model.load_state_dict(new_state_dict)
        return model

    def move_optimizer(self, device: str):
        """Move optimizer state to the specified device."""
        if isinstance(self.optimizer, ChainedOptimizer):
            optimizer_state = self.optimizer.state
        else:
            optimizer_state = self.optimizer._get_state()
        for _, state in optimizer_state.items():
            for k, v in state.items():
                if torch.is_tensor(v):
                    if device == "cpu" and v.is_cuda:
                        state[k] = v.to("cpu")
                    elif device == "cuda" and not v.is_cuda:
                        state[k] = v.to("cuda")

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        **kwargs,
    ):
        """Save a checkpoint of the value model.

        The value head is the model's ``output_layer`` and is saved as part of
        the normal Megatron dist-checkpoint — no separate value-head sidecar.
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed process group is not initialized. Cannot save checkpoint."
            )

        original_save_path = self.mcore_state.cfg.checkpoint.save

        try:
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=False,
            )
            self.mcore_state.cfg.checkpoint.save = weights_path

            optimizer_to_save = None
            scheduler_to_save = None
            if optimizer_path is not None:
                if self.optimizer is not None:
                    optimizer_to_save = self.optimizer
                if self.scheduler is not None:
                    scheduler_to_save = self.scheduler

            if self.should_disable_forward_pre_hook:
                self.disable_forward_pre_hook()

            # Save Megatron backbone checkpoint
            save_checkpoint(
                state=self.mcore_state,
                model=[self.model],
                optimizer=optimizer_to_save,
                opt_param_scheduler=scheduler_to_save,
                num_floating_point_operations_so_far=self.mcore_state.train_state.floating_point_operations_so_far,
                checkpointing_context=self.checkpointing_context,
            )

            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=True,
                terminate=True,
            )

            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

            torch.distributed.barrier()
            print(f"Saved value model checkpoint to {weights_path}")

        except Exception as e:
            print(f"Failed to save value checkpoint to {weights_path}: {e}")
            raise
        finally:
            self.mcore_state.cfg.checkpoint.save = original_save_path

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        """Load a checkpoint for the value model."""
        raise NotImplementedError(
            "Loading checkpoints outside of init is not yet implemented for MegatronValueWorker."
        )

    def finish_inference(self) -> None:
        """Offload model params to CPU after inference."""
        self.model = self.move_model(
            self.model, "cpu", move_params=True, move_grads=False
        )

        gc.collect()
        torch.cuda.empty_cache()

    def finish_training(self) -> None:
        """Offload model, gradients, and optimizer to CPU after training."""
        self.model = self.move_model(
            self.model, "cpu", move_params=True, move_grads=True
        )
        self.model.eval()

        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class MegatronValueWorker(MegatronValueWorkerImpl):
    pass
