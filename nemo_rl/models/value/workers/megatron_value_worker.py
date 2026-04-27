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
import warnings
from contextlib import AbstractContextManager, nullcontext
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
    get_tensor_model_parallel_group,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import get_rerun_state_machine
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import allgather_cp_sharded_tensor
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.megatron.common import get_moe_metrics
from nemo_rl.models.megatron.data import get_microbatch_iterator, process_global_batch
from nemo_rl.models.megatron.pipeline_parallel import (
    broadcast_loss_metrics_from_last_stage,
    broadcast_tensors_from_last_stage,
)
from nemo_rl.models.megatron.setup import (
    finalize_megatron_setup,
    handle_model_import,
    setup_distributed,
    setup_model_and_optimizer,
    validate_and_set_config,
    validate_model_paths,
)
from nemo_rl.models.megatron.train import aggregate_training_statistics
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.models.value.config import ValueConfig
from nemo_rl.models.value.interfaces import ValueOutputSpec
from nemo_rl.utils.nsys import wrap_with_nvtx_name

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


class ValueHead(torch.nn.Module):
    """Simple linear value head that maps hidden states to scalar values.

    Works correctly with tensor parallelism by operating on the full hidden
    dimension (which is not split across TP ranks at the output_layer input).
    With sequence parallelism, each TP rank processes its shard of the sequence
    independently; results are gathered later.
    """

    def __init__(self, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.dtype = dtype
        self.linear = torch.nn.Linear(hidden_size, 1, bias=True)
        self.linear.to(dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Map hidden states to scalar values.

        Args:
            hidden_states: [batch, seq, hidden_size] (may be seq-parallel sharded)

        Returns:
            values: [batch, seq, 1]
        """
        with torch.autocast(device_type=hidden_states.device.type, dtype=torch.float32):
            return self.linear(hidden_states.float())


def _unwrap_model(model):
    """Unwrap a model from DDP/Float16Module wrappers to get the base GPTModel."""
    m = model
    while hasattr(m, "module"):
        m = m.module
    return m


class _ValueOutputLayerBypass(torch.nn.Module):
    """Replaces the output_layer during value model forward to skip the expensive
    logits computation (hidden_size -> vocab_size).

    Instead of computing [S, B, vocab_size] logits, captures the hidden states
    and returns a minimal [S, B, 1] tensor, saving both memory and FLOPS.
    """

    def __init__(self, captured_hidden: dict):
        super().__init__()
        self.captured_hidden = captured_hidden

    def forward(self, hidden_states, *args, **kwargs):
        self.captured_hidden["hidden_states"] = hidden_states
        # Return (tensor, bias) tuple matching ColumnParallelLinear's signature,
        # since GPTModel._postprocess does `logits, _ = self.output_layer(...)`.
        # Uses a slice view to avoid allocating new memory.
        dummy = hidden_states[..., :1]
        return dummy, None


def forward_step_value(
    state,
    global_valid_seqs,
    global_valid_toks,
    data_iterator,
    model,
    *,
    value_head,
    loss_fn,
    pack_sequences=False,
    defer_fp32_logits=None,
    cp_normalize=True,
    policy_cfg=None,
):
    """Forward step for value model that captures hidden states and computes values.

    This is similar to forward_step_arbitrary_loss but intercepts hidden states
    before the language model head and applies a value head instead.
    """
    from nemo_rl.algorithms.loss import SequencePackingLossWrapper

    straggler_timer = state.straggler_timer

    # Get the pre-processed microbatch from the iterator
    processed_mb = next(data_iterator)

    data_dict = processed_mb.data_dict
    input_ids = processed_mb.input_ids
    input_ids_cp_sharded = processed_mb.input_ids_cp_sharded
    attention_mask = processed_mb.attention_mask
    position_ids = processed_mb.position_ids
    packed_seq_params = processed_mb.packed_seq_params
    cu_seqlens_padded = processed_mb.cu_seqlens_padded

    additional_kwargs = {}
    if packed_seq_params is not None:
        additional_kwargs["packed_seq_params"] = packed_seq_params
    if defer_fp32_logits:
        additional_kwargs["fp32_output"] = False

    # Bypass the output_layer to avoid computing expensive logits [S, B, vocab_size].
    # Instead, capture hidden_states and route them through the value head.
    captured_hidden = {}
    base_model = _unwrap_model(model)
    original_output_layer = None

    if hasattr(base_model, "output_layer"):
        original_output_layer = base_model.output_layer
        base_model.output_layer = _ValueOutputLayerBypass(captured_hidden)

    try:
        with straggler_timer:
            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **additional_kwargs,
            )
    finally:
        # Always restore the original output_layer
        if original_output_layer is not None:
            base_model.output_layer = original_output_layer

    # Compute values from captured hidden states
    if "hidden_states" in captured_hidden:
        hidden_states = captured_hidden["hidden_states"]
        # Megatron-Core always uses [S, B, H] layout internally; transpose to [B, S, H]
        hidden_states = hidden_states.transpose(0, 1).contiguous()
        values = value_head(hidden_states).squeeze(-1)  # [batch, seq]
        # Shift right by 1 to align with inference: values[t] = V(state before token t).
        # This must match the shift in get_values_impl so that the training targets
        # (returns, old_values) computed from inference values are aligned.
        values = torch.cat([torch.zeros_like(values[:, :1]), values[:, :-1]], dim=1)
        # Replace output_tensor with values for loss computation.
        # Include the original output_layer weight with a zero contribution so
        # that it receives a gradient during backward. Megatron DDP with
        # overlap_grad_reduce=True requires every registered parameter to
        # produce a gradient for bucket sync; without this the output_layer
        # weight has no grad and finish_grad_sync() raises an AssertionError.
        if original_output_layer is not None and False:
            values = values + 0.0 * original_output_layer.weight.view(-1)[0]
        output_tensor = values
    else:
        # On non-last PP stages, output_layer doesn't exist.
        # output_tensor is the intermediate hidden states, pass through.
        pass

    # Handle packed sequences
    if pack_sequences and packed_seq_params is not None:
        loss_fn = SequencePackingLossWrapper(
            loss_fn=loss_fn,
            cu_seqlens_q=packed_seq_params.cu_seqlens_q,
            cu_seqlens_q_padded=packed_seq_params.cu_seqlens_q_padded,
        )

    loss_data = data_dict

    # MseValueLossFn only accepts (values, data, global_valid_seqs, global_valid_toks).
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

    return output_tensor, loss_fn_wrapped


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class MegatronValueWorker(AbstractPolicyWorker):
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
        apply_transformer_engine_patch()

        self.cfg = config
        self.rank = get_rank_safe()

        # Step 1: Setup distributed
        setup_distributed()

        # Step 2: Validate and setup model paths
        # Value config uses the same model_name field as policy config
        # We need to adapt the config to look like a PolicyConfig for reuse
        policy_like_cfg = self._make_policy_like_config(config)

        hf_model_name, pretrained_path, pt_checkpoint_exists = validate_model_paths(
            policy_like_cfg
        )
        handle_model_import(
            policy_like_cfg, hf_model_name, pretrained_path, pt_checkpoint_exists
        )

        # Store tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Step 3: Setup model configuration
        runtime_config = validate_and_set_config(
            policy_like_cfg,
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
        self.is_generation_colocated = runtime_config.is_generation_colocated
        self.sampling_params = runtime_config.sampling_params
        self.final_padded_vocab_size = runtime_config.final_padded_vocab_size

        self.defer_fp32_logits = config["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (runtime_config.model_cfg.fp16 or runtime_config.model_cfg.bf16)

        # Validate configuration
        self.megatron_cfg.validate()

        # Step 4: Setup Megatron model and components
        # Freeze the output_layer (LM head) before DDP wrapping so it is not
        # registered in any DDP gradient bucket.  The value model never uses the
        # LM head — it bypasses it and routes hidden states through a value head
        # instead — so its weights should not participate in gradient sync.
        def _freeze_output_layer(megatron_model):
            if not isinstance(megatron_model, list):
                megatron_model = [megatron_model]
            for model_module in megatron_model:
                if hasattr(model_module, "module"):
                    model_module = model_module.module
                if hasattr(model_module, "output_layer"):
                    for param in model_module.output_layer.parameters():
                        param.requires_grad_(False)

        model_and_optimizer_state = setup_model_and_optimizer(
            policy_like_cfg,
            self.megatron_cfg,
            init_optimizer,
            additional_pre_wrap_hooks=[_freeze_output_layer],
        )

        self.mcore_state = model_and_optimizer_state.state
        self.model = model_and_optimizer_state.model
        self.optimizer = model_and_optimizer_state.optimizer
        self.scheduler = model_and_optimizer_state.scheduler
        self.checkpointing_context = model_and_optimizer_state.checkpointing_context
        param_sync_func = model_and_optimizer_state.param_sync_func

        if param_sync_func is not None:
            self.megatron_cfg.param_sync_func = param_sync_func

        # Step 5: Create value head
        hidden_size = self.megatron_cfg.model.hidden_size
        self.value_head = ValueHead(hidden_size, self.dtype).cuda()

        # Load value head weights: prefer training checkpoint, fall back to HF model
        value_head_loaded = False
        if weights_path is not None:
            value_head_path = os.path.join(weights_path, "value_head.pt")
            if os.path.exists(value_head_path):
                value_head_state = torch.load(
                    value_head_path, map_location="cuda", weights_only=True
                )
                self.value_head.load_state_dict(value_head_state)
                print(f"Loaded value head weights from {value_head_path}")
                value_head_loaded = True

        if not value_head_loaded and config.get("load_value_head_from_model", False):
            from nemo_rl.models.megatron.community_import import (
                extract_value_head_from_hf_checkpoint,
            )

            score_weights = extract_value_head_from_hf_checkpoint(config["model_name"])
            if "score.weight" in score_weights:
                self.value_head.linear.weight.data.copy_(score_weights["score.weight"])
                print(f"Loaded value head score.weight from {config['model_name']}")
            if "score.bias" in score_weights:
                # Add bias to the linear layer if the checkpoint has one
                if self.value_head.linear.bias is None:
                    self.value_head.linear.bias = torch.nn.Parameter(
                        torch.zeros(
                            self.value_head.linear.out_features,
                            dtype=self.value_head.dtype,
                            device=self.value_head.linear.weight.device,
                        )
                    )
                self.value_head.linear.bias.data.copy_(score_weights["score.bias"])
                print(f"Loaded value head score.bias from {config['model_name']}")

        # Add value head parameters to the optimizer
        if init_optimizer and self.optimizer is not None:
            # For Megatron optimizers, we need to handle this carefully.
            # The value head is a small module, so we add it as a separate param group.
            self._add_value_head_to_optimizer()

        # Step 6: Finalize setup
        (
            self.megatron_tokenizer,
            self.megatron_bridge,
            self.should_disable_forward_pre_hook,
            self.dp_size,
        ) = finalize_megatron_setup(
            policy_like_cfg,
            self.megatron_cfg,
            hf_model_name,
            worker_sharding_annotations,
            self.model,
            self.optimizer,
        )

        print(f"MegatronValueWorker initialized on rank {self.rank}")

    def _make_policy_like_config(self, config: ValueConfig) -> dict:
        """Adapt ValueConfig to look like a PolicyConfig for reusing setup functions.

        The Megatron setup functions expect PolicyConfig fields. We create a
        compatible dict from the ValueConfig.
        """
        megatron_cfg = dict(config["megatron_cfg"])

        # Ensure required fields have defaults
        megatron_cfg.setdefault("empty_unused_memory_level", 1)
        megatron_cfg.setdefault("freeze_moe_router", False)
        megatron_cfg.setdefault("moe_per_layer_logging", False)
        megatron_cfg.setdefault("moe_enable_deepep", False)
        megatron_cfg.setdefault("moe_token_dispatcher_type", "allgather")
        megatron_cfg.setdefault("moe_shared_expert_overlap", False)
        megatron_cfg.setdefault("moe_permute_fusion", False)
        megatron_cfg.setdefault("moe_router_load_balancing_type", "none")
        megatron_cfg.setdefault("moe_router_bias_update_rate", 0.0)
        megatron_cfg.setdefault("moe_router_dtype", None)
        megatron_cfg.setdefault("num_layers_in_first_pipeline_stage", None)
        megatron_cfg.setdefault("num_layers_in_last_pipeline_stage", None)
        megatron_cfg.setdefault("apply_rope_fusion", True)
        megatron_cfg.setdefault("bias_activation_fusion", True)
        megatron_cfg.setdefault("defer_fp32_logits", False)
        megatron_cfg.setdefault("force_overwrite_initial_ckpt", False)

        policy_cfg = {
            "model_name": config["model_name"],
            "tokenizer": config["tokenizer"],
            "train_global_batch_size": config["train_global_batch_size"],
            "train_micro_batch_size": config["train_micro_batch_size"],
            "logprob_batch_size": config.get(
                "logprob_batch_size", config["train_micro_batch_size"]
            ),
            "precision": config["precision"],
            "megatron_cfg": megatron_cfg,
            "dynamic_batching": config["dynamic_batching"],
            "sequence_packing": config.get("sequence_packing", {"enabled": False}),
            "make_sequence_length_divisible_by": config[
                "make_sequence_length_divisible_by"
            ],
            "max_total_sequence_length": config["max_total_sequence_length"],
            "max_grad_norm": config.get("max_grad_norm", 1.0),
            "hf_config_overrides": config.get("hf_config_overrides", {}),
            "offload_optimizer_for_logprob": False,
            # Value models don't use generation or reference models
            "generation": None,
        }

        return policy_cfg

    def _add_value_head_to_optimizer(self):
        """Add value head parameters to the Megatron optimizer.

        Since the Megatron optimizer manages parameters through DDP buffers,
        we create a separate PyTorch optimizer for the value head and step
        both during training.
        """
        lr = self.megatron_cfg.optimizer.lr
        weight_decay = self.megatron_cfg.optimizer.weight_decay

        self.value_head_optimizer = torch.optim.AdamW(
            self.value_head.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(
                self.megatron_cfg.optimizer.adam_beta1,
                self.megatron_cfg.optimizer.adam_beta2,
            ),
            eps=self.megatron_cfg.optimizer.adam_eps,
        )

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
        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None

        # Reset any cached attention states
        for module in self.model.modules():
            if hasattr(module, "reset_inference_cache"):
                module.reset_inference_cache()
            if hasattr(module, "_inference_key_value_memory"):
                module._inference_key_value_memory = None

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
            self.value_head.eval()
        else:
            ctx = nullcontext()
            self.model.train()
            self.value_head.train()

        with ctx:
            forward_step = partial(
                forward_step_value,
                loss_fn=loss_fn,
                value_head=self.value_head,
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
                    micro_batch_size,
                    seq_length,
                    padded_seq_length,
                ) = get_microbatch_iterator(
                    batch,
                    self._make_policy_like_config(self.cfg),
                    mbs,
                    straggler_timer=self.mcore_state.straggler_timer,
                )
                total_num_microbatches += int(num_microbatches)

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Zero gradients
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()
                    if hasattr(self, "value_head_optimizer"):
                        self.value_head_optimizer.zero_grad()

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
                        ),
                        data_iterator=data_iterator,
                        model=self.model,
                        num_microbatches=num_microbatches,
                        seq_length=padded_seq_length,
                        micro_batch_size=mbs,
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

                    # Step value head optimizer separately.
                    # The value head is NOT wrapped in DDP, so we must manually
                    # allreduce its gradients to keep weights in sync.
                    if hasattr(self, "value_head_optimizer"):
                        # When sequence parallelism is on, each TP rank processes
                        # a different shard of the sequence, so the value head
                        # receives different gradients on each TP rank. We must
                        # allreduce across TP first to get the correct full-sequence
                        # gradient before syncing across DP.
                        tp_size = parallel_state.get_tensor_model_parallel_world_size()
                        if tp_size > 1 and self.megatron_cfg.model.sequence_parallel:
                            tp_group = parallel_state.get_tensor_model_parallel_group()
                            for param in self.value_head.parameters():
                                if param.grad is not None:
                                    torch.distributed.all_reduce(
                                        param.grad,
                                        op=torch.distributed.ReduceOp.AVG,
                                        group=tp_group,
                                    )

                        dp_group = parallel_state.get_data_parallel_group()
                        for param in self.value_head.parameters():
                            if param.grad is not None:
                                torch.distributed.all_reduce(
                                    param.grad,
                                    op=torch.distributed.ReduceOp.AVG,
                                    group=dp_group,
                                )
                        # Clip value head gradients
                        max_grad_norm = self.cfg.get("max_grad_norm", 1.0)
                        if max_grad_norm is not None and max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.value_head.parameters(), max_grad_norm
                            )
                        self.value_head_optimizer.step()

                    pg_collection = get_pg_collection(self.model)
                    update_successful = logical_and_across_model_parallel_group(
                        update_successful, mp_group=pg_collection.mp
                    )
                    grad_norm: float = reduce_max_stat_across_model_parallel_group(
                        grad_norm, mp_group=pg_collection.mp
                    )
                    num_zeros_in_grad: float = (
                        reduce_max_stat_across_model_parallel_group(
                            num_zeros_in_grad, mp_group=pg_collection.mp
                        )
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
                else:
                    gb_loss_metrics = None

                # Broadcast loss metrics from last stage to all stages
                gb_loss_metrics = broadcast_loss_metrics_from_last_stage(
                    gb_loss_metrics
                )
                if not parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    mb_losses = [x["loss"] for x in gb_loss_metrics]

                all_mb_metrics.extend(gb_loss_metrics)
                losses.append(torch.tensor(mb_losses).sum().item())

                if not eval_mode:
                    # step LR scheduler after every optimizer step
                    self.scheduler.step(increment=gbs)

        # Aggregate metrics across all microbatches
        mb_metrics, global_loss = aggregate_training_statistics(
            all_mb_metrics=all_mb_metrics,
            losses=losses,
            data_parallel_group=parallel_state.get_data_parallel_group(),
        )

        metrics = {
            "global_loss": global_loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "gpu_name": torch.cuda.get_device_name(),
            "model_dtype": self.dtype,
            "all_mb_metrics": mb_metrics,
            "grad_norm": torch.tensor([grad_norm]),
        }

        # Collect MoE aux metrics if applicable
        num_moe_experts = getattr(self.model.config, "num_moe_experts", None)
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
        self.value_head.eval()

        policy_like_cfg = self._make_policy_like_config(self.cfg)

        (
            mb_iterator,
            num_microbatches,
            micro_batch_size_actual,
            seq_length,
            padded_seq_length,
        ) = get_microbatch_iterator(
            data,
            policy_like_cfg,
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

            # Bypass output_layer to avoid computing expensive logits
            captured_hidden = {}
            base_model = _unwrap_model(model)
            original_output_layer = None

            if hasattr(base_model, "output_layer"):
                original_output_layer = base_model.output_layer
                base_model.output_layer = _ValueOutputLayerBypass(captured_hidden)

            try:
                output_tensor = model(
                    input_ids=input_ids_cp_sharded,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    **additional_kwargs,
                )
            finally:
                if original_output_layer is not None:
                    base_model.output_layer = original_output_layer

            def collection_fn(output_tensor):
                # Compute values from hidden states
                if "hidden_states" in captured_hidden:
                    hidden_states = captured_hidden["hidden_states"]
                    # Megatron-Core always uses [S, B, H] layout; transpose to [B, S, H]
                    hidden_states = hidden_states.transpose(0, 1).contiguous()
                    values = self.value_head(hidden_states).squeeze(-1)  # [B, S]
                else:
                    # Non-last PP stage: return dummy
                    values = torch.zeros(1, device=output_tensor.device)

                # Handle sequence parallelism: gather across TP
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                if tp_size > 1 and self.megatron_cfg.model.sequence_parallel:
                    tp_group = get_tensor_model_parallel_group()
                    values = allgather_cp_sharded_tensor(
                        values.unsqueeze(0) if values.dim() == 1 else values,
                        tp_group,
                        seq_dim=1,
                    )

                # Handle context parallelism: gather across CP
                cp_size = parallel_state.get_context_parallel_world_size()
                if cp_size > 1:
                    cp_group = get_context_parallel_group()
                    values = allgather_cp_sharded_tensor(values, cp_group, seq_dim=1)

                # Prepend 0 value for first token to maintain sequence length
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

            tensors = {"values": torch.cat(all_values_padded, dim=0)}
        else:
            tensors = {"values": None}

        values_tensor = broadcast_tensors_from_last_stage(tensors)["values"]

        no_grad.__exit__(None, None, None)

        return BatchedDataDict[ValueOutputSpec](values=values_tensor.cpu())

    def prepare_for_training(self, *args, **kwargs):
        """Move model and optimizer to CUDA for training."""
        self.model = self.move_model(
            self.model, "cuda", move_grads=True, move_params=True
        )
        self.model.train()
        self.value_head.cuda().train()

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
        self.value_head.cuda().eval()

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

        # Also move value head optimizer
        if hasattr(self, "value_head_optimizer"):
            for state in self.value_head_optimizer.state.values():
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

        Saves both the Megatron backbone checkpoint and the value head weights.
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

            # Save value head weights separately (only on rank 0 to avoid conflicts)
            if torch.distributed.get_rank() == 0:
                value_head_path = os.path.join(weights_path, "value_head.pt")
                os.makedirs(os.path.dirname(value_head_path), exist_ok=True)
                torch.save(self.value_head.state_dict(), value_head_path)

                # Save value head optimizer if requested
                if optimizer_path is not None and hasattr(self, "value_head_optimizer"):
                    vh_opt_path = os.path.join(
                        optimizer_path, "value_head_optimizer.pt"
                    )
                    os.makedirs(os.path.dirname(vh_opt_path), exist_ok=True)
                    torch.save(self.value_head_optimizer.state_dict(), vh_opt_path)

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

    def finish_inference(self, *args: Any, **kwargs: Any) -> None:
        """Offload model params to CPU after inference."""
        self.model = self.move_model(
            self.model, "cpu", move_params=True, move_grads=False
        )
        self.value_head.cpu()

        gc.collect()
        torch.cuda.empty_cache()

    def finish_training(self, *args: Any, **kwargs: Any) -> None:
        """Offload model, gradients, and optimizer to CPU after training."""
        self.model = self.move_model(
            self.model, "cpu", move_params=True, move_grads=True
        )
        self.model.eval()
        self.value_head.cpu().eval()

        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()
