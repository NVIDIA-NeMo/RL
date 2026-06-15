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
import copy
import gc
import os
import re
import warnings
from collections import OrderedDict, defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Iterable, Iterator, Optional, TypeVar, cast

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
from megatron.core.inference.config import InferenceConfig
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_model_config
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.common import get_moe_metrics
from nemo_rl.models.megatron.config import MegatronGenerationConfig
from nemo_rl.models.megatron.data import (
    get_microbatch_iterator,
    process_global_batch,
)
from nemo_rl.models.megatron.pipeline_parallel import (
    broadcast_loss_metrics_from_last_stage,
    broadcast_obj_from_pp_rank,
    broadcast_tensors_from_last_stage,
)
from nemo_rl.models.megatron.setup import (
    finalize_megatron_setup,
    handle_model_import,
    setup_distributed,
    setup_model_and_optimizer,
    setup_reference_model_state,
    validate_and_set_config,
    validate_model_paths,
)
from nemo_rl.models.megatron.train import (
    LogprobsPostProcessor,
    LossPostProcessor,
    TopkLogitsPostProcessor,
    aggregate_training_statistics,
    megatron_forward_backward,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
)
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.nvml import log_gpu_memory_diagnostics
from nemo_rl.utils.packed_tensor import packed_broadcast_producer

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def _model_self_packs_for_cp(model: Any) -> bool:
    """Whether the model packs sequences + CP-shards inside its own forward.

    Such models (mbridge VLM wrappers) call ``preprocess_packed_seqs`` in their
    forward, so NeMo-RL must hand them an unpacked ``[B, S]`` batch instead of
    pre-packing + CP-sharding itself. The only such model today is mbridge's
    Qwen3VL, which is also the only mbridge VLM that supports context
    parallelism; classic mcore GPTModel and other VLMs do not self-pack.
    """
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.model import Qwen3VLModel
    from megatron.core.utils import unwrap_model

    unwrapped = unwrap_model(model)
    chunks = unwrapped if isinstance(unwrapped, (list, tuple)) else [unwrapped]
    return any(isinstance(chunk, Qwen3VLModel) for chunk in chunks)


@contextmanager
def _meta_tensor_alloc_context():
    """Skip real GPU work during metadata enumeration.

    Bridge's ``export_hf_weights`` does PP/TP/EP gathers to materialize
    full unsharded tensors, but the refit-info builders only need shape+dtype.
    Patch the allocators to redirect to ``meta`` and turn the collectives into
    no-ops.  Subsequent shape-only ops on meta tensors propagate correctly,
    while peak memory stays at zero extra GiB.
    """
    real_all_gather = torch.distributed.all_gather
    real_broadcast = torch.distributed.broadcast
    real_empty_like = torch.empty_like
    real_zeros_like = torch.zeros_like

    def _meta_empty_like(t, *a, **k):
        return torch.empty(t.shape, dtype=t.dtype, device="meta")

    def _meta_zeros_like(t, *a, **k):
        return torch.zeros(t.shape, dtype=t.dtype, device="meta")

    def _noop_all_gather(tensor_list, tensor, *a, **k):
        return None

    def _noop_broadcast(tensor, src, *a, **k):
        return None

    torch.distributed.all_gather = _noop_all_gather
    torch.distributed.broadcast = _noop_broadcast
    torch.empty_like = _meta_empty_like
    torch.zeros_like = _meta_zeros_like
    try:
        yield
    finally:
        torch.distributed.all_gather = real_all_gather
        torch.distributed.broadcast = real_broadcast
        torch.empty_like = real_empty_like
        torch.zeros_like = real_zeros_like


# Classes with @ray.remote can't be inherited from, so we split the implementation out.
# This is useful when using worker extension classes.
class MegatronPolicyWorkerImpl(
    TQWorkerMixin, AbstractPolicyWorker, ColocatablePolicyInterface
):
    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def _local_coords(self) -> dict[str, int]:
        if not torch.distributed.is_initialized():
            return {}
        return {
            "tensor_parallel": parallel_state.get_tensor_model_parallel_rank(),
            "context_parallel": parallel_state.get_context_parallel_rank(),
            "pipeline_parallel": parallel_state.get_pipeline_model_parallel_rank(),
        }

    def _get_replica_group(self) -> Optional[Any]:
        """Replica group = TP × CP × PP siblings within this DP rank.

        Always returns the real group so :meth:`_is_replica_leader` (used
        by both fetch and write-back) gives the correct single-writer
        answer even at CP=1 — gating on CP=1 here is what produced the
        ``-601 ILLEGAL_CLIENT`` duplicate-write bug. The fetch-path
        broadcast-vs-independent perf choice lives inside ``_fetch``
        keyed on ``replica_group.size()``.

        mcore exposes per-axis groups (``get_tensor_model_parallel_group``,
        ``get_context_parallel_group``, ``get_pipeline_model_parallel_group``)
        but no single combined group. We build the combined NCCL group
        once on first call by enumerating coordinates that share this
        rank's ``dp_rank``.
        """
        if not torch.distributed.is_initialized():
            return None
        cached = getattr(self, "_replica_group_cache", "uninit")
        if cached != "uninit":
            return cached

        world_size = torch.distributed.get_world_size()
        my_dp_rank = parallel_state.get_data_parallel_rank()
        # Collect global ranks that share this DP rank — they form the
        # replica group. Done collectively so every rank ends up with
        # the same ranks list and can pass it to new_group().
        my_replica_ranks_t = torch.full(
            (world_size,),
            -1,
            dtype=torch.long,
            device="cuda",
        )
        my_replica_ranks_t[torch.distributed.get_rank()] = my_dp_rank
        torch.distributed.all_reduce(
            my_replica_ranks_t, op=torch.distributed.ReduceOp.MAX
        )
        all_dp_ranks = my_replica_ranks_t.tolist()

        # Every (dp_rank → ranks) bucket must call new_group on its own
        # ranks list, but new_group itself must be called collectively
        # across the full world. Sort by dp_rank to keep call order
        # consistent across processes.
        groups: dict[int, Any] = {}
        for dp in sorted(set(all_dp_ranks)):
            ranks = [r for r, d in enumerate(all_dp_ranks) if d == dp]
            grp = torch.distributed.new_group(ranks=ranks, backend="nccl")
            groups[dp] = grp
        self._replica_group_cache = groups[my_dp_rank]
        return self._replica_group_cache

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: TokenizerType,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        *,
        worker_sharding_annotations: NamedSharding,
        **kwargs: Any,
    ):
        """Initialize the MegatronPolicyWorker."""
        log_gpu_memory_diagnostics(
            label="init_start", worker_type="MegatronPolicyWorker"
        )

        # Apply patch from https://github.com/NVIDIA/TransformerEngine/pull/2286/files
        apply_transformer_engine_patch()

        self.cfg = config

        # Set rank for non-collocated to check which ranks to broadcast from
        self.rank = get_rank_safe()

        # Step 1: Setup distributed
        setup_distributed()
        log_gpu_memory_diagnostics(
            label="after_nccl_init", worker_type="MegatronPolicyWorker"
        )

        # Step 2: Validate and setup model paths
        hf_model_name, pretrained_path, pt_checkpoint_exists = validate_model_paths(
            config
        )
        # Handle model import if needed. Subclasses (e.g. ModelOpt quant
        # worker) may set ``_model_import_post_wrap_hook`` and
        # ``_transformer_layer_spec`` on ``self`` before calling
        # super().__init__() to inject quantization hooks into HF->Megatron
        # import.
        handle_model_import(
            config,
            hf_model_name,
            pretrained_path,
            pt_checkpoint_exists,
            model_post_wrap_hook=getattr(self, "_model_import_post_wrap_hook", None),
            transformer_layer_spec=getattr(self, "_transformer_layer_spec", None),
        )
        log_gpu_memory_diagnostics(
            label="after_hf_import", worker_type="MegatronPolicyWorker"
        )

        # Store tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Step 3: Setup model configuration
        runtime_config = validate_and_set_config(
            config,
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

        self.defer_fp32_logits = self.cfg["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (runtime_config.model_cfg.fp16 or runtime_config.model_cfg.bf16)

        # Store FP8 config for later use
        self.fp8_cfg = config["megatron_cfg"].get("fp8_cfg", None)

        # Validate configuration
        self.megatron_cfg.validate()

        # Step 4: Setup Megatron model and components
        model_and_optimizer_state = setup_model_and_optimizer(
            config,
            self.megatron_cfg,
            init_optimizer,
            pre_load_checkpoint_hook=getattr(self, "_pre_load_checkpoint_hook", None),
        )

        self.mcore_state = model_and_optimizer_state.state
        self.model = model_and_optimizer_state.model
        self.optimizer = model_and_optimizer_state.optimizer
        self.scheduler = model_and_optimizer_state.scheduler
        self.checkpointing_context = model_and_optimizer_state.checkpointing_context
        param_sync_func = model_and_optimizer_state.param_sync_func
        self.draft_model = model_and_optimizer_state.draft_model
        log_gpu_memory_diagnostics(
            label="after_model_setup", worker_type="MegatronPolicyWorker"
        )

        # Set the param sync function for the model if needed
        if param_sync_func is not None:
            get_model_config(self.model).param_sync_func = param_sync_func

        # Step 5: Setup reference model if needed
        if init_reference_model:
            self.model = self.move_model(self.model, "cpu")
            self.reference_state_dict = setup_reference_model_state(
                config,
                self.megatron_cfg,
                pretrained_path,
                pre_load_checkpoint_hook=getattr(
                    self, "_pre_load_checkpoint_hook", None
                ),
            )
            self.model = self.move_model(self.model, "cuda")
            log_gpu_memory_diagnostics(
                label="after_ref_model", worker_type="MegatronPolicyWorker"
            )

        # Step 6: Finalize setup
        (
            self.megatron_tokenizer,
            self.megatron_bridge,
            self.should_disable_forward_pre_hook,
            self.dp_size,
        ) = finalize_megatron_setup(
            config,
            self.megatron_cfg,
            hf_model_name,
            worker_sharding_annotations,
            self.model,
            self.optimizer,
        )
        self._first_train_step_forward_pre_hook_disabled = False
        self._first_train_step_param_sync_func = None
        if self.should_disable_forward_pre_hook and self._forward_pre_hook_enabled():
            self._disable_forward_pre_hook_until_next_train_step()

        # Whether the model packs sequences + CP-shards inside its own forward
        # (mbridge VLM wrappers like Qwen3VL). If so, NeMo-RL must hand it an
        # unpacked [B, S] batch rather than pre-packing + CP-sharding itself.
        self.delegate_pack_to_model = _model_self_packs_for_cp(self.model)

        # vars used for refit
        ## will be initialized in prepare_refit_info
        # refit_param_info_mcore combines the conversion tasks with the param memory
        # [(mcore_param_name, estimated_memory), ...]
        # Note: here param name is local param name, with local layer number and
        # local expert id etc.
        self.refit_conversion_tasks = None
        self.refit_conversion_tasks_current_index = None
        self.refit_param_info_mcore = None

        ## used for streaming update inference engine weights
        self._held_gather_buffer = None

        log_gpu_memory_diagnostics(
            label="init_complete", worker_type="MegatronPolicyWorker"
        )

    def enable_forward_pre_hook(self):
        assert isinstance(self.model, DistributedDataParallel)
        if not self._forward_pre_hook_enabled():
            self.model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        assert isinstance(self.model, DistributedDataParallel)
        if not self._forward_pre_hook_enabled():
            return
        if param_sync:
            self._copy_main_params_to_param_buffer(zero_grad_buffer=True)
        self.model.disable_forward_pre_hook(param_sync=param_sync)

    def _forward_pre_hook_enabled(self) -> bool:
        if not isinstance(self.model, DistributedDataParallel):
            return False
        return len(getattr(self.model, "remove_forward_pre_hook_handles", {})) > 0

    def _disable_forward_pre_hook_until_next_train_step(self) -> None:
        assert isinstance(self.model, DistributedDataParallel)
        if self._forward_pre_hook_enabled():
            self.disable_forward_pre_hook(param_sync=False)
        model_config = get_model_config(self.model)
        self._first_train_step_param_sync_func = model_config.param_sync_func
        model_config.param_sync_func = None
        self._first_train_step_forward_pre_hook_disabled = True

    def _copy_main_params_to_param_buffer(self, zero_grad_buffer: bool = False) -> None:
        if not isinstance(self.model, DistributedDataParallel):
            return

        if not self._uses_mxfp8_overlap_shared_param_buffer():
            return

        if not self._forward_pre_hook_enabled():
            return

        if zero_grad_buffer:
            self.model.zero_grad_buffer()

        optimizers = (
            self.optimizer.chained_optimizers
            if isinstance(self.optimizer, ChainedOptimizer)
            else [self.optimizer]
        )
        for optim_instance in optimizers:
            if hasattr(optim_instance, "_copy_main_params_to_param_buffer"):
                optim_instance._copy_main_params_to_param_buffer()

    def _uses_mxfp8_overlap_shared_param_buffer(self) -> bool:
        return getattr(
            self.megatron_cfg.optimizer, "reuse_grad_buf_for_mxfp8_param_ag", False
        ) and getattr(self.megatron_cfg.ddp, "overlap_param_gather", False)

    def _get_model_extra_state_dict(self) -> dict[str, Any]:
        fp8_enabled = self.fp8_cfg and self.fp8_cfg.get("enabled", False)
        if not fp8_enabled:
            return {}
        extra_state = {}
        for key, value in self.model.state_dict().items():
            if "._extra_state" not in key:
                continue
            if isinstance(value, torch.Tensor):
                extra_state[key] = value.detach().clone()
            else:
                extra_state[key] = copy.deepcopy(value)
        return extra_state

    def _restore_model_extra_state_dict(self, extra_state: dict[str, Any]) -> None:
        if not extra_state:
            return
        self.model.load_state_dict(extra_state, strict=False)

    @wrap_with_nvtx_name("megatron_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        check_dim_skip_keys: Optional[Iterable[str]] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function.

        ``check_dim_skip_keys`` is accepted for parity with the v1/v2 DTensor
        workers (cross-tokenizer ride-along tensors whose dim 1 is not the
        student sequence axis). Megatron doesn't run cross-tokenizer, so it
        must be None.
        """
        assert check_dim_skip_keys is None, (
            "check_dim_skip_keys is only supported by the v2 DTensor worker; "
            "Megatron does not run cross-tokenizer distillation."
        )
        # Note: zero_grad_buffer is called at the start of each global batch iteration
        # in the loop below, so we don't need to call it here.
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
            saved_extra_state = self._get_model_extra_state_dict()
            reenable_forward_pre_hook_after_eval = (
                self.should_disable_forward_pre_hook
                and self._forward_pre_hook_enabled()
            )
            if reenable_forward_pre_hook_after_eval:
                self.disable_forward_pre_hook()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()
            saved_extra_state = None
            reenable_forward_pre_hook_after_eval = False

        with ctx:
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
                    self.cfg,
                    mbs,
                    straggler_timer=self.mcore_state.straggler_timer,
                    delegate_pack_to_model=self.delegate_pack_to_model,
                )
                # Track total microbatches for MoE aux-loss averaging
                total_num_microbatches += int(num_microbatches)

                loss_post_processor = LossPostProcessor(
                    loss_fn=loss_fn,
                    cfg=self.cfg,
                    num_microbatches=num_microbatches,
                    sampling_params=self.sampling_params,
                    draft_model=self.draft_model,
                )

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Set grad to zero. For MXFP8 overlap eval, the param and
                    # grad buffers are shared and pre-hooks are disabled above.
                    # Avoid zeroing the shared param buffer before forward-only eval.
                    if not (
                        eval_mode and self._uses_mxfp8_overlap_shared_param_buffer()
                    ):
                        self.model.zero_grad_buffer()
                        self.optimizer.zero_grad()
                        self._copy_main_params_to_param_buffer()

                    # Forward pass.
                    draft_enabled = "draft" in self.cfg and self.cfg["draft"]["enabled"]
                    losses_reduced = megatron_forward_backward(
                        model=self.model,
                        data_iterator=data_iterator,
                        num_microbatches=num_microbatches,
                        seq_length=padded_seq_length,
                        mbs=micro_batch_size,
                        post_processing_fn=loss_post_processor,
                        forward_only=eval_mode,
                        defer_fp32_logits=self.defer_fp32_logits,
                        global_valid_seqs=global_valid_seqs,
                        global_valid_toks=global_valid_toks,
                        sampling_params=self.sampling_params,
                        straggler_timer=self.mcore_state.straggler_timer,
                        draft_model=self.draft_model,
                        enable_hidden_capture=draft_enabled,
                        use_linear_ce_fusion_loss=self.cfg["megatron_cfg"].get(
                            "use_linear_ce_fusion_loss", False
                        ),
                    )

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters.
                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (True, 0.0, 0.0)

                pg_collection = get_pg_collection(self.model)

                # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
                # so we must gather across mp ranks
                update_successful = logical_and_across_model_parallel_group(
                    update_successful, mp_group=pg_collection.mp
                )
                # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
                # so we must gather across mp ranks
                grad_norm: float = reduce_max_stat_across_model_parallel_group(
                    grad_norm, mp_group=pg_collection.mp
                )
                num_zeros_in_grad: float = reduce_max_stat_across_model_parallel_group(
                    num_zeros_in_grad, mp_group=pg_collection.mp
                )
                if (
                    not eval_mode
                    and self._first_train_step_forward_pre_hook_disabled
                    and update_successful
                ):
                    self.enable_forward_pre_hook()
                    get_model_config(
                        self.model
                    ).param_sync_func = self._first_train_step_param_sync_func
                    self._first_train_step_param_sync_func = None
                    self._first_train_step_forward_pre_hook_disabled = False

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
                    torch.cuda.empty_cache()

                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # keep all microbatch metrics to be normalized later
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

        if saved_extra_state is not None:
            self._restore_model_extra_state_dict(saved_extra_state)
        if reenable_forward_pre_hook_after_eval:
            # A forced param sync leaves the next training step with no pending
            # param AG to finish. Keep hooks disabled for that one step so grad
            # accumulation starts from a clean shared param/grad buffer.
            self._disable_forward_pre_hook_until_next_train_step()

        if not eval_mode:
            # Step LR scheduler once per train() call, not per global batch.
            # Megatron's OptimizerParamScheduler.step takes an `increment` in
            # samples: NeMo init scales lr_warmup_steps by gbs internally, so
            # passing increment=gbs cancels that scaling and one tick == one
            # train() call regardless of batch size.
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
        # Read "config" via getattr-by-string so the token stays out of
        # train.__code__.co_names; with torch 2.11 cloudpickle otherwise
        # matches torch.distributed.config (a non-pickleable ConfigModuleInstance).
        model_config = getattr(self.model, "config", None)
        num_moe_experts = getattr(model_config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"]["moe_per_layer_logging"],
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics
        return metrics

    @wrap_with_nvtx_name("megatron_policy_worker/get_logprobs")
    def get_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.
        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.
        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        self.model.eval()

        (
            mb_iterator,
            num_microbatches,
            micro_batch_size,
            seq_length,
            padded_seq_length,
        ) = get_microbatch_iterator(
            data,
            self.cfg,
            logprob_batch_size,
            straggler_timer=self.mcore_state.straggler_timer,
            delegate_pack_to_model=self.delegate_pack_to_model,
        )

        use_linear_ce_fusion = self.cfg["megatron_cfg"].get(
            "use_linear_ce_fusion_loss", False
        )
        logprobs_post_processor = LogprobsPostProcessor(
            cfg=self.cfg,
            sampling_params=self.sampling_params,
            use_linear_ce_fusion=use_linear_ce_fusion,
        )

        list_of_logprobs = megatron_forward_backward(
            model=self.model,
            data_iterator=mb_iterator,
            seq_length=padded_seq_length,
            mbs=micro_batch_size,
            num_microbatches=num_microbatches,
            post_processing_fn=logprobs_post_processor,
            forward_only=True,
            defer_fp32_logits=self.defer_fp32_logits,
            sampling_params=self.sampling_params,
            straggler_timer=self.mcore_state.straggler_timer,
            use_linear_ce_fusion_loss=use_linear_ce_fusion,
        )

        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            all_log_probs_padded = []
            all_logprobs = [l["logprobs"] for l in list_of_logprobs]
            for lp in all_logprobs:
                padding_needed = seq_length - lp.shape[1]
                if padding_needed > 0:
                    lp = torch.nn.functional.pad(
                        lp, (0, padding_needed), mode="constant", value=0.0
                    )
                all_log_probs_padded.append(lp)

            logprobs = torch.cat(all_log_probs_padded, dim=0)
            tensors = {"logprobs": logprobs}
        else:
            tensors = {"logprobs": None}
        logprobs = broadcast_tensors_from_last_stage(tensors)["logprobs"]

        no_grad.__exit__(None, None, None)
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs).to("cpu")

    def _apply_state_dict_to_model(
        self,
        source_state_dict: dict,
        *,
        raise_if_key_missing: bool = False,
    ) -> None:
        """Apply a state dict to self.model in-place.

        - Tensors with matching shape: in-place copy (parameters / buffers).
        - _extra_state keys (e.g. FP8 scale/amax) with shape mismatch or non-Tensor value:
          resolve the submodule and call set_extra_state(); supports DDP and Float16Module unwrap.

        Args:
            source_state_dict: State dict to apply (e.g. reference_state_dict or saved model_state_dict).
            raise_if_key_missing: If True, raise when a key in self.model.state_dict() is missing
                from source_state_dict; if False, skip such keys.
        """
        for state_dict_key, param_or_buf in self.model.state_dict().items():
            if (
                not isinstance(param_or_buf, torch.Tensor)
                or "draft_model." in state_dict_key
            ):
                continue
            if state_dict_key not in source_state_dict:
                if raise_if_key_missing:
                    raise ValueError(
                        f"Key '{state_dict_key}' not in source state_dict."
                    )
                continue
            source_value = source_state_dict[state_dict_key]

            # Case 1: Same shape → in-place copy (parameters / buffers)
            if (
                isinstance(source_value, torch.Tensor)
                and param_or_buf.shape == source_value.shape
            ):
                param_or_buf.copy_(source_value)
                continue

            # Case 2: _extra_state (shape mismatch or non-Tensor) → set_extra_state()
            assert "extra_state" in state_dict_key, (
                f"the {state_dict_key} is not an extra_state, but the param_or_buf is mismatched with the reference_state_dict {source_value.shape} != {param_or_buf.shape}."
            )

            submodule_path = state_dict_key.rsplit("._extra_state", 1)[0]
            base_module = getattr(self.model, "module", self.model)
            # Unwrap Float16Module/MoEFloat16Module: state_dict keys are relative to inner .module
            top_level_name = submodule_path.split(".", 1)[0]
            if not hasattr(base_module, top_level_name):
                base_module = getattr(base_module, "module", base_module)
            target_module = base_module.get_submodule(submodule_path)
            target_module.set_extra_state(source_value)

    @contextmanager
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references.
                  Also disables top-k/top-p filtering since the reference policy's distribution
                  is different from the current policy, making filtered logprobs incompatible.
        On exit: Restores original references and re-flips cuda/cpu, restores sampling_params.
        """
        ## disable overlap param gather when swapping weights
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        with torch.no_grad():
            # Save original references
            model_state_dict = {}
            for name, item in self.model.state_dict().items():
                if isinstance(item, torch.Tensor):
                    item = item.detach().to(device="cpu", non_blocking=True, copy=True)
                model_state_dict[name] = item

            # Swap reference model state_dict into self.model (reference weights + optional FP8 extra_state)
            self._apply_state_dict_to_model(
                self.reference_state_dict,
                raise_if_key_missing=True,
            )

            if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                gc.collect()
                torch.cuda.empty_cache()

            # Temporarily disable top-k/top-p filtering for reference policy logprobs.
            # The reference policy has different weights, so its top-k/top-p set is
            # inherently different from the current policy. Using filtered logprobs
            # would cause -inf mismatches that cannot be resolved by masking.
            # Note: We keep temperature scaling since it was applied to prev_logprobs.
            saved_sampling_params = self.sampling_params
            if saved_sampling_params is not None:
                self.sampling_params = TrainingSamplingParams(
                    top_k=None,
                    top_p=1.0,
                    temperature=saved_sampling_params.temperature,
                )
            else:
                self.sampling_params = None

            # - self.model is the original reference_model, now on CUDA
            # - self.reference_model is the original model, now on CPU
            yield

            # Restore sampling_params
            self.sampling_params = saved_sampling_params

            # Restore original policy state (weights + FP8 extra_state) from saved model_state_dict
            self._apply_state_dict_to_model(
                model_state_dict,
                raise_if_key_missing=True,
            )

            if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                gc.collect()
                torch.cuda.empty_cache()

            ## re-enable overlap param gather after weight swap
            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

    @wrap_with_nvtx_name("megatron_policy_worker/get_topk_logits")
    def get_topk_logits(
        self,
        *,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
    ):
        """Get the top-k logits and indices for a batch of data.

        The major difference from get_logprobs is that we compute top-k logits and indices for each position in the sequence.

        Returns:
            BatchedDataDict containing:
                - topk_logits: Tensor of top-k logits for each position in the sequence
                - topk_indices: Tensor of top-k indices for each position in the sequence
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()

        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        self.model.eval()

        (
            mb_iterator,
            num_microbatches,
            micro_batch_size,
            seq_length,
            padded_seq_length,
        ) = get_microbatch_iterator(
            data,
            self.cfg,
            logprob_batch_size,
            straggler_timer=self.mcore_state.straggler_timer,
            delegate_pack_to_model=self.delegate_pack_to_model,
        )

        list_of_outputs = megatron_forward_backward(
            model=self.model,
            data_iterator=mb_iterator,
            seq_length=padded_seq_length,
            mbs=micro_batch_size,
            num_microbatches=num_microbatches,
            post_processing_fn=TopkLogitsPostProcessor(cfg=self.cfg, k=k),
            forward_only=True,
            defer_fp32_logits=self.defer_fp32_logits,
            sampling_params=self.sampling_params,
            straggler_timer=self.mcore_state.straggler_timer,
        )

        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            logits_chunks = []
            indices_chunks = []
            for out in list_of_outputs:
                tk = out["topk_logits"]
                ti = out["topk_indices"]
                pad_len = seq_length - tk.shape[1]
                if pad_len > 0:
                    tk = torch.nn.functional.pad(tk, (0, 0, 0, pad_len), value=0.0)
                    ti = torch.nn.functional.pad(ti, (0, 0, 0, pad_len), value=0)
                logits_chunks.append(tk)
                indices_chunks.append(ti)

            topk_logits = torch.cat(logits_chunks, dim=0)
            topk_indices = torch.cat(indices_chunks, dim=0)

            tensors_to_broadcast = {
                "topk_logits": topk_logits,
                "topk_indices": topk_indices,
            }
        else:
            tensors_to_broadcast = {
                "topk_logits": None,
                "topk_indices": None,
            }

        # Broadcast tensors from last stage to all stages
        broadcasted = broadcast_tensors_from_last_stage(tensors_to_broadcast)
        topk_logits = broadcasted["topk_logits"]
        topk_indices = broadcasted["topk_indices"]

        no_grad.__exit__(None, None, None)
        return BatchedDataDict.from_batches(
            [{"topk_logits": topk_logits.cpu(), "topk_indices": topk_indices.cpu()}]
        )

    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - ``output_ids``: input + generated token IDs
                - ``logprobs``: Log probabilities for each token
                - ``generation_lengths``: Lengths of each response
        """
        # 512 bATCH SIZE (200 tokens)
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model.config.flash_decode = False
        if self.should_disable_forward_pre_hook:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
        # Verify input is right padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            f"input_ids and input_lengths must be present in the BatchedDataDict, got keys: {data.keys()}"
        )
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
            )

        mcore_generation_config = cast(
            MegatronGenerationConfig, self.cfg["generation"]["mcore_generation_config"]
        )

        from megatron.core.inference.contexts.dynamic_context import (
            DynamicInferenceContext,
        )
        from megatron.core.inference.engines.dynamic_engine import (
            DynamicInferenceEngine,
        )
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )
        from megatron.core.inference.sampling_params import SamplingParams
        from megatron.core.inference.utils import InferenceMode

        model_config = self.model.config
        model_config.cuda_graph_impl = "local"

        local_rank = torch.cuda.current_device()
        num_gpus_per_node = torch.cuda.device_count()
        node_idx = self.rank // num_gpus_per_node if num_gpus_per_node > 0 else 0
        model_config.inference_sampling_seed = (node_idx * 1024) + local_rank

        inference_config = InferenceConfig(
            max_sequence_length=self.cfg["generation"]["max_new_tokens"],
            buffer_size_gb=mcore_generation_config["buffer_size_gb"],
            num_cuda_graphs=mcore_generation_config["num_cuda_graphs"],
            block_size_tokens=mcore_generation_config["block_size_tokens"],
            use_cuda_graphs_for_non_decode_steps=mcore_generation_config[
                "use_cuda_graphs_for_non_decode_steps"
            ],
            enable_chunked_prefill=mcore_generation_config["enable_chunked_prefill"],
            unified_memory_level=mcore_generation_config["unified_memory_level"],
            max_tokens=mcore_generation_config["max_tokens"],
            materialize_only_last_token_logits=False,
            use_flashinfer_fused_rope=False,
        )

        dynamic_context = DynamicInferenceContext(model_config, inference_config)
        inference_wrapped_model = GPTInferenceWrapper(self.model, dynamic_context)

        inference_wrapped_model.prep_model_for_inference()
        # Set pipeline parallel flag
        inference_wrapped_model.model_is_pipeline_parallel = (
            self.cfg["megatron_cfg"]["pipeline_model_parallel_size"] > 1
        )

        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )

        with InferenceMode.active():
            dynamic_engine = DynamicInferenceEngine(
                text_generation_controller,
                dynamic_context,
            )

            # Handle None values for top_k - convert to integer as required by Megatron
            top_k_cfg = self.cfg["generation"]["top_k"]
            top_k_val = (
                1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)
            )

            top_p_cfg = self.cfg["generation"]["top_p"]
            top_p_val = (
                0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
            )

            # New API: SamplingParams now includes termination_id and uses num_tokens_total
            sampling_params = SamplingParams(
                temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
                top_k=top_k_val,
                top_p=top_p_val,
                skip_prompt_log_probs=False,
                return_log_probs=True,
                num_tokens_total=self.cfg["generation"]["max_new_tokens"],
                num_tokens_to_generate=None,
                termination_id=self.megatron_tokenizer.eod,
            )

            input_ids = data["input_ids"]
            prompt_tokens_tensor = input_ids.cuda()
            prompt_lengths_tensor = data["input_lengths"]
            request_id = 0

            # New API: add_request now takes sampling_params as a parameter
            for p, prompt_len in zip(
                prompt_tokens_tensor, prompt_lengths_tensor, strict=True
            ):
                dynamic_engine.add_request(
                    request_id,
                    p[:prompt_len],
                    sampling_params=sampling_params,
                )
                request_id += 1

            result = []
            while dynamic_engine.has_unfinished_requests():
                result_step = dynamic_engine.step_modern()
                result.extend(result_step["finished_request_records"])

        # Sort results by request_id to maintain original batch order
        result.sort(key=lambda x: x.request_id)

        out = {
            "tokens": [
                x.requests[0].prompt_tokens.tolist() + x.requests[0].generated_tokens
                for x in result
            ],
            "logprobs": [
                x.requests[0].prompt_log_probs + x.requests[0].generated_log_probs
                for x in result
            ],
        }

        input_lengths = data["input_lengths"]
        # pad the out "tokens" and "logprobs" and make them into tensors from lists
        batch_size = data["input_ids"].size(0)
        max_gen_seq_len = max([len(x.requests[0].generated_tokens) for x in result])
        padded_input_length = input_ids.size(1)

        max_seq_len = padded_input_length + max_gen_seq_len
        # Create padded tensors for tokens and logprobs
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=data["input_ids"].device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=data["input_ids"].device,
        )

        # Fill in the padded tensors with actual values
        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        for i in range(batch_size):
            seq_len = len(out["tokens"][i])
            output_ids_padded[i, :seq_len] = torch.tensor(
                out["tokens"][i], dtype=torch.long, device=data["input_ids"].device
            )
            generation_lengths[i] = seq_len - input_lengths[i].item()
            unpadded_sequence_lengths[i] = seq_len
            logprob_len = len(out["logprobs"][i])
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                out["logprobs"][i],
                dtype=torch.float,
                device=data["input_ids"].device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        self.model.config.flash_decode = False
        no_grad.__exit__(None, None, None)

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/prepare_refit_info")
    def prepare_refit_info(self) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming."""
        self.refit_param_info_mcore = self._calculate_refit_param_info()

        # Collect tensor metadata for refit / hf side info.
        refit_param_info_hf = {}
        for name, tensor in self._iter_params_with_optional_kv_scales():
            refit_param_info_hf[name] = (tensor.shape, tensor.dtype)

        return refit_param_info_hf

    def _is_fp8_export(self) -> bool:
        """Return True if the train side stores weights as TE blockwise FP8."""
        if self.fp8_cfg is None:
            return False
        return bool(
            self.fp8_cfg.get("fp8_param", False)
            and self.fp8_cfg.get("fp8_recipe") == "blockwise"
        )

    def _build_refit_conversion_tasks(self) -> list:
        """Build the conversion-task list driving refit (BF16 or FP8 export).

        For BF16 / FP8-but-fp8_param=False training: standard ``get_conversion_tasks``.
        For FP8-with-fp8_param=True: Bridge's ``build_export_fp8_tasks``, which
        emits a *pair* of tasks per FP8 weight (the FP8 data and a ``*_scale_inv``
        scale tensor).
        """
        if self._is_fp8_export():
            return self.megatron_bridge._model_bridge.build_export_fp8_tasks(
                self.megatron_bridge.hf_pretrained, [self.model]
            )
        return [
            task
            for task in self.megatron_bridge.get_conversion_tasks([self.model])
            if task is not None
        ]

    def _calculate_refit_param_info(self) -> list[tuple[str, int]]:
        """Calculate parameter information for refit.

        Each task contains:
        - param_name: Local parameter name without module prefixes
        - mapping: MegatronParamMapping instance for weight transformation
        - pp_rank: Pipeline-parallel rank owning the parameter
        - vp_stage: Virtual-pipeline stage index
        - megatron_module: Reference to Megatron model/submodule
        - param_weight: Target parameter tensor for converted weight

        Returns:
            List of (parameter_name, size_in_bytes) tuples.
        """
        self.refit_conversion_tasks = self._build_refit_conversion_tasks()
        param_info = []

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                # need to broadcast for other pp ranks
                size_in_bytes = None
            else:
                # Calculate size for this parameter
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float16: 2,
                    torch.float32: 4,
                    torch.float8_e4m3fn: 1,
                    torch.float8_e5m2: 1,
                    torch.uint8: 1,
                }
                scale = prec_to_bytes[self.dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = (
                    param.element_size() * param.numel() * tp_size * ep_size * scale
                )

            # Broadcast size_in_bytes across pipeline parallel ranks
            return broadcast_obj_from_pp_rank(size_in_bytes)

        for task in self.refit_conversion_tasks:
            param_info.append(
                (
                    task.param_name,
                    calculate_size_in_bytes(
                        task.param_weight,
                        task.mapping.tp_size,
                        task.mapping.ep_size if task.mapping.is_expert else 1,
                    ),
                )
            )
        return param_info

    def _iter_params_with_optional_kv_scales(
        self,
        kv_scales: Optional[dict[str, float]] = None,
        conversion_tasks=None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield exported HF parameters and optionally append FP8 KV/Q scale tensors.

        This helper is used by both IPC-based streaming and collective broadcast
        so that the logic for adding KV scales stays consistent in one place.

        ``conversion_tasks`` (optional) overrides ``self.refit_conversion_tasks``
        — used by the nccl_xfer_refit misc-refit path to pass a filtered subset so
        Bridge only does TP/EP all-gather for those tasks instead of the full model.
        """
        from nemo_rl.models.generation.vllm.quantization.fp8_train_utils import (
            get_vllm_qkv_scale_names,
        )

        if conversion_tasks is None:
            # Default to the full conversion tasks
            conversion_tasks = self.refit_conversion_tasks

        base_iter = self.megatron_bridge.export_hf_weights(
            [self.model],
            show_progress=False,
            conversion_tasks=conversion_tasks,  # used for metadata caching
        )

        # Yield the original parameters first.
        for name, tensor in base_iter:
            yield name, tensor

        if self.draft_model is not None:
            from nemo_rl.models.megatron.draft import export_eagle_weights_to_hf

            draft_weights = export_eagle_weights_to_hf(
                self.draft_model,
            )
            for name, tensor in draft_weights:
                yield f"draft.{name}", tensor

        # Check whether FP8 KV cache is enabled.
        use_fp8_kv_cache = False
        if (
            "generation" in self.cfg
            and self.cfg["generation"] is not None
            and self.cfg["generation"]["backend"] == "vllm"
        ):
            generation_cfg = cast(VllmConfig, self.cfg["generation"])
            use_fp8_kv_cache = (
                "vllm_cfg" in generation_cfg
                and "kv_cache_dtype" in generation_cfg["vllm_cfg"]
                and generation_cfg["vllm_cfg"]["kv_cache_dtype"].startswith("fp8")
            )

        if not use_fp8_kv_cache:
            return

        # Append KV (and potentially Q) scale entries to match metadata.
        num_layers = self.megatron_bridge.transformer_config.num_layers
        keys: list[str] = []
        for layer_idx in range(num_layers):
            scale_names = get_vllm_qkv_scale_names(layer_idx)
            keys.extend(scale_names.values())

        for param_name in keys:
            if kv_scales and param_name in kv_scales:
                scale_value = kv_scales[param_name]
            else:
                scale_value = 1.0
            scale_tensor = torch.tensor(
                scale_value, dtype=torch.float32, device="cuda"
            ).reshape(1)
            yield param_name, scale_tensor

    def _iter_local_hf_param_shards(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield (hf_name, local_tp_shard) for locally owned params.

        This function is used by the nccl_xfer_refit path, when building the
        _param_map in build_nccl_xfer_refit_info.

        Unlike ``_iter_params_with_optional_kv_scales`` which does PP broadcast
        + TP gather via ``export_hf_weights``, this method yields TP-local
        shards directly from Megatron parameters — no collectives needed.

        Compound mappings (QKV, GatedMLP) are split into individual HF params
        on the TP-local shard. The returned tensors are views of the model
        parameters and must not be modified in-place.

        Regarding the EP case, the self.refit_conversion_tasks already contains
        the local expert weights only.

        For the PP, non-local params, task.param_weight is None from
        the refit_conversion_tasks.
        """
        from types import SimpleNamespace

        from megatron.bridge.models.conversion.param_mapping import (
            GatedMLPMapping,
            QKVMapping,
            split_qkv_weights,
        )

        config = self.megatron_bridge.transformer_config

        # non-local expert weights are not included in the refit_conversion_tasks
        for task in self.refit_conversion_tasks:
            # task.param_weight is the local tensor of the megatron model
            local_tensor = task.param_weight

            if local_tensor is None:
                continue  # Non-local PP rank

            # scale tensors are routed to the misc path, so this function should
            # skip the scale tensors. See is_nccl_xfer_param in nccl_xfer_utils.py
            if task.global_param_name.endswith("_scale_inv"):
                continue

            hf_param = task.mapping.hf_param

            if isinstance(hf_param, dict):
                if isinstance(task.mapping, QKVMapping):
                    # Split interleaved QKV into separate Q, K, V on TP-local shard.
                    # When tp_size > num_query_groups, we route the QKV to the misc path
                    # in prepare_nccl_xfer_refit_info since there is too much complication there
                    tp_size = task.mapping.tp_size
                    if config.num_query_groups < tp_size:
                        continue
                    local_config = SimpleNamespace(
                        num_attention_heads=config.num_attention_heads // tp_size,
                        num_query_groups=config.num_query_groups // tp_size,
                        kv_channels=config.kv_channels,
                        hidden_size=config.hidden_size,
                        attention_output_gate=getattr(
                            config, "attention_output_gate", False
                        ),
                    )
                    q, k, v = split_qkv_weights(local_config, local_tensor)
                    yield hf_param["q"], q
                    yield hf_param["k"], k
                    yield hf_param["v"], v
                elif isinstance(task.mapping, GatedMLPMapping):
                    # Each TP shard of linear_fc1 is [gate_shard; up_shard] along dim 0
                    gate, up = torch.chunk(local_tensor, 2, dim=0)
                    yield hf_param["gate"], gate
                    yield hf_param["up"], up
                else:
                    raise ValueError(
                        f"Unsupported compound mapping: {type(task.mapping).__name__}"
                    )
            else:
                # Simple 1→1 mapping
                yield str(hf_param), local_tensor

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(
        self, buffer_size_bytes: int = 0, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        self.maybe_init_zmq()

        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        # Use the shared implementation to append optional KV scales.
        stream_weights_via_ipc_zmq_impl(
            params_generator=self._iter_params_with_optional_kv_scales(
                kv_scales=kv_scales
            ),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Broadcast the weights for collective communication."""
        # param_iterator will return (name, tensor), we only need tensor.
        packed_broadcast_producer(
            iterator=self._iter_params_with_optional_kv_scales(kv_scales=kv_scales),
            group=self.model_update_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

    def _build_layer_to_pp_stage(self, pp_size: int) -> dict[str, int]:
        """Build mapping from layer group name to PP stage index.

        Returns a dictionary that maps the layer group name to the PP stage index.

        Mirrors Megatron-LM's ``get_num_layers_to_build``
        (``transformer_block.py``) for the standard (non-VP, non-custom-layout)
        path: middle stages share ``num_layers - first - last`` evenly, while
        the first/last stages get their explicit counts when set.

        Cases not yet supported are asserted out so failures are loud rather
        than silently producing wrong layer→stage mappings:
          - ``pipeline_model_parallel_layout`` (e.g. DeepSeek-V3)
          - ``virtual_pipeline_model_parallel_size`` (interleaved PP)
          - ``account_for_embedding_in_pipeline_split``
          - ``account_for_loss_in_pipeline_split``
        These cases are checked in check_nccl_xfer_refit_support function.
        """
        # Read from the runtime model's config rather than the bridge's
        # default — the user's per-stage layout overrides
        # (num_layers_in_first/last_pipeline_stage) are applied to the model
        # in setup but never make it into bridge.transformer_config.
        config = self.model.config

        assert getattr(config, "pipeline_model_parallel_layout", None) is None, (
            "nccl_xfer_refit does not support custom pipeline_model_parallel_layout yet"
        )
        assert getattr(config, "virtual_pipeline_model_parallel_size", None) in (
            None,
            1,
        ), (
            "nccl_xfer_refit does not support virtual_pipeline_model_parallel_size > 1 yet"
        )
        assert not getattr(config, "account_for_embedding_in_pipeline_split", False), (
            "nccl_xfer_refit does not support account_for_embedding_in_pipeline_split yet"
        )
        assert not getattr(config, "account_for_loss_in_pipeline_split", False), (
            "nccl_xfer_refit does not support account_for_loss_in_pipeline_split yet"
        )

        num_layers = config.num_layers
        n_first = getattr(config, "num_layers_in_first_pipeline_stage", None)
        n_last = getattr(config, "num_layers_in_last_pipeline_stage", None)

        layers_to_distribute = num_layers
        stages_left = pp_size
        if n_first is not None:
            layers_to_distribute -= n_first
            stages_left -= 1
        if n_last is not None:
            layers_to_distribute -= n_last
            stages_left -= 1

        if stages_left > 0:
            assert layers_to_distribute % stages_left == 0, (
                f"With uneven pipelining the leftover layers ({layers_to_distribute}) "
                f"must be divisible by leftover stages ({stages_left})"
            )
            middle_per_stage = layers_to_distribute // stages_left
        else:
            middle_per_stage = 0

        layer_to_pp_stage: dict[str, int] = {}
        layer_idx = 0
        for stage in range(pp_size):
            if stage == 0 and n_first is not None:
                count = n_first
            elif stage == pp_size - 1 and n_last is not None:
                count = n_last
            else:
                count = middle_per_stage
            for _ in range(count):
                # Emit both HF naming conventions so the lookup in
                # build_nccl_xfer_refit_info hits regardless of model family:
                # Llama/Qwen use ``model.layers.N``; NemotronH uses
                # ``backbone.layers.N``.  Keys are looked up (not iterated), so
                # the unused convention's extra keys are harmless.
                layer_to_pp_stage[f"model.layers.{layer_idx}"] = stage
                layer_to_pp_stage[f"backbone.layers.{layer_idx}"] = stage
                layer_idx += 1

        assert layer_idx == num_layers, (
            f"Layer assignment incomplete: assigned {layer_idx} of {num_layers}"
        )

        # Embedding on the first stage, final norm + lm_head on the last.
        # model.* = Llama/Qwen; backbone.* = NemotronH (embeddings / norm_f).
        layer_to_pp_stage["model.embed_tokens"] = 0
        layer_to_pp_stage["backbone.embeddings"] = 0
        layer_to_pp_stage["model.norm"] = pp_size - 1
        layer_to_pp_stage["backbone.norm_f"] = pp_size - 1
        layer_to_pp_stage["lm_head"] = pp_size - 1
        return layer_to_pp_stage

    @torch.no_grad()
    def prepare_nccl_xfer_refit_info(
        self,
        train_parallelism,
        gen_parallelism,
        train_world_size,
        gen_world_size,
    ):
        """Prepare per-layer parameter metadata for nccl_xfer-based refit.

        The builder groups per-expert MoE params into backend-agnostic grouped
        HF entries (gate_proj/up_proj/down_proj); the gen backend maps those into
        its own fused layout (e.g., vLLM w13/w2) gen-side, so this train worker
        stays agnostic to any gen backend's MoE-fusion layout.
        """
        from nemo_rl.distributed.nccl_xfer_utils import (
            build_nccl_xfer_refit_info,
            is_nccl_xfer_param,
        )

        self.refit_param_info_mcore = self._calculate_refit_param_info()

        # Single pass over Bridge's stream: classify each param as major
        # (xferdtensor) or misc (packed_broadcast), preserve yield order so
        # producer/consumer agree on the packed-broadcast iteration.

        # Route QKV to the misc (packed_broadcast) path whenever
        # KV heads can't be cleanly 1/tp-sharded on EITHER side,
        # for the code simplicity.
        train_tp = train_parallelism.get("tp_size", 1)
        gen_tp = gen_parallelism.get("tp_size", 1)
        qkv_to_misc = self.megatron_bridge.transformer_config.num_query_groups < max(
            train_tp, gen_tp
        )

        state_dict_metadata = {}
        misc_meta = OrderedDict()
        _xfer_bytes = _bcast_bytes = 0  # full-tensor payload routed to each path

        # Handle tied LM head.
        from megatron.core.utils import unwrap_model

        _model0 = self.model[0] if isinstance(self.model, list) else self.model
        tie_lm_head = bool(
            getattr(unwrap_model(_model0), "share_embeddings_and_output_weights", False)
        )

        # Iterates all the params to construct the state_dict_metadata (xferdtensor path)
        # state_dict_metadata[hf_name] -> [shape, dtype]
        # At the same time, filter the params to the misc subset (packed_broadcast path).
        # misc_meta[hf_name] -> [shape, dtype]
        with _meta_tensor_alloc_context():
            for name, tensor in self._iter_params_with_optional_kv_scales():
                meta = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
                _nbytes = tensor.numel() * tensor.element_size()
                # Whitelist: only known-shardable params take the nccl-xfer path;
                # everything else (incl. unknown/new params) -> misc (packed_broadcast).
                if is_nccl_xfer_param(name, qkv_to_misc=qkv_to_misc):
                    if tie_lm_head and name == "lm_head.weight":
                        # Tied to embed_tokens (transferred); no standalone tensor
                        # to source, and the gen backend has none either.
                        continue
                    state_dict_metadata[name] = meta
                    _xfer_bytes += _nbytes
                else:
                    misc_meta[name] = meta
                    _bcast_bytes += _nbytes

        _gib = 1024**3
        _tot = _xfer_bytes + _bcast_bytes
        print(
            f"[xferd-payload] qkv_to_misc={qkv_to_misc} "
            f"nccl_xfer={_xfer_bytes / _gib:.2f}GiB "
            f"bcast_misc={_bcast_bytes / _gib:.2f}GiB "
            f"total={_tot / _gib:.2f}GiB "
            f"xfer_frac={_xfer_bytes / max(_tot, 1):.1%}",
            flush=True,
        )

        pp_size = train_parallelism.get("pp_size", 1)
        # Construct a dict[layer_name:str] -> pp_stage:int
        layer_to_pp_stage = (
            self._build_layer_to_pp_stage(pp_size) if pp_size > 1 else None
        )

        # The key metadata, which should shared with generation workers
        self.nccl_xfer_refit_info = build_nccl_xfer_refit_info(
            state_dict_metadata,
            train_parallelism,
            gen_parallelism,
            train_world_size,
            gen_world_size,
            layer_to_pp_stage=layer_to_pp_stage,
        )
        # _param_map is a dictionary that holds the pointer to the real local tensor
        # All of the non-misc params should be in the _param_map.
        # _param_map[hf_name:str] -> [local_tensor:Torch.Tensor]
        self._param_map = {
            name: tensor for name, tensor in self._iter_local_hf_param_shards()
        }
        # _expert_groups is a
        # Dict[(hf_layer_name.layer_id.expert:str, expert_param_suffix:str)]
        # -> List[local_tensor:Torch.Tensor]   (index-sorted _param_map views)
        # Later at the refit time, you just need to call torch.stack to
        # consolidate the expert tensors into one tensor.
        self._expert_groups = self._build_expert_groups()

        # Keep the misc_meta in the nccl_xfer_refit_info
        # misc_meta[hf_name] -> [shape, dtype]
        self.nccl_xfer_refit_info["misc_meta"] = misc_meta
        # Filter conversion_tasks to the misc subset.
        _misc_names = set(misc_meta.keys())

        def _task_is_misc(task) -> bool:
            # FP8 scale siblings carry the suffix on global_param_name and are
            # always misc (packed_broadcast).
            if task.global_param_name.endswith("_scale_inv"):
                return True
            # Compound mappings (QKV/GatedMLP) export homogeneous sub-params
            # (all nccl-xfer or all misc), so the first HF name is representative.
            hf = task.mapping.hf_param
            name = next(iter(hf.values())) if isinstance(hf, dict) else str(hf)
            return name in _misc_names

        self._misc_conversion_tasks = [
            task
            for task in self.refit_conversion_tasks
            if task is not None and _task_is_misc(task)
        ]

        return self.nccl_xfer_refit_info

    def _build_expert_groups(self):
        """Group this rank's local expert params into stack-ready views.

        Keyed by (prefix, proj_type) and resolved to ordered ``_param_map``
        views ready for ``torch.stack``.

        Megatron exposes each expert's projection as a separate param; this bins
        them so ``_group_experts`` can stack a layer's experts into one grouped
        HF tensor per projection.  Built once in ``prepare_nccl_xfer_refit_info``.

        ``_INDIVIDUAL_EXPERT_RE`` captures three fields from a name like
        ``model.layers.3.mlp.experts.17.gate_proj.weight``:
          * group 1 = prefix       -> ``"model.layers.3.mlp.experts"``
          * group 2 = expert index -> ``17``
          * group 3 = proj type    -> ``"gate_proj"``
        so the name keys into ``("model.layers.3.mlp.experts", "gate_proj")``.

        Returns ``{(prefix, proj): [tensor_0, tensor_1, ...]}`` — the per-expert
        ``_param_map`` views sorted by expert index.  Example — a layer with 2
        local experts (gated MoE) yields three keys:
          ``(".../experts", "gate_proj"): [view(expert 0), view(expert 1)]``
          ``(".../experts", "up_proj")  : [view(expert 0), view(expert 1)]``
          ``(".../experts", "down_proj"): [view(expert 0), view(expert 1)]``

        Resolving names → views here (rather than per refit in ``_group_experts``)
        costs nothing extra — ``_param_map`` already owns these views and they
        stay valid across refits (weights are updated in place; the name→view
        mapping is stable), so ``_group_experts`` only has to ``torch.stack``.
        The index sort matters: the views are stacked in this order, so expert 0
        must precede expert 1 to match the EP ``Shard(0)`` layout the gen side
        expects.
        """
        from nemo_rl.distributed.nccl_xfer_utils import _INDIVIDUAL_EXPERT_RE

        index_groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for name in self._param_map:
            # find all the expert params
            m = _INDIVIDUAL_EXPERT_RE.match(name)
            if m:
                # key = (group1 prefix, group3 proj_type); value (group2 idx, name)
                # example: ("model.layers.3.mlp.experts", "gate_proj") -> (0, name)
                index_groups.setdefault((m.group(1), m.group(3)), []).append(
                    (int(m.group(2)), name)
                )
        # Sort by expert index, then resolve each name to its _param_map view once.
        return {
            key: [self._param_map[n] for _, n in sorted(idx_names)]
            for key, idx_names in index_groups.items()
        }

    def _group_experts(self, proj, grouped_name, expert_groups):
        """Stack this rank's local experts for one projection into ``[E_local, ...]``.

        Using the pre-calculated expert_groups (=self._expert_groups) it is
        just calling torch.stack of all the local expert params.
        """
        prefix = grouped_name.rsplit(f".{proj}.weight", 1)[0]
        expert_tensors = expert_groups.get((prefix, proj))
        assert expert_tensors, (
            f"no local experts for {grouped_name!r} (proj={proj!r}); "
            "PP-filter / expert-group-metadata inconsistency"
        )
        return torch.stack(expert_tensors)

    def _get_src_local_tensor(self, param_info, expert_groups):
        """Get the TP/EP-local source tensor for one param."""
        name = param_info["name"]
        grouped_expert_proj = param_info.get("grouped_expert_proj")
        if grouped_expert_proj:
            # expert_groups has pre-calculated pointer to the local expert tensors
            # which was extracted from the _param_map in _build_expert_groups.
            return self._group_experts(grouped_expert_proj, name, expert_groups)
        return self._param_map.get(name)

    @torch.no_grad()
    def nccl_xfer_refit(self, kv_scales=None):
        """Transfer weights to generation workers via xferdtensor.

        Uses TP-local shards directly from Megatron parameters, bypassing
        the Bridge's PP broadcast + TP gather.  The modified xferdtensor
        reconstructs the full tensor from per-rank shards internally.

        ``kv_scales`` (FP8 KV cache): the per-layer k/v(/q) scales ride the misc
        packed-broadcast as plain scale tensors (the is_nccl_xfer_param whitelist
        excludes ``.k_scale``/``.v_scale``/``.q_scale`` -> misc); the gen side finalizes
        them via ``_maybe_process_fp8_kv_cache``.  No out-of-band channel needed.
        """
        from nemo_rl.distributed.xferdtensor import DTensorRef, xferdtensor

        # _param_map and _expert_groups are built once in
        # prepare_nccl_xfer_refit_info; weight values change but the
        # name → view mapping is stable across refits.
        use_per_stage = hasattr(self, "pp_comm_group")

        for layer_name in self.nccl_xfer_refit_info["layer_names"]:
            for param_info in self.nccl_xfer_refit_info["per_layer_params"][layer_name]:
                if use_per_stage:
                    if param_info.get("pp_stage", 0) != self.my_pp_stage:
                        continue
                    group = self.pp_comm_group
                else:
                    group = self.model_update_group

                # expert stacking is done in _get_src_local_tensor
                local = self._get_src_local_tensor(param_info, self._expert_groups)
                assert local is not None, f"no local tensor for {param_info['name']!r}"
                src_tensor = DTensorRef(
                    local_tensor=local, global_shape=param_info["global_shape"]
                )
                xferdtensor(
                    src_tensor,
                    param_info["src_mesh_info"],
                    param_info["src_placements"],
                    None,
                    param_info["dst_mesh_info"],
                    param_info["dst_placements"],
                    group,
                )
                # Drop refs to the per-iteration grouped MoE tensor so its CUDA
                # memory returns to the caching allocator
                del local, src_tensor

        self._broadcast_misc_params_packed(kv_scales=kv_scales)

    def _broadcast_misc_params_packed(self, kv_scales=None) -> None:
        """Broadcast misc params via the existing packed_broadcast machinery."""
        misc_meta = self.nccl_xfer_refit_info.get("misc_meta", {})
        if not misc_meta:
            return

        misc_iter = self._iter_params_with_optional_kv_scales(
            kv_scales=kv_scales,
            conversion_tasks=self._misc_conversion_tasks,
        )

        packed_broadcast_producer(
            iterator=misc_iter,
            group=self.model_update_group,
            src=0,
            post_iter_func=lambda x: x[1].contiguous(),
        )

    def prepare_for_lp_inference(self):
        self.model = self.move_model(self.model, "cuda", move_grads=False)
        self.model.eval()

        # offload grads to cpu
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and self.offload_optimizer_for_logprob
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.model = self.move_model(
            self.model, "cuda", move_grads=True, move_params=True
        )
        self.model.train()

        # Training expects optimizer state on CUDA. Keep this unconditional rather
        # than trying to mirror every path that may have offloaded it to CPU.
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cuda")

        if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
            torch.cuda.empty_cache()

    def finish_inference(self) -> None:
        """Offload model params to CPU after inference."""
        self.model = self.move_model(
            self.model, "cpu", move_params=True, move_grads=False
        )
        self.model.eval()

        gc.collect()
        torch.cuda.empty_cache()

    def _clear_fp8_caches(self):
        """Clear FP8 workspace caches and release fragmented GPU memory.

        The main memory issue in the train→offload→refit→generate cycle is CUDA
        allocator fragmentation, not leaked FP8 tensors. This method clears
        per-module _fp8_workspaces buffers (scratch memory references). The
        caller is responsible for running gc.collect() + empty_cache() once
        all references have been dropped.

        For anti-fragmentation, configure PYTORCH_CUDA_ALLOC_CONF in the recipe YAML:
        - "max_split_size_mb:512" — fast, prevents large-block splitting
        - "expandable_segments:True" — most effective but ~5x slower weight transfer
        """
        # 1. Clear Transformer Engine workspaces
        workspace_count = 0
        for module in self.model.modules():
            if hasattr(module, "_fp8_workspaces"):
                module._fp8_workspaces.clear()
                workspace_count += 1

        print(
            f"[_clear_fp8_caches] Cleared {workspace_count} workspace modules on rank {self.rank}"
        )

    @wrap_with_nvtx_name("megatron_policy_worker/offload_before_refit")
    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory before optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers

        # When True, clear Transformer Engine's per-module _fp8_workspaces scratch
        # buffers in offload_before_refit (before weight transfer to the inference
        # engine).
        if self.fp8_cfg and self.fp8_cfg.get("force_clear_fp8_caches", False):
            self._clear_fp8_caches()

        if self.cfg["megatron_cfg"].get("clear_memory_caches_before_refit", False):
            # Clear RotaryEmbedding's @lru_cache(maxsize=32). The cache accumulates one
            # entry per unique (max_seq_len, offset, packed_seq) seen, and each entry is
            # a GPU tensor (the concatenated sin/cos embedding). With training + logprob
            # runs at different sequence lengths, the cache fills quickly and the tensors
            # anchor large CUDA segments.
            try:
                from megatron.core.models.common.embeddings.rotary_pos_embedding import (
                    RotaryEmbedding,
                )

                RotaryEmbedding.forward.cache_clear()
            except Exception:
                pass

            # Clear MoE token dispatcher persistent routing tensors.
            #
            # MoETokenDispatcher is a plain Python class (NOT an nn.Module), so iterating
            # self.model.modules() never yields it. We must access it via the token_dispatcher
            # attribute on MoELayer nn.Module objects.
            #
            # When recompute_mlp=True and fp8=True,
            # transformer_layer._forward_mlp wraps self.mlp (the MoE layer) with te_checkpoint.
            # te_checkpoint._CheckpointFunction.backward recomputes the forward with
            # torch.enable_grad(), which causes dispatch_preprocess to store
            #   dispatcher.probs = routing_probs   (with grad_fn, under enable_grad)
            # This creates a reference cycle:
            #   _CheckpointFunctionBackward → ctx → ctx.run_function=mlp
            #   → mlp.token_dispatcher.probs → probs.grad_fn → ... → _CheckpointFunctionBackward
            #
            # Breaking this cycle by nulling dispatcher.probs frees BOTH:
            #   - the routing tensors
            #   - the te_checkpoint ctx saved tensors
            try:
                for module in self.model.modules():
                    if not hasattr(module, "token_dispatcher"):
                        continue
                    dispatcher = module.token_dispatcher
                    if dispatcher is None:
                        continue
                    for attr in (
                        "probs",  # AllToAll + AllGather
                        "routing_map",  # AllToAll
                        "reversed_local_input_permutation_mapping",  # AllToAll
                        "local_probs",  # AllGather
                        "local_map",  # AllGather
                    ):
                        if isinstance(getattr(dispatcher, attr, None), torch.Tensor):
                            setattr(dispatcher, attr, None)
            except Exception:
                pass

        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @wrap_with_nvtx_name("megatron_policy_worker/offload_after_refit")
    def offload_after_refit(self):
        """Offload as much as possible on the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model = self.move_model(self.model, "cpu")
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @torch.no_grad()
    def move_model(
        self,
        model: torch.nn.Module,
        device: str,
        move_params: bool = True,
        move_grads: bool = True,
    ) -> torch.nn.Module:
        # move all param and grad buffers to the device
        if isinstance(model, DistributedDataParallel):
            # DDP case
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
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
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
                    f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                )
        else:
            # Ordinary offload case
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
        # Iterate through the state dictionaries for each parameter group
        if isinstance(self.optimizer, ChainedOptimizer):
            optimizer_state = self.optimizer.state
        else:
            optimizer_state = self.optimizer._get_state()
        for _, state in optimizer_state.items():
            # Iterate through the state items (e.g., momentum, variance) for a parameter
            for k, v in state.items():
                # Check if the item is a tensor
                if torch.is_tensor(v):
                    # Move the tensor to device and update the state dictionary
                    if device == "cpu":
                        if v.is_cuda:
                            state[k] = v.to("cpu")
                    elif device == "cuda":
                        if not v.is_cuda:
                            state[k] = v.to("cuda")
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        **kwargs,
    ):
        """Save a training checkpoint.

        Args:
            weights_path: The specific directory path where the checkpoint will be saved.
            optimizer_path: If not None, optimizer and scheduler states are saved if they exist.
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed process group is not initialized. Cannot save checkpoint."
            )

        if self.mcore_state is None or self.model is None:
            raise RuntimeError(
                "Megatron core state or model is not initialized. Cannot save checkpoint."
            )

        original_save_path = self.mcore_state.cfg.checkpoint.save
        # save_dir = os.path.dirname(weights_path)
        release_name = os.path.basename(weights_path)

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

            # Ensure model is in eval mode for consistent saving, unless actively training
            # This is a common practice, though NeMo's save might handle this.
            # For safety, if not in training loop, setting to eval.
            is_training = self.model.training
            if not is_training:
                self.model.eval()

            if self.should_disable_forward_pre_hook:
                self.disable_forward_pre_hook()
            save_checkpoint(
                state=self.mcore_state,
                model=[self.model],
                optimizer=optimizer_to_save,
                opt_param_scheduler=scheduler_to_save,
                num_floating_point_operations_so_far=self.mcore_state.train_state.floating_point_operations_so_far,
                checkpointing_context=self.checkpointing_context,
            )
            print(f"Saved checkpoint to {weights_path}")
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=True,
                terminate=True,
            )
            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

            if not is_training:  # Restore training state if it was changed
                self.model.train()

        except Exception as e:
            print(f"Failed to save checkpoint to {weights_path}: {e}")
            raise
        finally:
            self.mcore_state.cfg.checkpoint.save = original_save_path

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        """Load a training checkpoint.

        Args:
            weights_path: The exact directory path from which to load the checkpoint.
            optimizer_path: If not None, attempts to load optimizer and scheduler states
                            if self.optimizer and self.scheduler are initialized.
        """
        raise NotImplementedError(
            "Loading checkpoints outside of the init function is not yet implemented for Megatron policy."
        )

    def check_tensor_parallel_attributes(self) -> dict[str, Any]:
        """Check tensor parallel attributes on model parameters.

        Returns:
            Dictionary containing information about tensor parallel parameters:
            - tp_params: List of parameter names that have tensor_model_parallel=True
            - non_tp_params: List of parameter names that have tensor_model_parallel=False
            - total_params: Total number of parameters checked
            - tp_size: Tensor parallel size from config
        """
        tp_params = []
        non_tp_params = []
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += 1
            tensor_model_parallel = getattr(param, "tensor_model_parallel", False)

            if tensor_model_parallel:
                tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "partition_dim": getattr(param, "partition_dim", None),
                        "partition_stride": getattr(param, "partition_stride", None),
                        "shape": list(param.shape),
                    }
                )
            else:
                non_tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "shape": list(param.shape),
                    }
                )

        return {
            "tp_params": tp_params,
            "non_tp_params": non_tp_params,
            "total_params": total_params,
            "tp_size": self.megatron_cfg.model.tensor_model_parallel_size,
        }

    @torch.no_grad()
    def calibrate_qkv_fp8_scales(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """One-shot calibration of Q/K/V activation scales (for FP8 KV cache).

        - Captures each layer's `query_key_value` output through forward hooks, splits Q/K/V, and computes percentile amax.
        - In parallel (DP/TP/PP) environments, first computes local percentiles, then takes max across all ranks for conservativeness.
        - By default only returns and saves K/V scales, optionally returns Q.

        Args:
            data: Representative sample batch for calibration, following get_logprobs input conventions.
            micro_batch_size: Micro batch size during calibration; if None, reuses logprob_batch_size.
            percentile: Percentile for amax (e.g. 99.9).
            margin: Margin factor, e.g. 1.05.
            save_path: If provided, rank0 will save results as JSON.
            include_q: Whether to also return Q scale (usually only K/V needed).

        Returns:
            { "format": "fp8", "percentile": float, "margin": float,
              "layers": { layer_name: {"k_scale": float, "v_scale": float[, "q_scale": float] } } }
        """
        from nemo_rl.models.generation.vllm.quantization.fp8_train_utils import (
            convert_calibration_to_vllm_format,
        )

        # Allow overriding FP8 max for Q, K, V via environment variables for ease of testing.
        # Defaults align with FP8 e4m3 max magnitude.
        # Use different defaults for Q, K, V to adapt to distribution diffefences
        def _get_env_float(name: str, default: float) -> float:
            try:
                val = os.getenv(name, None)
                return float(val) if val is not None and val != "" else default
            except Exception:
                return default

        FP8_MAX_Q = _get_env_float("FP8_MAX_Q", 448.0)
        FP8_MAX_K = _get_env_float("FP8_MAX_K", 448.0)
        FP8_MAX_V = _get_env_float("FP8_MAX_V", 448.0)

        self.model.eval()

        # Record local percentile amax for q/k/v of each layer
        layer_to_samples_q: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_k: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_v: dict[str, list[float]] = defaultdict(list)
        hook_handles = []

        def _extract_layer_key(module_name: str) -> str:
            # Expected format: "module.decoder.layers.<idx>.self_attention.query_key_value"
            m = re.search(r"module\.decoder\.layers\.(\d+)", module_name)
            if m is not None:
                return f"layer_{m.group(1)}"
            return module_name

        # Hook to capture q/k/v after q/k norm and RoPE
        def _pre_hook_builder_core_attention(module_name: str):
            layer_key = _extract_layer_key(module_name)

            def _pre_hook(module, inputs):
                args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                if len(args) == 1 and isinstance(args[0], (tuple, list)):
                    args = args[0]
                # Expected first 3 args to be q, k, v (typical signature for Megatron CoreAttention)
                q = args[0]
                k = args[1]
                v = args[2]
                if include_q:
                    layer_to_samples_q[layer_key].append(
                        float(torch.amax(torch.abs(q)).item())
                    )
                layer_to_samples_k[layer_key].append(
                    float(torch.amax(torch.abs(k)).item())
                )
                layer_to_samples_v[layer_key].append(
                    float(torch.amax(torch.abs(v)).item())
                )

            return _pre_hook

        matched_modules = []
        # Try to register forward_pre_hook on core_attention first
        for name, module in self.model.named_modules():
            if "self_attention.core_attention" in name:
                try:
                    handle = module.register_forward_pre_hook(
                        _pre_hook_builder_core_attention(name)
                    )
                    hook_handles.append(handle)
                    matched_modules.append((name, module.__class__.__name__, "pre"))
                except Exception as e:
                    print(
                        f"Error registering pre-hook for qkv scale calibration on {name}: {e}"
                        " Please check if the model is compatible with the current calibration logic. "
                        "The expected module name is 'self_attention.core_attention'."
                    )
                    raise

        # Run a forward pass to trigger hooks (reuse get_logprobs forward path)
        try:
            _ = self.get_logprobs(data=data, micro_batch_size=micro_batch_size)
        finally:
            for h in hook_handles:
                try:
                    h.remove()
                except Exception as e:
                    print(f"Error removing hook for qkv scale calibration: {e}")
                    raise

        # Compute local percentile amax
        def _percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            t = torch.tensor(sorted(values), device="cuda", dtype=torch.float32)
            rank = max(
                0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1))))
            )
            return float(t[rank].item())

        local_layer_to_pamax = {}
        for layer_key in set(
            list(layer_to_samples_k.keys())
            + list(layer_to_samples_v.keys())
            + (list(layer_to_samples_q.keys()) if include_q else [])
        ):
            entry = {}
            if include_q:
                entry["q_amax_p"] = _percentile(
                    layer_to_samples_q.get(layer_key, []), percentile
                )
            entry["k_amax_p"] = _percentile(
                layer_to_samples_k.get(layer_key, []), percentile
            )
            entry["v_amax_p"] = _percentile(
                layer_to_samples_v.get(layer_key, []), percentile
            )
            local_layer_to_pamax[layer_key] = entry

        # Merge across all ranks: take maximum of percentile amax (conservative approach)
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        gathered = [None for _ in range(world_size)] if world_size > 1 else None
        if world_size > 1:
            torch.distributed.all_gather_object(gathered, local_layer_to_pamax)
            merged = defaultdict(dict)
            for d in gathered:  # type: ignore
                if d is None:
                    continue
                for k, v in d.items():
                    dst = merged[k]
                    for kk, vv in v.items():
                        dst[kk] = max(dst.get(kk, 0.0), float(vv))
            layer_to_pamax = dict(merged)
        else:
            layer_to_pamax = local_layer_to_pamax

        # Compute scale (symmetric quantization): scale = pamax / fp8_max
        result_layers = {}
        for layer_key, vals in layer_to_pamax.items():
            out_entry = {}
            if include_q:
                q_scale = (vals.get("q_amax_p", 0.0) * margin) / FP8_MAX_Q
                out_entry["q_scale"] = float(q_scale)
            k_scale = (vals.get("k_amax_p", 0.0) * margin) / FP8_MAX_K
            v_scale = (vals.get("v_amax_p", 0.0) * margin) / FP8_MAX_V
            out_entry["k_scale"] = float(k_scale)
            out_entry["v_scale"] = float(v_scale)
            result_layers[layer_key] = out_entry

        vllm_format_scales = convert_calibration_to_vllm_format(result_layers)

        final_result = {
            "format": "fp8",
            "percentile": percentile,
            "margin": margin,
            "layers": vllm_format_scales,
        }

        # Sync results across all ranks (broadcast rank0's result)
        if world_size > 1:
            if torch.distributed.get_rank() == 0:
                obj_list = [final_result]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]
            else:
                obj_list = [None]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]  # type: ignore

        return final_result


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class MegatronPolicyWorker(MegatronPolicyWorkerImpl):
    pass
