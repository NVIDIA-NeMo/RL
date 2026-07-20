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
import copy
import gc
import logging
import os
import re
import time
import warnings
from collections import OrderedDict, defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import Any, Iterable, Iterator, Optional, TypeVar, cast

log = logging.getLogger(__name__)

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
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_model_config
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.logits_sampling_utils import TrainingSamplingParams
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.interfaces import GenerationDatumSpec
from nemo_rl.models.generation.megatron.megatron_worker import (
    MegatronGenerationMixin,
    MegatronGenerationRefitMixin,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.common import get_moe_metrics
from nemo_rl.models.megatron.data import (
    get_microbatch_iterator,
    process_global_batch,
)
from nemo_rl.models.megatron.pipeline_parallel import (
    broadcast_loss_metrics_from_last_stage,
    broadcast_obj_from_pp_rank,
    broadcast_tensors_from_last_stage,
)
from nemo_rl.models.megatron.router_replay import router_replay_enabled
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
    ReferenceLogprobOutputSpec,
)
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.utils.grad_norm import warn_if_inf_grad_norm
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.nvml import log_gpu_memory_diagnostics
from nemo_rl.utils.packed_tensor import packed_broadcast_producer
from nemo_rl.utils.r3_trace import maybe_r3_trace_stage
from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.nccl_reshard_utils import (
    HFToLocalParamMap,
    LocalParamSpec,
    RefitCtx,
)

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def _should_use_router_replay(
    *,
    enabled: bool,
    data: BatchedDataDict[Any],
    stage: str,
    require: bool,
) -> bool:
    if not enabled or not require:
        return False
    if "routed_experts" in data:
        return True
    raise RuntimeError(
        "policy.router_replay.enabled=true requires routed_experts for "
        f"{stage}, but the fetched batch does not contain that field. This "
        "usually means the TQ schema, field selection, or rollout write path "
        "stopped carrying routed_experts. Reference-logprob intentionally skips "
        "routed_experts; prev-logprob and train must not."
    )


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
    MegatronGenerationMixin,
    MegatronGenerationRefitMixin,
    TQWorkerMixin,
    AbstractPolicyWorker,
    ColocatablePolicyInterface,
):
    # Holds the split-API train-step state between begin/finish or
    # begin/abort; None when no step is open. Declared at class level so
    # ``self._train_step_state = None`` after finish/abort type-checks.
    _train_step_state: Optional[dict[str, Any]] = None
    _remote_sparse_refit: Any = None

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

    @staticmethod
    def configure_worker(
        num_gpus: int | float,
        bundle_indices: Optional[tuple[int, list[int]]] = None,
        num_gpus_per_node: Optional[int] = None,
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], dict[str, Any]]:
        """Worker-controlled Ray actor configuration.

        Ensures that communication via NVLS functions correctly.

        Args:
            num_gpus: Original GPU allocation for this worker based on the placement group
            bundle_indices: Tuple of (node_idx, local_bundle_indices) for this server
            num_gpus_per_node: Per-node GPU count (unused here; part of the shared
                configure_worker contract).

        Returns:
            tuple with complete worker configuration:
              - 'resources': Resource allocation (e.g., num_gpus)
              - 'env_vars': Environment variables for this worker
              - 'init_kwargs': Parameters to pass to __init__ of the worker
              - 'runtime_env': Additional runtime_env options (e.g., nsight config)
        """
        del bundle_indices  # one GPU per worker; no per-bundle seeding needed
        del num_gpus_per_node  # not needed; one GPU per worker
        resources: dict[str, Any] = {"num_gpus": num_gpus}
        env_vars: dict[str, str] = {"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"}
        init_kwargs: dict[str, Any] = {}
        return resources, env_vars, init_kwargs, {}

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
        # NVML-based and guarded on torch.cuda.is_initialized(), so this does
        # not initialize a CUDA context ahead of the set_device below.
        log_gpu_memory_diagnostics(
            label="init_start", worker_type="MegatronPolicyWorker"
        )

        # Must be the first CUDA-touching call in this process.
        # With `RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1` (set by `configure_worker()`),
        gpu_ids = ray.get_gpu_ids()
        local_rank = int(gpu_ids[0])
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(local_rank)

        # Apply patch from https://github.com/NVIDIA/TransformerEngine/pull/2286/files
        apply_transformer_engine_patch()

        from nemo_rl.distributed.numa_utils import bind_to_gpu_numa

        # local_rank (== ray.get_gpu_ids()[0]) is the physical GPU index that
        # keys the affinity file. Pass it explicitly: CUDA_VISIBLE_DEVICES lists
        # all node devices here (RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1,
        # set by configure_worker), so it can't identify this worker's GPU.
        bind_to_gpu_numa(local_rank)

        self.cfg = config
        self._router_replay_enabled = router_replay_enabled(config)

        # Set rank for non-collocated to check which ranks to broadcast from
        self.rank = get_rank_safe()
        self.timer = Timer(context={"worker": "megatron_policy", "rank": self.rank})

        # Step 1: Setup distributed
        setup_distributed()
        log_gpu_memory_diagnostics(
            label="after_nccl_init", worker_type="MegatronPolicyWorker"
        )

        # Defensive assert to ensure to ensure `local_rank` setter worked correctly.
        assert torch.cuda.current_device() == local_rank, (
            f"device drift after setup_distributed: current_device="
            f"{torch.cuda.current_device()}, LOCAL_RANK={local_rank}."
        )

        # Step 2: Validate and setup model paths
        hf_model_name, pretrained_path, pt_checkpoint_exists = validate_model_paths(
            config
        )
        # Handle model import if needed. Subclasses (e.g. ModelOpt quant
        # worker) may set ``_model_import_post_wrap_hook`` and
        # layer-spec hooks on ``self`` before calling
        # super().__init__() to inject quantization hooks into HF->Megatron
        # import.
        handle_model_import(
            config,
            hf_model_name,
            pretrained_path,
            pt_checkpoint_exists,
            model_post_wrap_hook=getattr(self, "_model_import_post_wrap_hook", None),
            transformer_layer_spec=getattr(self, "_transformer_layer_spec", None),
            mamba_stack_spec=getattr(self, "_mamba_stack_spec", None),
        )
        log_gpu_memory_diagnostics(
            label="after_hf_import", worker_type="MegatronPolicyWorker"
        )

        # Store tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Step 3: Setup model configuration
        # Training workers cannot use inference_optimized transformer spec.
        if init_optimizer:
            assert (
                config["megatron_cfg"].get("transformer_impl") != "inference_optimized"
            ), (
                "transformer_impl=inference_optimized must not be set on training workers. "
                "Use policy.generation.mcore_generation_config.transformer_impl=inference_optimized instead."
            )
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
        self.final_padded_vocab_size = runtime_config.final_padded_vocab_size
        self.sampling_params = runtime_config.sampling_params

        self.defer_fp32_logits = self.cfg["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (runtime_config.model_cfg.fp16 or runtime_config.model_cfg.bf16)

        # Store FP8 config for later use
        self.fp8_cfg = config["megatron_cfg"].get("fp8_cfg", None)

        # Full-iteration CUDA graphs cannot be interrupted, so disable the
        # NaN-in-loss check that would otherwise require breaking out of the graph.
        if self.megatron_cfg.model.cuda_graph_impl == "full_iteration":
            warnings.warn(
                "Disabling check_for_nan_in_loss: full-iteration CUDA graph cannot be interrupted."
            )
            self.megatron_cfg.rerun_state_machine.check_for_nan_in_loss = False

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

        self._init_inference_engine_state()

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
        self.timer.start("train")
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

        torch.distributed.barrier()  # pragma: no cover
        torch.cuda.synchronize()  # pragma: no cover
        _train_t0 = time.perf_counter()  # pragma: no cover

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

                # Pre-compute the MTP loss mask, only when MTP is enabled, so
                # process_microbatch can pack it.
                model_config = self._get_model_config()
                mtp_num_layers = getattr(model_config, "mtp_num_layers", None)
                mtp_enabled = mtp_num_layers is not None and mtp_num_layers > 0
                if mtp_enabled and "token_mask" in batch and "sample_mask" in batch:
                    mtp_loss_mask = batch["token_mask"] * batch[
                        "sample_mask"
                    ].unsqueeze(-1)
                    batch["mtp_loss_mask"] = mtp_loss_mask

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

                    # Set moe_grad_scale_func for MoE aux-loss gradient scaling.
                    # With calculate_per_token_loss=True, the router pre-multiplies
                    # the aux loss by (num_local_tokens * tp_cp_group.size()), and
                    # MoEAuxLossAutoScaler applies loss_scale to the gradient. Setting
                    # loss_scale = 1/global_valid_toks (G = global valid token count)
                    # normalizes the aux gradient consistently with the main per-token
                    # SFT loss:
                    #   (1/G) * N_local * tp_cp_size * aux_grad -> DDP SUM -> aux_grad / G
                    self._set_moe_grad_scale_func(  # pragma: no cover
                        self._compute_moe_grad_scale(global_valid_toks)
                    )
                    # Set mtp_grad_scale_func for MTP loss scaling (scales by valid tokens)
                    mtp_scale = 1.0 / global_valid_toks.clamp(min=1).float()
                    self._set_mtp_grad_scale_func(lambda: mtp_scale)

                    # Forward pass.
                    draft_enabled = "draft" in self.cfg and self.cfg["draft"]["enabled"]
                    use_router_replay = _should_use_router_replay(
                        enabled=self._router_replay_enabled,
                        data=batch,
                        stage="train",
                        require=True,
                    )
                    with maybe_r3_trace_stage("train", enabled=use_router_replay):
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
                            use_fused_linear_logprobs=self.cfg["megatron_cfg"].get(
                                "use_fused_linear_logprobs", False
                            ),
                            use_router_replay=use_router_replay,
                            router_replay_train=not eval_mode,
                        )

                # Clear mtp_grad_scale_func after the forward-backward pass so
                # it doesn't get serialized in the run_config.yaml when saving
                self._set_mtp_grad_scale_func(None)

                # Clear moe_grad_scale_func after the forward-backward pass
                self._set_moe_grad_scale_func(None)  # pragma: no cover

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters.
                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )
                    # Megatron-LM PR #4116 replaced the optimizer.mtp_grad_norm attribute
                    # with a per-group dict populated during gradient clipping. Value is
                    # None when clip_grad == 0 or this rank owns no MTP-tagged params
                    # (MTP params are tagged only when mtp_detach_heads=True, on the last
                    # pipeline stage). grad_norms_by_group always exists after step().
                    mtp_grad_norm = self.optimizer.grad_norms_by_group.get("mtp")
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (True, 0.0, 0.0)
                    mtp_grad_norm = None

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
                # Max-reduce across the model-parallel group so every rank (including
                # non-last-PP-stage ranks, where it is None) has the MTP grad norm.
                mtp_grad_norm = reduce_max_stat_across_model_parallel_group(
                    mtp_grad_norm, mp_group=pg_collection.mp
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

                warn_if_inf_grad_norm(grad_norm)

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

        torch.distributed.barrier()  # pragma: no cover
        torch.cuda.synchronize()  # pragma: no cover
        metrics_train_elapsed = time.perf_counter() - _train_t0  # pragma: no cover

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
            "train_elapsed_seconds": metrics_train_elapsed,  # pragma: no cover
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
        # Collect MTP metrics (kept out of train()'s body so cloudpickle does not
        # pull an unpicklable torch ConfigModuleInstance into the worker actor).
        self._collect_mtp_metrics(metrics, total_num_microbatches, mtp_grad_norm)

        # Skip FLOPs estimation when sequence packing is enabled: gbs counts original
        # samples but each packed sequence spans max_total_sequence_length tokens,
        # so flops_per_sample * gbs would overcount by the packing factor.
        if not self.cfg.get("sequence_packing", {}).get("enabled", False):
            try:  # pragma: no cover
                from megatron.bridge.training.utils import flop_utils as _mb_flop_utils

                # cfg.model.seq_length is set from max_position_embeddings (model max context)
                # via CONFIG_MAPPING, not from max_total_sequence_length. Override it with the
                # actual training sequence length so FLOPs are not inflated.
                _orig_seq = self.mcore_state.cfg.model.seq_length
                self.mcore_state.cfg.model.seq_length = self.cfg[
                    "max_total_sequence_length"
                ]
                try:
                    flops_per_sample = _mb_flop_utils.num_floating_point_operations(
                        self.mcore_state.cfg, batch_size=1
                    )
                finally:
                    self.mcore_state.cfg.model.seq_length = _orig_seq

                metrics["total_flops"] = flops_per_sample * gbs * num_global_batches
                metrics["num_ranks"] = torch.distributed.get_world_size()
            except Exception as e:
                warnings.warn(f"Failed to compute FLOPs for MFU reporting: {e}")
        self.timer.stop("train")
        return metrics

    def _compute_moe_grad_scale(self, global_valid_toks):
        """Build a moe_grad_scale_func that normalizes the aux-loss gradient.

        Returns a callable yielding loss_scale = 1/global_valid_toks (clamped to
        avoid division by zero) so the MoE aux gradient is normalized consistently
        with the main per-token SFT loss. See the call site in train() for the
        full derivation.
        """
        moe_scale = 1.0 / global_valid_toks.clamp(min=1).float()
        return lambda: moe_scale

    def _set_moe_grad_scale_func(self, func):
        """Set moe_grad_scale_func on the model config for MOE aux loss scaling."""
        config = self._get_model_config()
        if config is not None:
            config.moe_grad_scale_func = func

    @wrap_with_nvtx_name("megatron_policy_worker/get_reference_policy_logprobs")
    def get_reference_policy_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
    ) -> BatchedDataDict[ReferenceLogprobOutputSpec]:
        with self.use_reference_model():
            reference_logprobs = self.get_logprobs(
                data=data,
                micro_batch_size=micro_batch_size,
                require_router_replay=False,
            )

        return_data = BatchedDataDict[ReferenceLogprobOutputSpec]()
        return_data["reference_logprobs"] = reference_logprobs["logprobs"].cpu()
        return return_data

    # ── split-API train-step state machine (SingleController async path) ──
    #
    # SC drives one ``begin / train_microbatch×N / finish`` cycle per
    # optimizer step. The worker exposes:
    #   begin_train_step      — open the step (zero grads, null mcore sync hooks)
    #   train_microbatch      — one DP slice of fwd/bwd, grads accumulate locally
    #   finish_train_step     — all_reduce + opt.step + scheduler.step
    #   abort_train_step      — drop partial state (no opt.step)
    #
    # Key mcore-specific details:
    #
    # 1. mcore DDP accumulates ``param.main_grad`` per backward and dispatches
    #    a cross-DP reduce when ``is_last_microbatch=True`` (one per
    #    ``forward_backward_func`` call). Naively chaining multiple
    #    ``forward_backward_func`` calls between ``optimizer.step()`` would
    #    over-count: each call's terminal reduce sums an already-reduced
    #    bucket again. We wrap every call in ``self.model.no_sync()`` so
    #    hooks accumulate locally only; one explicit ``start_grad_sync`` +
    #    ``finish_grad_sync`` at finish does the single true reduce.
    # 2. PP>1: the pipeline scheduler invokes ``config.grad_sync_func``
    #    directly on last-microbatch boundaries — this bypasses the
    #    ``no_sync`` gate. We null it for the duration of the step and
    #    restore at finish/abort.
    # 3. Grad clip is bundled inside ``MegatronOptimizer.step()``; the 1/N
    #    rescale via ``self.model.scale_gradients(1/N)`` must run before
    #    ``optimizer.step()`` so the clip operates on the rescaled grad.
    # 4. With ``calculate_per_token_loss=True`` + ``average_in_collective=
    #    False``, mcore's DDP sums (does not average) grads across DP, so
    #    no FSDP-style ``loss *= dp_size*cp_size`` cancellation is needed
    #    per microbatch.

    def _split_step_state_init(
        self,
        loss_fn: LossFunction,
        gbs: Optional[int],
        mbs: Optional[int],
    ) -> dict[str, Any]:
        from nemo_rl.algorithms.loss.interfaces import LossType

        # Losses advertise per-metric denominators (see MetricNormalizer in
        # the loss interfaces); guard the type so non-advertising losses and
        # auto-attributed test doubles fall back to gradient normalization
        # for every metric.
        metric_normalizations = getattr(loss_fn, "metric_normalizations", None)
        if not isinstance(metric_normalizations, dict):
            metric_normalizations = {}

        return {
            "loss_fn": loss_fn,
            "loss_type": getattr(loss_fn, "loss_type", LossType.TOKEN_LEVEL),
            "metric_normalizations": metric_normalizations,
            "gbs": gbs or self.cfg["train_global_batch_size"],
            "mbs": mbs or self.cfg["train_micro_batch_size"],
            "local_valid_seqs": torch.zeros((), dtype=torch.float64, device="cuda"),
            "local_valid_toks": torch.zeros((), dtype=torch.float64, device="cuda"),
            "all_mb_metrics": [],
            "mb_losses": [],
            "total_num_microbatches": 0,
            # Saved across the step so we can restore at finish/abort.
            "saved_grad_sync_func": None,
            "saved_no_sync_func": None,
        }

    def _assert_step_open(self) -> dict[str, Any]:
        state = getattr(self, "_train_step_state", None)
        if state is None:
            raise RuntimeError(
                "no train step open; begin_train_step must be called first"
            )
        return state

    def _restore_saved_grad_sync_func(self, state: dict[str, Any]) -> None:
        """Restore the mcore hooks nulled in ``begin_train_step``.

        Restores both ``grad_sync_func`` and ``no_sync_func`` from the
        saved values on the open-step state. Idempotent on those values;
        safe to call from the happy-path finish/abort or from a try/except
        cleanup in train_microbatch / finish_train_step when those raise
        mid-body. See begin_train_step for why ``.config`` is read via
        getattr-by-string.
        """
        model_config = getattr(self.model, "config", None)
        if model_config is not None:
            model_config.grad_sync_func = state.get("saved_grad_sync_func")
            model_config.no_sync_func = state.get("saved_no_sync_func")

    @wrap_with_nvtx_name("megatron_policy_worker/begin_train_step")
    def begin_train_step(
        self,
        loss_fn: LossFunction,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> None:
        existing = getattr(self, "_train_step_state", None)
        if existing is not None:
            raise RuntimeError(
                "a train step is already open; "
                "call finish_train_step or abort_train_step before begin"
            )
        # Match sync train() inference-state reset (line 332-340).
        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None
        for module in self.model.modules():
            if hasattr(module, "reset_inference_cache"):
                module.reset_inference_cache()
            if hasattr(module, "_inference_key_value_memory"):
                module._inference_key_value_memory = None

        self.model.train()
        self.model.zero_grad_buffer()
        self.optimizer.zero_grad()

        state = self._split_step_state_init(loss_fn=loss_fn, gbs=gbs, mbs=mbs)

        # Null both mcore hooks that would fire a mid-step DP reduce:
        #   grad_sync_func — PP scheduler's direct call on last-MB boundaries
        #                    (PP>1 path).
        #   no_sync_func   — ``forward_backward_no_pipelining`` (PP=1, the
        #                    common case) wraps inner microbatches in this
        #                    context and runs the LAST microbatch OUTSIDE
        #                    of it. Without overriding, ``register_grad_ready``
        #                    leaks per-param counts past the outer
        #                    ``model.no_sync()`` we apply in train_microbatch,
        #                    triggering an assertion on the next begin/microbatch
        #                    pair (typically step 2).
        # Save both so finish/abort restores them.
        # Read "config" via getattr-by-string so the token stays out of
        # begin_train_step.__code__.co_names; otherwise cloudpickle matches
        # torch.distributed.config (a non-pickleable ConfigModuleInstance).
        model_config = getattr(self.model, "config", None)
        if model_config is not None:
            state["saved_grad_sync_func"] = getattr(
                model_config, "grad_sync_func", None
            )
            state["saved_no_sync_func"] = getattr(model_config, "no_sync_func", None)
            model_config.grad_sync_func = None
            model_config.no_sync_func = nullcontext
        else:
            state["saved_grad_sync_func"] = None
            state["saved_no_sync_func"] = None

        self._train_step_state = state

    @wrap_with_nvtx_name("megatron_policy_worker/train_microbatch")
    def train_microbatch(
        self,
        data: BatchedDataDict[Any],
    ) -> None:
        """One DP slice of data → one ``forward_backward_func`` invocation.

        Wrapped in ``self.model.no_sync()`` so the mcore DDP hooks
        accumulate ``param.main_grad`` locally on each rank without
        dispatching a per-call DP reduce. The single true reduce is done
        explicitly in ``finish_train_step``. Returns nothing: gradients
        land in ``param.main_grad`` and per-microbatch metrics accumulate
        in the open-step state until ``finish_train_step`` surfaces them.
        """
        state = self._assert_step_open()
        try:
            self._train_microbatch_body(state, data)
        except Exception:
            # The body left ``grad_sync_func`` nulled when begin_train_step
            # opened the step. If we propagate without restoring, future
            # steps run with the PP scheduler bypass disabled. Restore here;
            # the caller is still expected to invoke abort_train_step
            # (idempotent on the saved value) to drop ``_train_step_state``.
            try:
                self._restore_saved_grad_sync_func(state)
            except Exception:
                log.exception(
                    "failed to restore grad_sync_func after train_microbatch error"
                )
            raise

    def _train_microbatch_body(
        self,
        state: dict[str, Any],
        data: BatchedDataDict[Any],
    ) -> None:
        loss_fn = state["loss_fn"]

        # Accumulate local mask sums for the finish-time all_reduce.
        # Inlined from process_global_batch (data.py:319-332) — we can't
        # call process_global_batch directly because it eagerly all_reduces
        # the local sums, which is exactly what we're trying to defer.
        assert "sample_mask" in data, "sample_mask required on microbatch data"
        sample_mask = data["sample_mask"]
        call_local_seqs = torch.sum(sample_mask).to(torch.float64)
        if "token_mask" in data:
            token_mask = data["token_mask"]
            call_local_toks = torch.sum(
                token_mask[:, 1:] * sample_mask.unsqueeze(-1)
            ).to(torch.float64)
        else:
            call_local_toks = call_local_seqs * data["input_ids"].shape[1]

        state["local_valid_seqs"] = state["local_valid_seqs"] + call_local_seqs
        state["local_valid_toks"] = state["local_valid_toks"] + call_local_toks

        # Build the per-call iterator. Each ``train_microbatches_from_meta``
        # call carries one DP slice; the iterator subdivides into pipeline
        # microbatches.
        (
            data_iterator,
            num_microbatches,
            micro_batch_size,
            seq_length,
            padded_seq_length,
        ) = get_microbatch_iterator(
            data,
            self.cfg,
            state["mbs"],
            straggler_timer=self.mcore_state.straggler_timer,
        )
        state["total_num_microbatches"] += int(num_microbatches)

        loss_post_processor = LossPostProcessor(
            loss_fn=loss_fn,
            cfg=self.cfg,
            num_microbatches=num_microbatches,
            sampling_params=self.sampling_params,
            draft_model=self.draft_model,
        )

        # Placeholder N=1: loss returns un-normalized sums. ``backward``
        # deposits raw ``d(sum)/dθ`` into ``param.main_grad`` via the DDP
        # hooks. The 1/N rescale happens once at finish.
        placeholder_n = torch.tensor(1.0, device="cuda")

        draft_enabled = "draft" in self.cfg and self.cfg["draft"]["enabled"]
        use_router_replay = _should_use_router_replay(
            enabled=self._router_replay_enabled,
            data=data,
            stage="train",
            require=True,
        )

        # The critical wrap: hooks fire (accumulate main_grad) but the
        # per-call reduce dispatch is gated off.
        with (
            maybe_r3_trace_stage("train", enabled=use_router_replay),
            self.model.no_sync(),
        ):
            rerun_state_machine = get_rerun_state_machine()
            while rerun_state_machine.should_run_forward_backward(data_iterator):
                losses_reduced = megatron_forward_backward(
                    model=self.model,
                    data_iterator=data_iterator,
                    num_microbatches=num_microbatches,
                    seq_length=padded_seq_length,
                    mbs=micro_batch_size,
                    post_processing_fn=loss_post_processor,
                    forward_only=False,
                    defer_fp32_logits=self.defer_fp32_logits,
                    global_valid_seqs=placeholder_n,
                    global_valid_toks=placeholder_n,
                    sampling_params=self.sampling_params,
                    straggler_timer=self.mcore_state.straggler_timer,
                    draft_model=self.draft_model,
                    enable_hidden_capture=draft_enabled,
                    use_fused_linear_logprobs=self.cfg["megatron_cfg"].get(
                        "use_fused_linear_logprobs", False
                    ),
                    use_router_replay=use_router_replay,
                    router_replay_train=True,
                )

        if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
            torch.cuda.empty_cache()

        # Collect per-mb metrics from the last PP stage; broadcast to all
        # PP ranks so non-last-stage ranks have something to all_reduce
        # against at finish. Metrics carry the N=1 placeholder for now —
        # ``finish_train_step`` rescales by the true 1/N.
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            mb_metrics_collected = []
            for x in losses_reduced:
                mb_metrics_collected.append(dict(x))
        else:
            mb_metrics_collected = None

        mb_metrics_collected = broadcast_loss_metrics_from_last_stage(
            mb_metrics_collected
        )

        for m in mb_metrics_collected:
            state["all_mb_metrics"].append(m)
            # ``loss`` key is the un-normalized per-mb scalar; collect for
            # the global_loss aggregation at finish.
            if "loss" in m:
                state["mb_losses"].append(m["loss"])

    @wrap_with_nvtx_name("megatron_policy_worker/finish_train_step")
    def finish_train_step(self) -> dict[str, Any]:
        state = self._assert_step_open()
        try:
            return self._finish_train_step_body(state)
        except Exception:
            # Mid-finish failure: state machine is in a partial state and
            # ``grad_sync_func`` may still be nulled (or restored, depending
            # on how far the body got). Restore unconditionally so future
            # steps run with the right config. Leave ``_train_step_state``
            # for the caller's abort_train_step to clear.
            try:
                self._restore_saved_grad_sync_func(state)
            except Exception:
                log.exception(
                    "failed to restore grad_sync_func after finish_train_step error"
                )
            raise

    def _finish_train_step_body(self, state: dict[str, Any]) -> dict[str, Any]:
        from nemo_rl.algorithms.loss.interfaces import LossType

        # All-reduce accumulated mask sums across DP to recover true N.
        to_reduce = torch.stack(
            [state["local_valid_seqs"], state["local_valid_toks"]]
        ).to(torch.float64)
        torch.distributed.all_reduce(
            to_reduce, group=parallel_state.get_data_parallel_group()
        )
        global_valid_seqs = to_reduce[0]
        global_valid_toks = to_reduce[1]

        if state["loss_type"] == LossType.TOKEN_LEVEL:
            n_true = global_valid_toks
        else:
            n_true = global_valid_seqs
        n_safe = n_true if n_true.item() > 0 else torch.tensor(1.0, device="cuda")
        inv_n = float((1.0 / n_safe).item())

        # Rescale all locally-accumulated gradients by 1/N. The reduce
        # below sees the rescaled grads; for all_reduce the result is the
        # global mean grad; for reduce_scatter (dist-opt) it's the shard.
        # Either way, opt.step sees the right-normalized gradient.
        self.model.scale_gradients(inv_n)

        # Cross-DP grad reduce. Megatron-core's BucketGroup.finish_grad_sync,
        # when overlap_grad_reduce=False, internally dispatches the synchronous
        # collective via start_grad_sync(force_all_reduce=...). So calling
        # both unconditionally double-reduces the grads (scales by world_size).
        # Mirror the contract: only fire start_grad_sync ourselves when the
        # overlap path needs it; finish_grad_sync handles the rest.
        if self.cfg["megatron_cfg"]["distributed_data_parallel_config"][
            "overlap_grad_reduce"
        ]:
            self.model.start_grad_sync()
        self.model.finish_grad_sync()

        # opt.step clips internally (clip_grad config); operates on the
        # already-rescaled grad. Returns (success, grad_norm, num_zeros).
        update_successful, grad_norm, num_zeros_in_grad = self.optimizer.step()

        pg_collection = get_pg_collection(self.model)
        update_successful = logical_and_across_model_parallel_group(
            update_successful, mp_group=pg_collection.mp
        )
        grad_norm = reduce_max_stat_across_model_parallel_group(
            grad_norm, mp_group=pg_collection.mp
        )
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(
            num_zeros_in_grad, mp_group=pg_collection.mp
        )

        if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
            torch.cuda.empty_cache()

        # Restore grad_sync_func before scheduler.step / further state.
        self._restore_saved_grad_sync_func(state)

        # Capture lr/wd BEFORE scheduler.step so the per-mb metrics carry
        # the value of THIS step, not the next one. (terrykong, #2683:832).
        curr_lr = self.scheduler.get_lr(self.optimizer.param_groups[0])
        curr_wd = self.scheduler.get_wd()

        # Scheduler increment matches sync path's ``increment=gbs``.
        self.scheduler.step(increment=state["gbs"])

        # Per-mb metrics were computed with global_valid_*=1 (raw sums);
        # rescale to match what the sync path produces. Different metrics
        # use different denominators — the loss advertises them per metric
        # (see MetricNormalizer in the loss interfaces). masked_mean is
        # linear in 1/N so per-metric scalar multiplies recover the right
        # normalized values.
        n_toks_safe = (
            global_valid_toks
            if global_valid_toks.item() > 0
            else torch.tensor(1.0, device="cuda")
        )
        n_seqs_safe = (
            global_valid_seqs
            if global_valid_seqs.item() > 0
            else torch.tensor(1.0, device="cuda")
        )
        inv_toks = float((1.0 / n_toks_safe).item())
        inv_seqs = float((1.0 / n_seqs_safe).item())

        from nemo_rl.algorithms.loss.interfaces import MetricNormalizer

        metric_normalizations: dict[str, Any] = state["metric_normalizations"]

        def _scale_metric(name: str, value: Any) -> Any:
            """Apply the loss-advertised per-metric denominator.

            Metrics the loss did not advertise fall back to the gradient
            normalization (the ``loss_type`` denominator).
            """
            kind = metric_normalizations.get(name)
            if kind is MetricNormalizer.NONE:
                return value
            if kind is MetricNormalizer.TOKENS:
                scale = inv_toks
            elif kind is MetricNormalizer.SEQUENCES:
                scale = inv_seqs
            else:  # not advertised → same as gradient normalization
                scale = inv_n
            return (
                value.detach() * scale
                if isinstance(value, torch.Tensor)
                else value * scale
            )

        rescaled_metrics: list[dict[str, Any]] = []
        # curr_lr/curr_wd captured pre-scheduler.step above.
        global_valid_seqs_f = float(global_valid_seqs.item())
        global_valid_toks_f = float(global_valid_toks.item())

        for m in state["all_mb_metrics"]:
            out: dict[str, Any] = {}
            for k, v in m.items():
                if "_min" in k or "_max" in k:
                    out[k] = v
                else:
                    out[k] = _scale_metric(k, v)
            out["lr"] = curr_lr
            out["wd"] = curr_wd
            out["global_valid_seqs"] = global_valid_seqs_f
            out["global_valid_toks"] = global_valid_toks_f
            rescaled_metrics.append(out)

        # Scale per-mb losses by 1/N and reduce per-call sums.
        scaled_losses = [lv * inv_n for lv in state["mb_losses"]]
        losses_to_aggregate = [torch.tensor(scaled_losses).sum().item()]

        mb_metrics, global_loss = aggregate_training_statistics(
            all_mb_metrics=rescaled_metrics,
            losses=losses_to_aggregate,
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

        # MoE aux-loss metrics: same convention as sync train() — scale
        # by the total pipeline-microbatch count accumulated across all
        # train_microbatch calls.
        model_config = getattr(self.model, "config", None)
        num_moe_experts = getattr(model_config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, state["total_num_microbatches"])
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"]["moe_per_layer_logging"],
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics

        self._train_step_state = None
        return metrics

    @wrap_with_nvtx_name("megatron_policy_worker/abort_train_step")
    def abort_train_step(self) -> None:
        state = getattr(self, "_train_step_state", None)
        if state is None:
            return
        # Restore grad_sync_func first so the model is back to a normal
        # state before zero_grad_buffer touches anything.
        self._restore_saved_grad_sync_func(state)
        self.model.zero_grad_buffer()
        self.optimizer.zero_grad()
        self._train_step_state = None

    @wrap_with_nvtx_name("megatron_policy_worker/get_logprobs")
    def get_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        require_router_replay: bool = True,
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
        self.timer.start("get_logprobs")
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

        use_fused_linear_logprobs = self.cfg["megatron_cfg"].get(
            "use_fused_linear_logprobs", False
        )
        logprobs_post_processor = LogprobsPostProcessor(
            cfg=self.cfg,
            sampling_params=self.sampling_params,
            use_fused_linear_logprobs=use_fused_linear_logprobs,
        )
        use_router_replay = _should_use_router_replay(
            enabled=self._router_replay_enabled,
            data=data,
            stage="prev-logprob",
            require=require_router_replay,
        )

        with maybe_r3_trace_stage("prev-logprob", enabled=use_router_replay):
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
                use_fused_linear_logprobs=use_fused_linear_logprobs,
                use_router_replay=use_router_replay,
                router_replay_train=False,
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
        self.timer.stop("get_logprobs")
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

            # Swap reference state into self.model. Use _apply_state_dict_to_model
            # (rather than load_state_dict) so FP8 _extra_state with mismatched shape
            # is routed through set_extra_state() correctly.
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

    def _collect_mtp_metrics(
        self,
        metrics: dict[str, Any],
        total_num_microbatches: int,
        mtp_grad_norm: Optional[float],
    ) -> None:
        """Add Multi-Token Prediction metrics to ``metrics`` when MTP is enabled.

        get_mtp_metrics is imported lazily (not a module global) so cloudpickle
        does not pull an unpicklable torch ConfigModuleInstance into the worker
        actor's serialization.

        Args:
            metrics: Metrics dict to populate with MTP metrics (under "mtp_metrics").
            total_num_microbatches: Microbatches accumulated this step. The MTP loss
                logging helper sums the per-microbatch loss without dividing, so we pass
                1/total_num_microbatches to recover the mean (mirroring the MoE path).
            mtp_grad_norm: The MTP parameter group's gradient norm, already reduced across
                the model-parallel group, or None when unavailable (e.g. clip_grad == 0 or
                mtp_detach_heads=False). Logged under "mtp_metrics" as "grad_norm".
        """
        mtp_num_layers = getattr(self.model.config, "mtp_num_layers", None)
        if mtp_num_layers is not None and mtp_num_layers > 0:
            from nemo_rl.models.megatron.common import get_mtp_metrics

            # MTP layers live only on the last pipeline stage, so the tracker is
            # populated there alone. Broadcast to all stages so downstream metric
            # aggregation (which reads rank 0's results) sees them when PP > 1.
            mtp_loss_scale = 1.0 / max(1, total_num_microbatches)
            mtp_metrics = get_mtp_metrics(loss_scale=mtp_loss_scale)
            mtp_metrics = broadcast_loss_metrics_from_last_stage(mtp_metrics)
            # mtp_grad_norm is already MP-reduced (same value on every rank); expose it
            # under the "mtp/" namespace so it logs as train/mtp/grad_norm.
            if mtp_grad_norm is not None:
                mtp_metrics["grad_norm"] = float(mtp_grad_norm)
            if mtp_metrics:
                metrics["mtp_metrics"] = mtp_metrics

    def _set_mtp_grad_scale_func(self, func):
        """Set mtp_grad_scale_func on the model config for MTP loss scaling."""
        config = self._get_model_config()
        if config is not None:
            config.mtp_grad_scale_func = func

    def _get_model_config(self):
        """Get the underlying model config (handle Float16Module wrapper)."""
        model = self.model
        if hasattr(model, "module") and hasattr(model.module, "config"):
            return model.module.config
        elif hasattr(model, "config"):
            return model.config
        return None

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/init_remote_sparse_delta_baseline")
    def init_remote_sparse_delta_baseline(
        self,
        *,
        shard_rank: int,
        shard_count: int,
        transport: str,
    ) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
        return self._require_remote_sparse_refit().initialize_baseline(
            shard_rank=shard_rank,
            shard_count=shard_count,
            transport=transport,
        )

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/stream_remote_sparse_weights")
    def stream_remote_sparse_weights(
        self,
        transport: str,
        targets: list[str],
        *,
        transfer_id: str,
        api_key_env_var: Optional[str],
        timeout_s: float,
        shard_rank: int,
        shard_count: int,
        overwrite_names: list[str],
    ) -> dict[str, int]:
        return self._require_remote_sparse_refit().stream(
            transport,
            targets,
            transfer_id=transfer_id,
            api_key_env_var=api_key_env_var,
            timeout_s=timeout_s,
            shard_rank=shard_rank,
            shard_count=shard_count,
            overwrite_names=overwrite_names,
        )

    def _require_remote_sparse_refit(self) -> Any:
        if self._remote_sparse_refit is None:
            from nemo_rl.models.policy.workers.megatron_remote_sparse_refit import (
                MegatronRemoteSparseRefit,
            )

            refit_config = self.cfg["generation"]["refit_cfg"]
            assert refit_config is not None
            self._remote_sparse_refit = MegatronRemoteSparseRefit(self, refit_config)
        return self._remote_sparse_refit

    def finish_remote_sparse_delta_sync(self, *, succeeded: bool) -> None:
        self._require_remote_sparse_refit().finish(succeeded)

    def _is_fp8_export(self) -> bool:
        """Return True if the train side stores weights as TE FP8 (blockwise or MXFP8)."""
        if self.fp8_cfg is None:
            return False
        return bool(
            self.fp8_cfg.get("fp8_param", False)
            and self.fp8_cfg.get("fp8_recipe") in ("blockwise", "mxfp8")
        )

    def _build_refit_conversion_tasks(self) -> list:
        """Build the conversion-task list driving refit (BF16 or FP8 export).

        For BF16 / FP8-but-fp8_param=False training: standard ``get_conversion_tasks``.
        For FP8-with-fp8_param=True: Bridge's ``build_export_fp8_tasks``, which
        emits a *pair* of tasks per FP8 weight (the FP8 data and a ``*_scale_inv``
        scale tensor).
        """
        if self._is_fp8_export():
            # vLLM's MXFP8 (is_mx) loader expects scales named
            # "<name>_scale_from_checkpoint" (unswizzled E8M0 receive buffer);
            # the blockwise loader expects "<name>_scale_inv".
            scale_suffix = (
                "_scale_from_checkpoint"
                if self.fp8_cfg.get("fp8_recipe") == "mxfp8"
                else "_scale_inv"
            )
            return self.megatron_bridge._model_bridge.build_export_fp8_tasks(
                self.megatron_bridge.hf_pretrained,
                [self.model],
                scale_inv_suffix=scale_suffix,
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
        — used by the nccl_reshard_refit misc-refit path to pass a filtered subset so
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
        """Yield (hf_name, local_tp_shard) for this rank's locally owned FFN params.

        Used by the nccl_reshard_refit bulk path (``build_hf_to_local_param_map``).
        Only the FFN projections (gate/up/down_proj) take the bulk
        path, so this yields ONLY those. Others. take the misc packed_broadcast path
        and are skipped here (see ``is_nccl_reshard_param``).

        Unlike ``_iter_params_with_optional_kv_scales`` (PP broadcast + TP gather
        via ``export_hf_weights``), this yields TP-local shards directly from the
        Megatron params — no collectives.  Returned tensors are views and must
        not be modified in place.  EP: ``refit_conversion_tasks`` already holds
        only this rank's local experts; PP non-local params have
        ``param_weight is None``.
        """
        from megatron.bridge.models.conversion.param_mapping import (
            FusedExpertMapping,
            FusedGatedExpertMapping,
            GatedMLPMapping,
        )

        from nemo_rl.weight_sync.nccl_reshard_utils import is_nccl_reshard_param

        def _expert_idx(megatron_name: str) -> str:
            # Grouped-GEMM experts are numbered by the megatron param name
            m = re.search(r"\d+$", megatron_name)
            assert m, f"expected trailing expert index in {megatron_name!r}"
            return m.group()

        for task in self.refit_conversion_tasks:
            local_tensor = task.param_weight  # local megatron tensor
            if local_tensor is None:
                continue  # non-local PP rank
            # FP8 scale siblings take the misc path.
            if task.global_param_name.endswith(("_scale_inv", "_scale_from_checkpoint")):
                continue

            if isinstance(task.mapping, GatedMLPMapping):
                # FFN gate/up fused in linear_fc1 as [gate_shard; up_shard] (dim 0).
                gate, up = torch.chunk(local_tensor, 2, dim=0)
                yield task.mapping.hf_param["gate"], gate
                yield task.mapping.hf_param["up"], up
                continue

            if isinstance(task.mapping, FusedGatedExpertMapping):
                # Grouped-GEMM MoE (e.g. Qwen3.5-VL): linear_fc1 fuses gate+up per
                # expert [gate; up] (dim 0) — same layout as the dense branch
                # above, but the hf_param is a single, index-less string.  Un-fuse
                # into gate/up AND re-attach the per-expert index.
                idx = _expert_idx(task.global_param_name)
                prefix = str(task.mapping.hf_param)[: -len(".gate_up_proj")]
                gate, up = torch.chunk(local_tensor, 2, dim=0)
                yield f"{prefix}.{idx}.gate_proj.weight", gate
                yield f"{prefix}.{idx}.up_proj.weight", up
                continue

            if isinstance(task.mapping, FusedExpertMapping):
                # Grouped-GEMM down (linear_fc2): re-attach the per-expert index +
                # ``.weight`` so it matches standard per-expert down_proj.
                idx = _expert_idx(task.global_param_name)
                prefix = str(task.mapping.hf_param)[: -len(".down_proj")]
                yield f"{prefix}.{idx}.down_proj.weight", local_tensor
                continue

            # Simple 1:1 mappings: only the FFN down_proj (and any non-gated
            # simple gate/up) hits this branch. QKV (a compound mapping) and
            # every non-FFN param fall through to misc, so they are skipped.
            hf_param = task.mapping.hf_param
            if not isinstance(hf_param, dict) and is_nccl_reshard_param(str(hf_param)):
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

    def _build_layer_to_pp_stage(
        self, pp_size: int, layer_prefix: str
    ) -> dict[str, int]:
        """Build mapping from layer group name to PP stage index.

        Returns a dictionary that maps the layer group name to the PP stage
        index.  ``layer_prefix`` is the module path before ``layers.N`` in the
        exported HF names (e.g. ``model``, ``model.language_model``, ``backbone``)

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
        These cases are checked in check_nccl_reshard_refit_support function.
        """
        # Read from the runtime model's config rather than the bridge's
        # default — the user's per-stage layout overrides
        # (num_layers_in_first/last_pipeline_stage) are applied to the model
        # in setup but never make it into bridge.transformer_config.
        config = self.model.config

        assert getattr(config, "pipeline_model_parallel_layout", None) is None, (
            "nccl_reshard_refit does not support custom pipeline_model_parallel_layout yet"
        )
        assert getattr(config, "virtual_pipeline_model_parallel_size", None) in (
            None,
            1,
        ), (
            "nccl_reshard_refit does not support virtual_pipeline_model_parallel_size > 1 yet"
        )
        assert not getattr(config, "account_for_embedding_in_pipeline_split", False), (
            "nccl_reshard_refit does not support account_for_embedding_in_pipeline_split yet"
        )
        assert not getattr(config, "account_for_loss_in_pipeline_split", False), (
            "nccl_reshard_refit does not support account_for_loss_in_pipeline_split yet"
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
                layer_to_pp_stage[f"{layer_prefix}.layers.{layer_idx}"] = stage
                layer_idx += 1

        assert layer_idx == num_layers, (
            f"Layer assignment incomplete: assigned {layer_idx} of {num_layers}"
        )
        # Embeddings and the final lm_head are taking misc path, we can ignore them here.
        return layer_to_pp_stage

    @torch.no_grad()
    def prepare_nccl_reshard_refit_info(
        self,
        train_parallelism,
        gen_parallelism,
        train_world_size,
        gen_world_size,
    ):
        """Prepare per-layer parameter metadata for nccl_reshard-based refit.

        The builder groups per-expert MoE params into backend-agnostic grouped
        HF entries (gate_proj/up_proj/down_proj); the gen backend maps those into
        its own fused layout (e.g., vLLM w13/w2) gen-side, so this train worker
        stays agnostic to any gen backend's MoE-fusion layout.
        """
        from nemo_rl.weight_sync.nccl_reshard_utils import (
            build_nccl_reshard_refit_info,
            is_nccl_reshard_param,
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

        # Only the FFN gate/up/down weights take the bulk
        # xferdtensor path (>97% of payload for the large models this targets);
        # everything else (attention, embeddings, norms, router, MLA, scales)
        # goes to the misc packed_broadcast + vLLM load_weights path.
        state_dict_metadata = {}
        misc_meta = OrderedDict()
        _xfer_bytes = _bcast_bytes = 0  # full-tensor payload routed to each path

        # Iterates all the params to construct the state_dict_metadata (xferdtensor path)
        # state_dict_metadata[hf_name] -> [shape, dtype]
        # At the same time, filter the params to the misc subset (packed_broadcast path).
        # misc_meta[hf_name] -> [shape, dtype]
        from nemo_rl.weight_sync.nccl_reshard_utils import _extract_layer_prefix

        layer_prefix = None
        with _meta_tensor_alloc_context():
            for name, tensor in self._iter_params_with_optional_kv_scales():
                meta = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
                _nbytes = tensor.numel() * tensor.element_size()
                # Downsized whitelist: only FFN gate/up/down weights take the bulk
                # nccl-reshard path; everything else -> misc (packed_broadcast).
                if is_nccl_reshard_param(name):
                    state_dict_metadata[name] = meta
                    _xfer_bytes += _nbytes
                    if layer_prefix:
                        assert layer_prefix == _extract_layer_prefix(name), (
                            f"layer_prefix mismatch: {layer_prefix} != {_extract_layer_prefix(name)}"
                        )
                    else:  # first param layer_prefix=None
                        layer_prefix = _extract_layer_prefix(name)
                else:
                    misc_meta[name] = meta
                    _bcast_bytes += _nbytes

        _gib = 1024**3
        _tot = _xfer_bytes + _bcast_bytes
        print(
            f"[xferd-payload] ffn_only "
            f"nccl_reshard={_xfer_bytes / _gib:.2f}GiB "
            f"bcast_misc={_bcast_bytes / _gib:.2f}GiB "
            f"total={_tot / _gib:.2f}GiB "
            f"xfer_frac={_xfer_bytes / max(_tot, 1):.1%}",
            flush=True,
        )

        pp_size = train_parallelism.get("pp_size", 1)
        # Construct a dict[layer_name:str] -> pp_stage:int.
        layer_to_pp_stage = None
        assert layer_prefix is not None, "layer_prefix is not set"
        if pp_size > 1:
            layer_to_pp_stage = self._build_layer_to_pp_stage(pp_size, layer_prefix)

        # The key metadata, which should shared with generation workers
        self.nccl_reshard_refit_info = build_nccl_reshard_refit_info(
            state_dict_metadata,
            train_parallelism,
            gen_parallelism,
            train_world_size,
            gen_world_size,
            layer_to_pp_stage=layer_to_pp_stage,
        )
        # Build HFToLocalParamMap (see nccl_reshard_utils)
        self.hf_to_local_param_map = self.build_hf_to_local_param_map(
            self.nccl_reshard_refit_info
        )

        # Keep the misc_meta in the nccl_reshard_refit_info
        # misc_meta[hf_name] -> [shape, dtype]
        self.nccl_reshard_refit_info["misc_meta"] = misc_meta
        # Filter conversion_tasks to the misc subset.
        _misc_names = set(misc_meta.keys())

        def _task_is_misc(task) -> bool:
            # FP8 scale siblings carry the suffix on global_param_name and are
            # always misc (packed_broadcast).
            if task.global_param_name.endswith(("_scale_inv", "_scale_from_checkpoint")):
                return True
            # Compound mappings (QKV/GatedMLP) export homogeneous sub-params
            # (all nccl-reshard or all misc), so the first HF name is representative.
            hf = task.mapping.hf_param
            name = next(iter(hf.values())) if isinstance(hf, dict) else str(hf)
            return name in _misc_names

        self._misc_conversion_tasks = [
            task
            for task in self.refit_conversion_tasks
            if task is not None and _task_is_misc(task)
        ]

        return self.nccl_reshard_refit_info

    def _build_expert_groups(self, param_map):
        """Group this rank's local expert params into stack-ready views.

        Keyed by (prefix, proj_type) and resolved to ordered ``param_map``
        views ready for ``torch.stack``.

        Megatron exposes each expert's projection as a separate param; this bins
        them so ``_group_experts`` can stack a layer's experts into one grouped
        HF tensor per projection.  Called from ``build_hf_to_local_param_map``
        with this rank's local ``param_map``.

        ``_INDIVIDUAL_EXPERT_RE`` captures three fields from a name like
        ``model.layers.3.mlp.experts.17.gate_proj.weight``:
          * group 1 = prefix       -> ``"model.layers.3.mlp.experts"``
          * group 2 = expert index -> ``17``
          * group 3 = proj type    -> ``"gate_proj"``
        so the name keys into ``("model.layers.3.mlp.experts", "gate_proj")``.

        Returns ``{(prefix, proj): [tensor_0, tensor_1, ...]}`` — the per-expert
        ``param_map`` views sorted by expert index.  Example — a layer with 2
        local experts (gated MoE) yields three keys:
          ``(".../experts", "gate_proj"): [view(expert 0), view(expert 1)]``
          ``(".../experts", "up_proj")  : [view(expert 0), view(expert 1)]``
          ``(".../experts", "down_proj"): [view(expert 0), view(expert 1)]``

        Resolving names → views here (rather than per refit in ``_group_experts``)
        costs nothing extra — ``param_map`` already owns these views and they
        stay valid across refits (weights are updated in place; the name→view
        mapping is stable), so ``_group_experts`` only has to ``torch.stack``.
        The index sort matters: the views are stacked in this order, so expert 0
        must precede expert 1 to match the EP ``Shard(0)`` layout the gen side
        expects.
        """
        from nemo_rl.weight_sync.nccl_reshard_utils import _INDIVIDUAL_EXPERT_RE

        index_groups: dict[tuple[str, str], list[tuple[int, str]]] = {}
        for name in param_map:
            # find all the expert params
            m = _INDIVIDUAL_EXPERT_RE.match(name)
            if m:
                # key = (group1 prefix, group3 proj_type); value (group2 idx, name)
                # example: ("model.layers.3.mlp.experts", "gate_proj") -> (0, name)
                index_groups.setdefault((m.group(1), m.group(3)), []).append(
                    (int(m.group(2)), name)
                )
        # Sort by expert index, then resolve each name to its param_map view once.
        return {
            key: [param_map[n] for _, n in sorted(idx_names)]
            for key, idx_names in index_groups.items()
        }

    def _group_experts(self, proj, grouped_name, expert_groups):
        """Stack this rank's local experts for one projection into ``[E_local, ...]``.

        Using the pre-calculated ``expert_groups`` (from ``_build_expert_groups``)
        it is just calling torch.stack of all the local expert params.
        """
        prefix = grouped_name.rsplit(f".{proj}.weight", 1)[0]
        expert_tensors = expert_groups.get((prefix, proj))
        assert expert_tensors, (
            f"no local experts for {grouped_name!r} (proj={proj!r}); "
            "PP-filter / expert-group-metadata inconsistency"
        )
        return torch.stack(expert_tensors)

    def build_hf_to_local_param_map(self, refit_info: dict) -> HFToLocalParamMap:
        """Build the Megatron-backend ``hf_to_local_param_map`` (HFToLocalParamMap).

        Wraps this rank's local Megatron shards into ``LocalParamSpec``s:
        - direct: ``base`` is sharded local tensor view, sent as-is.
        - grouped MoE expert: ``pre`` stacks the per-expert views into
          ``[E_local, ...]`` fresh each refit via ``_group_experts``.
        """
        # This rank's local TP/EP HF param shards (live views), and the
        # per-expert views grouped for torch.stack.  Build-time only.
        param_map = dict(self._iter_local_hf_param_shards())
        expert_groups = self._build_expert_groups(param_map)

        def _expert_spec(proj, grouped_name):
            def pre(_base):
                return RefitCtx(
                    buf=self._group_experts(proj, grouped_name, expert_groups)
                )

            return LocalParamSpec(base=None, pre=pre)

        mapping = {}
        for layer_name in refit_info["layer_names"]:
            for p in refit_info["per_layer_params"][layer_name]:
                name = p["name"]
                if p.get("grouped_expert_proj"):
                    mapping[name] = _expert_spec(p["grouped_expert_proj"], name)
                else:
                    mapping[name] = LocalParamSpec(base=param_map.get(name))
        return HFToLocalParamMap(specs=mapping)

    @torch.no_grad()
    def nccl_reshard_refit(self, kv_scales=None):
        """Transfer weights to generation workers via xferdtensor.

        Uses TP-local shards directly from Megatron parameters, bypassing
        the Bridge's PP broadcast + TP gather.  The modified xferdtensor
        reconstructs the full tensor from per-rank shards internally.

        ``kv_scales`` (FP8 KV cache): the per-layer k/v(/q) scales ride the misc
        packed-broadcast as plain scale tensors (the is_nccl_reshard_param whitelist
        excludes ``.k_scale``/``.v_scale``/``.q_scale`` -> misc); the gen side finalizes
        them via ``_maybe_process_fp8_kv_cache``.  No out-of-band channel needed.
        """
        # hf_to_local_param_map is built once in prepare_nccl_reshard_refit_info;
        # weight values change but the name → spec mapping is stable across
        # refits.
        from nemo_rl.weight_sync.xferdtensor import DTensorRef, xferdtensor

        # spec.pre (grouped-MoE expert stacking) and spec.post enqueue on this
        # worker's current stream; xferdtensor should use the same stream.
        nccl_reshard_stream = torch.cuda.current_stream()
        for layer_name in self.nccl_reshard_refit_info["layer_names"]:
            for param_info in self.nccl_reshard_refit_info["per_layer_params"][
                layer_name
            ]:
                # Each train worker handles only its own PP stage's params
                # (non-PP = every param is in pp_stage 0).
                if param_info.get("pp_stage", 0) != self.my_pp_stage:
                    continue
                group = self.pp_comm_group

                spec = self.hf_to_local_param_map.get(param_info["name"])
                assert spec is not None, (
                    f"no spec for {param_info['name']!r} in hf_to_local_param_map"
                )
                # pre stacks grouped MoE experts fresh each refit; a direct
                # param sends its live TP/EP-local view as-is.
                ctx = (
                    spec.pre(spec.base)  # stack grouped MoE experts
                    if spec.pre is not None
                    else RefitCtx(buf=spec.base)  # send local shard as-is
                )
                assert ctx.buf is not None, (
                    f"no local tensor for {param_info['name']!r}"
                )
                src_tensor = DTensorRef(
                    local_tensor=ctx.buf, global_shape=param_info["global_shape"]
                )
                xferdtensor(
                    src_tensor,
                    param_info["src_mesh_info"],
                    param_info["src_placements"],
                    None,
                    param_info["dst_mesh_info"],
                    param_info["dst_placements"],
                    group,
                    nccl_reshard_stream,
                )
                if spec.post is not None:
                    spec.post(ctx)
                # Drop refs to the per-iteration grouped MoE tensor so its CUDA
                # memory returns to the caching allocator
                del ctx, src_tensor

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        import time

        misc_t0 = time.perf_counter()
        self._broadcast_misc_params_packed(kv_scales=kv_scales)
        torch.cuda.synchronize()
        if torch.distributed.get_rank() == 0:
            print(
                f"[nccl_reshard_refit] misc broadcast (train side): "
                f"{time.perf_counter() - misc_t0:.2f}s",
                flush=True,
            )

    def _broadcast_misc_params_packed(self, kv_scales=None) -> None:
        """Broadcast misc params via the existing packed_broadcast machinery."""
        misc_meta = self.nccl_reshard_refit_info.get("misc_meta", {})
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

    def _use_real_quant_refit(self) -> bool:
        return False

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
        """Offload model params to CPU after inference. Only used in PPO."""
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

        With async_save=True, this method returns after D2H staging. The actual
        disk write continues in a background persistent worker process. Callers
        must call finalize_async_save() before renaming the directory or starting
        another save.

        With async_save=False (default), this blocks until the write is complete.

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
        is_async = self.mcore_state.cfg.checkpoint.async_save

        try:
            # Block until any previous async save is fully written to disk.
            # With sync save this is a no-op.
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=True,
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

            if not is_async:
                # Sync path: finalize immediately (runs finalize_fns + barrier).
                maybe_finalize_async_save(
                    self.mcore_state,
                    ckpt_cfg=self.mcore_state.cfg.checkpoint,
                    blocking=True,
                )
            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

            if not is_training:
                self.model.train()

        except Exception as e:
            print(f"Failed to save checkpoint to {weights_path}: {e}")
            raise
        finally:
            self.mcore_state.cfg.checkpoint.save = original_save_path

    def finalize_async_save(self):
        """Block until the in-flight async write completes and run finalize_fns.

        Safe to call when async_save is disabled (no-op).
        Does NOT terminate the persistent worker — it stays alive for the next save.
        """
        maybe_finalize_async_save(
            self.mcore_state,
            ckpt_cfg=self.mcore_state.cfg.checkpoint,
            blocking=True,
        )

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

        # Run a forward pass to trigger hooks (reuse get_logprobs forward path).
        # Calibration batches are prompt-only model inputs, not rollout replay
        # batches, so they intentionally do not carry routed_experts.
        try:
            _ = self.get_logprobs(
                data=data,
                micro_batch_size=micro_batch_size,
                require_router_replay=False,
            )
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
