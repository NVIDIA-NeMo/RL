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

"""Template Megatron worker for block-diffusion logprob objectives.

Diffusion policies do not score ``input_ids`` shifted by one. They score a
*transformed* batch -- a ``[noisy | clean]`` layout built from the rollout --
at same-position targets, under an attention mask that is block-bidirectional
on the noisy half. That makes four things subclass-specific:

* how the rollout batch becomes a Megatron batch
  (``_build_{training,logprob}_megatron_batch``),
* what attention state the forward runs under
  (``_{training,logprob}_attention_context``),
* which post-processors gather the logprobs
  (``_make_{loss,logprobs}_post_processor``),
* how the resulting logprobs map back to NeMo RL's ``[B, S]`` convention
  (``_finalize_logprobs_from_outputs``).

This class owns the surrounding ``train`` / ``get_logprobs`` loops so each
concrete diffusion objective only supplies those hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import deepcopy
from typing import Any, Iterable, Iterator, Optional

import torch
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.core import parallel_state
from megatron.core.rerun_state_machine import get_rerun_state_machine

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.common import get_aux_loss_track_names, get_moe_metrics
from nemo_rl.models.megatron.data import get_microbatch_iterator, process_global_batch
from nemo_rl.models.megatron.pipeline_parallel import (
    broadcast_loss_metrics_from_last_stage,
    broadcast_tensors_from_last_stage,
)
from nemo_rl.models.megatron.train import (
    LogprobsPostProcessor,
    LossPostProcessor,
    aggregate_training_statistics,
    megatron_forward_backward,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.draft_config import coerce_draft_config
from nemo_rl.models.policy.interfaces import LogprobOutputSpec
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from nemo_rl.utils.grad_norm import warn_if_inf_grad_norm
from nemo_rl.utils.nsys import wrap_with_nvtx_name


class DiffusionMegatronPolicyWorkerImpl(MegatronPolicyWorkerImpl, ABC):
    """Template worker for diffusion-policy Megatron logprob objectives."""

    def _cfg_without_sequence_packing(self) -> PolicyConfig:
        cfg = deepcopy(self.cfg)
        cfg["sequence_packing"] = {**cfg["sequence_packing"], "enabled": False}
        if "dynamic_batching" in cfg:
            cfg["dynamic_batching"] = {**cfg["dynamic_batching"], "enabled": False}
        return cfg

    def _validate_diffusion_support(self, algorithm_name: str) -> None:
        # Context parallelism is rejected outright. The diffusion layout runs
        # unpacked, and Megatron's unpacked microbatch path does not CP-shard
        # its inputs (nemo_rl/models/megatron/data.py sets
        # ``input_ids_cp_sharded = input_ids`` on that branch), so every CP rank
        # would see the full sequence while the post-processors assume a
        # zigzag-sharded one. The dense pad factor is already inflated by
        # lcm(..., cp_size * 2), so the shapes would line up and the logprobs
        # would silently come from the wrong tokens.
        if self.cfg["megatron_cfg"]["context_parallel_size"] != 1:
            raise NotImplementedError(
                f"{algorithm_name} Megatron logprobs require "
                "megatron_cfg.context_parallel_size=1; got "
                f"{self.cfg['megatron_cfg']['context_parallel_size']}"
            )
        draft = coerce_draft_config(self.cfg.get("draft"))
        if draft is not None and draft.enabled:
            raise NotImplementedError(
                f"{algorithm_name} Megatron training does not support draft model "
                "training"
            )
        # MTP predicts n-ahead tokens from input_ids; the diffusion layout's
        # targets are not input_ids shifted by one, so the MTP loss would be
        # computed against the wrong tokens without ever raising.
        if self.cfg["megatron_cfg"].get("mtp_num_layers"):
            raise NotImplementedError(
                f"{algorithm_name} Megatron training does not support MTP"
            )

    @abstractmethod
    def _validate_diffusion_algorithm_support(self) -> None:
        """Validate subclass-specific diffusion objective support."""

    def _clear_diffusion_inference_state(self) -> None:
        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None

        for module in self.model.modules():
            if hasattr(module, "reset_inference_cache"):
                module.reset_inference_cache()
            if hasattr(module, "_inference_key_value_memory"):
                module._inference_key_value_memory = None
            if hasattr(module, "set_inference_mode"):
                module.set_inference_mode(False)
            if hasattr(module, "clear_kv_cache"):
                module.clear_kv_cache()
            if hasattr(module, "clear_block_bidirectional_mask"):
                module.clear_block_bidirectional_mask()

    @staticmethod
    def _diffusion_attention_modules(model) -> list[Any]:
        return [
            module
            for module in model.modules()
            if hasattr(module, "set_inference_mode")
            and hasattr(module, "set_inference_params")
            and hasattr(module, "clear_kv_cache")
        ]

    @contextmanager
    def _fixed_diffusion_mask_seq_length_context(self, half_seq_length: int):
        attention_modules = self._diffusion_attention_modules(self.model)
        previous = []
        for module in attention_modules:
            if not hasattr(module, "mask_seq_length"):
                continue
            previous.append(
                (
                    module,
                    getattr(module, "mask_seq_length"),
                    getattr(module, "_mask_cache", None),
                )
            )
            module.mask_seq_length = int(half_seq_length)
            if hasattr(module, "_mask_cache"):
                module._mask_cache = {}
        try:
            yield
        finally:
            for module, mask_seq_length, mask_cache in previous:
                module.mask_seq_length = mask_seq_length
                if mask_cache is not None:
                    module._mask_cache = mask_cache

    @contextmanager
    def _megatron_attention_context(self, attention_mode: str):
        if attention_mode == "training":
            yield
            return
        if attention_mode not in {
            "inference_causal",
            "inference_bidirectional",
            "inference_block_bidirectional",
        }:
            raise ValueError(f"Unsupported megatron_attention_mode: {attention_mode}")

        attention_modules = self._diffusion_attention_modules(self.model)
        causal = attention_mode == "inference_causal"
        for module in attention_modules:
            module.clear_kv_cache()
            module.set_inference_mode(True)
            module.set_inference_params(causal=causal, cache_enabled=False)
        try:
            yield
        finally:
            for module in attention_modules:
                module.set_inference_mode(False)
                module.clear_kv_cache()
                if hasattr(module, "clear_block_bidirectional_mask"):
                    module.clear_block_bidirectional_mask()

    @contextmanager
    def _fixed_training_diffusion_attention_context(self, half_seq_length: int):
        with self._fixed_diffusion_mask_seq_length_context(half_seq_length):
            with self._megatron_attention_context("training"):
                yield

    def _diffusion_block_size(self) -> int:
        config = getattr(self.model, "config", None)
        return int(getattr(config, "block_size", 32))

    def _maybe_print_diffusion_block_size(self, source: str, block_size: int) -> None:
        printed = getattr(self, "_diffusion_block_size_printed", set())
        if source in printed:
            return
        try:
            rank = (
                torch.distributed.get_rank()
                if torch.distributed.is_initialized()
                else self.rank
            )
        except Exception:
            rank = self.rank
        if rank != 0:
            return

        model_config = getattr(self.model, "config", None)
        config_block_size = getattr(model_config, "block_size", None)
        print(
            "DIFFUSION_BLOCK_SIZE "
            f"source={source} rank={rank} "
            f"block_size_used={block_size} model_config_block_size={config_block_size}",
            flush=True,
        )
        printed.add(source)
        self._diffusion_block_size_printed = printed

    def _debug_rank(self) -> int:
        try:
            if torch.distributed.is_initialized():
                return torch.distributed.get_rank()
        except Exception:
            pass
        return self.rank

    def _debug_print_batch_shapes(self, label: str, data: BatchedDataDict[Any]) -> None:
        if self._debug_rank() != 0:
            return
        shapes = {
            key: tuple(value.shape)
            for key, value in data.items()
            if isinstance(value, torch.Tensor)
        }
        print(
            f"[Diffusion-DEBUG] {self.__class__.__name__} {label} shapes={shapes}",
            flush=True,
        )

    def _debug_print_attention_mode(self, label: str) -> None:
        if self._debug_rank() != 0:
            return
        modules = self._diffusion_attention_modules(self.model)
        if not modules:
            print(
                f"[Diffusion-DEBUG] {self.__class__.__name__} {label}: "
                "no NemotronLabsDiffusionAttention modules",
                flush=True,
            )
            return
        module = modules[0]
        print(
            f"[Diffusion-DEBUG] {self.__class__.__name__} {label}: "
            f"num_attn_modules={len(modules)} "
            f"inference_mode={getattr(module, '_inference_mode', None)} "
            f"inference_causal={getattr(module, '_inference_causal', None)} "
            f"cache_enabled={getattr(module, '_cache_enabled', None)} "
            f"mask_seq_length={getattr(module, 'mask_seq_length', None)} "
            f"block_bidirectional="
            f"{getattr(module, '_block_bidirectional_starts', None) is not None}",
            flush=True,
        )

    def _training_attention_context(self) -> AbstractContextManager[Any]:
        return nullcontext()

    def _logprob_attention_context(self) -> AbstractContextManager[Any]:
        return nullcontext()

    def _wrap_training_microbatch_iterator(
        self,
        data_iterator: Iterator[Any],
        cfg: PolicyConfig,
    ) -> Iterator[Any]:
        return data_iterator

    def _wrap_logprob_microbatch_iterator(
        self,
        data_iterator: Iterator[Any],
        cfg: PolicyConfig,
    ) -> Iterator[Any]:
        return data_iterator

    @abstractmethod
    def _build_training_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        mbs: int,
    ) -> tuple[BatchedDataDict[Any], PolicyConfig, int, dict[str, Any]]:
        """Return transformed training data, config, microbatch size, metadata."""

    @abstractmethod
    def _build_logprob_megatron_batch(
        self,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int],
    ) -> tuple[
        BatchedDataDict[Any] | None,
        PolicyConfig,
        int,
        dict[str, Any],
    ]:
        """Return transformed logprob data, config, microbatch size, metadata."""

    @abstractmethod
    def _make_loss_post_processor(
        self,
        loss_fn: LossFunction,
        cfg: PolicyConfig,
        num_microbatches: int,
    ) -> LossPostProcessor:
        """Create the subclass loss post-processor."""

    @abstractmethod
    def _make_logprobs_post_processor(
        self,
        cfg: PolicyConfig,
    ) -> LogprobsPostProcessor:
        """Create the subclass logprob post-processor."""

    @abstractmethod
    def _finalize_logprobs_from_outputs(
        self,
        list_of_logprobs: list[dict[str, torch.Tensor]],
        *,
        original_data: BatchedDataDict[Any],
        transformed_data: BatchedDataDict[Any],
        metadata: dict[str, Any],
    ) -> torch.Tensor:
        """Convert post-processor outputs back to NeMo RL's [B, S] convention."""

    def _valid_toks_correction(self, batch) -> int:
        """Tokens counted in ``token_mask`` but excluded from the loss.

        Subtracted from ``global_valid_toks`` at the normalizer seam so the
        loss normalizer matches the loss coverage. Zero by default.
        """
        return 0

    def _coupled_pair_count_scale(self) -> int:
        """Multiplier on the global valid seq/token counts used to normalize the loss.

        Defaults to 1. Subclasses whose training schedule accumulates each
        response token / sequence more than once per optimizer step return that
        multiplicity so the loss stays a proper Monte-Carlo average instead of
        being over-scaled.
        """
        return 1

    @wrap_with_nvtx_name("diffusion_megatron_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        check_dim_skip_keys: Optional[Iterable[str]] = None,
    ) -> dict[str, Any]:
        """Train on a batch, routing every forward through the diffusion hooks.

        ``check_dim_skip_keys`` is accepted for parity with the DTensor worker
        (``lm_policy`` passes it unconditionally) and must be ``None``:
        Megatron does not run cross-tokenizer distillation.
        """
        assert check_dim_skip_keys is None, (
            "check_dim_skip_keys is only supported by the v2 DTensor worker; "
            "Megatron does not run cross-tokenizer distillation."
        )
        self._validate_diffusion_algorithm_support()
        self.timer.start("train")
        self._clear_diffusion_inference_state()

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
            self.model.train()
            saved_extra_state = None
            reenable_forward_pre_hook_after_eval = False

        with ctx, self._training_attention_context():
            self._debug_print_attention_mode("train forward")
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
                self._debug_print_batch_shapes(
                    "train: before _build_training_megatron_batch",
                    gb_result["batch"],
                )
                (
                    megatron_batch,
                    cfg_for_training,
                    train_mbs,
                    _metadata,
                ) = self._build_training_megatron_batch(gb_result["batch"], mbs)
                self._debug_print_batch_shapes(
                    "train: after _build_training_megatron_batch",
                    megatron_batch,
                )
                global_valid_seqs = gb_result["global_valid_seqs"]
                global_valid_toks = gb_result["global_valid_toks"]
                # Subtract algo-specific tokens that are counted in token_mask but
                # excluded from the loss (0 by default) so the loss normalizer
                # matches the loss coverage.
                global_valid_toks = global_valid_toks - self._valid_toks_correction(
                    gb_result["batch"]
                )
                # Scale the loss normalizer when the training schedule accumulates
                # each token / sequence more than once per step. Only the counts
                # passed to forward/backward are scaled; the reported metrics below
                # keep the true (unscaled) counts. No-op at scale 1.
                count_scale = self._coupled_pair_count_scale()
                fb_global_valid_seqs = global_valid_seqs * count_scale
                fb_global_valid_toks = global_valid_toks * count_scale

                (
                    data_iterator,
                    num_microbatches,
                    micro_batch_size,
                    _seq_length,
                    padded_seq_length,
                ) = get_microbatch_iterator(
                    megatron_batch,
                    cfg_for_training,
                    train_mbs,
                    straggler_timer=self.mcore_state.straggler_timer,
                    delegate_pack_to_model=self.delegate_pack_to_model,
                    delegate_mtp_loss_mask_to_model=self.delegate_mtp_loss_mask_to_model,
                    model_slices_context_parallel_inputs=self.model_slices_context_parallel_inputs,
                    mtp_enabled=self.mtp_enabled,
                )
                data_iterator = self._wrap_training_microbatch_iterator(
                    data_iterator,
                    cfg_for_training,
                )
                total_num_microbatches += int(num_microbatches)

                loss_post_processor = self._make_loss_post_processor(
                    loss_fn=loss_fn,
                    cfg=cfg_for_training,
                    num_microbatches=num_microbatches,
                )

                rerun_state_machine = get_rerun_state_machine()
                losses_reduced: list[dict[str, Any]] = []
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # For MXFP8 overlap eval the param and grad buffers are
                    # shared and pre-hooks were disabled above; zeroing the
                    # shared param buffer before a forward-only pass would
                    # corrupt the weights.
                    if not (
                        eval_mode and self._uses_mxfp8_overlap_shared_param_buffer()
                    ):
                        self.model.zero_grad_buffer()
                        self.optimizer.zero_grad()
                        self._copy_main_params_to_param_buffer()

                    # Normalize the MoE aux-loss gradient consistently with the
                    # per-token main loss; see MegatronPolicyWorkerImpl.train.
                    self._set_moe_grad_scale_func(
                        self._compute_moe_grad_scale(global_valid_toks)
                    )
                    try:
                        losses_reduced = megatron_forward_backward(
                            model=self.model,
                            data_iterator=data_iterator,
                            num_microbatches=num_microbatches,
                            seq_length=padded_seq_length,
                            mbs=micro_batch_size,
                            post_processing_fn=loss_post_processor,
                            forward_only=eval_mode,
                            defer_fp32_logits=self.defer_fp32_logits,
                            global_valid_seqs=fb_global_valid_seqs,
                            global_valid_toks=fb_global_valid_toks,
                            sampling_params=self.sampling_params,
                            straggler_timer=self.mcore_state.straggler_timer,
                            draft_model=None,
                            enable_hidden_capture=False,
                            use_fused_linear_logprobs=False,
                        )
                    finally:
                        self._set_moe_grad_scale_func(None)

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (True, 0.0, 0.0)

                pg_collection = get_pg_collection(self.model)
                update_successful = logical_and_across_model_parallel_group(
                    update_successful,
                    mp_group=pg_collection.mp,
                )
                grad_norm = reduce_max_stat_across_model_parallel_group(
                    grad_norm,
                    mp_group=pg_collection.mp,
                )
                num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(
                    num_zeros_in_grad,
                    mp_group=pg_collection.mp,
                )
                if not eval_mode:
                    self._restore_first_train_step_param_sync(update_successful)
                warn_if_inf_grad_norm(grad_norm)

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
                    torch.cuda.empty_cache()

                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
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
            self.scheduler.step(increment=gbs)

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
        # train.__code__.co_names; with torch 2.11 cloudpickle otherwise matches
        # torch.distributed.config (a non-pickleable ConfigModuleInstance).
        model_config = getattr(self.model, "config", None)
        num_moe_experts = getattr(model_config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"]["moe_per_layer_logging"],
                # Pre-initialize the aux-loss tracker on every PP rank so the
                # cross-PP all_reduce inside get_moe_metrics does not hang when a
                # rank recorded no aux loss this step.
                num_layers=getattr(model_config, "num_layers", None),
                mtp_num_layers=getattr(model_config, "mtp_num_layers", None),
                track_names=get_aux_loss_track_names(model_config),
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics
        self.timer.stop("train")
        return metrics

    @wrap_with_nvtx_name("diffusion_megatron_policy_worker/get_logprobs")
    def get_logprobs(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        require_router_replay: bool = True,
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Score a rollout batch through the diffusion forward path.

        ``require_router_replay`` is accepted because
        ``get_reference_policy_logprobs`` passes it explicitly. It is ignored:
        router replay traces an autoregressive forward and the diffusion path
        scores a different (transformed) batch, so a replayed trace would not
        apply.
        """
        del require_router_replay
        self._validate_diffusion_algorithm_support()
        self.timer.start("get_logprobs")
        no_grad = torch.no_grad()
        no_grad.__enter__()

        self._debug_print_batch_shapes(
            "logprob: before _build_logprob_megatron_batch", data
        )
        (
            transformed_data,
            cfg_for_logprobs,
            logprob_mbs,
            metadata,
        ) = self._build_logprob_megatron_batch(data, micro_batch_size)
        if transformed_data is not None:
            self._debug_print_batch_shapes(
                "logprob: after _build_logprob_megatron_batch", transformed_data
            )
        if transformed_data is None:
            no_grad.__exit__(None, None, None)
            self.timer.stop("get_logprobs")
            return BatchedDataDict[LogprobOutputSpec](
                logprobs=metadata["empty_logprobs"]
            ).to("cpu")

        self._clear_diffusion_inference_state()
        self.model.eval()

        (
            mb_iterator,
            num_microbatches,
            micro_batch_size,
            _seq_length,
            padded_seq_length,
        ) = get_microbatch_iterator(
            transformed_data,
            cfg_for_logprobs,
            logprob_mbs,
            straggler_timer=self.mcore_state.straggler_timer,
            delegate_pack_to_model=self.delegate_pack_to_model,
            delegate_mtp_loss_mask_to_model=self.delegate_mtp_loss_mask_to_model,
            model_slices_context_parallel_inputs=self.model_slices_context_parallel_inputs,
            mtp_enabled=self.mtp_enabled,
        )
        mb_iterator = self._wrap_logprob_microbatch_iterator(
            mb_iterator,
            cfg_for_logprobs,
        )

        logprobs_post_processor = self._make_logprobs_post_processor(
            cfg=cfg_for_logprobs,
        )

        with self._logprob_attention_context():
            self._debug_print_attention_mode("logprob forward")
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
                use_fused_linear_logprobs=False,
            )

        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            logprobs = self._finalize_logprobs_from_outputs(
                list_of_logprobs,
                original_data=data,
                transformed_data=transformed_data,
                metadata=metadata,
            )
            tensors = {"logprobs": logprobs}
        else:
            tensors = {"logprobs": None}
        logprobs = broadcast_tensors_from_last_stage(tensors)["logprobs"]

        no_grad.__exit__(None, None, None)
        self.timer.stop("get_logprobs")
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs).to("cpu")
