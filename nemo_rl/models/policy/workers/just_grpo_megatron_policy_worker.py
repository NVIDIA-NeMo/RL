# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

import os
from contextlib import AbstractContextManager, contextmanager, nullcontext
from copy import deepcopy
from typing import Any, Iterator, Optional

import ray
import torch
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.core import parallel_state
from megatron.core.rerun_state_machine import get_rerun_state_machine

from nemo_rl.algorithms.just_grpo_logprobs import (
    build_leftmost_reveal_batch,
    build_leftmost_reveal_loss_batch,
    get_leftmost_reveal_logprob_estimation_cfg,
    pad_reveal_batch_to_multiple,
    scatter_leftmost_reveal_logprobs,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.common import get_moe_metrics
from nemo_rl.models.megatron.data import get_microbatch_iterator, process_global_batch
from nemo_rl.models.megatron.just_grpo_train import (
    JustGRPOLogprobsPostProcessor,
    JustGRPOLossPostProcessor,
)
from nemo_rl.models.megatron.pipeline_parallel import (
    broadcast_loss_metrics_from_last_stage,
    broadcast_tensors_from_last_stage,
)
from nemo_rl.models.megatron.train import (
    aggregate_training_statistics,
    megatron_forward_backward,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import LogprobOutputSpec
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import MegatronPolicyWorkerImpl
from nemo_rl.utils.nsys import wrap_with_nvtx_name


class JustGRPOMegatronPolicyWorkerImpl(MegatronPolicyWorkerImpl):
    """Megatron worker with JustGRPO leftmost-reveal logprob semantics."""

    def _cfg_without_sequence_packing(self) -> PolicyConfig:
        cfg = deepcopy(self.cfg)
        cfg["sequence_packing"] = {**cfg["sequence_packing"], "enabled": False}
        if "dynamic_batching" in cfg:
            cfg["dynamic_batching"] = {**cfg["dynamic_batching"], "enabled": False}
        return cfg

    def _validate_just_grpo_support(self) -> None:
        if self.cfg["megatron_cfg"]["context_parallel_size"] != 1:
            raise NotImplementedError(
                "JustGRPO Megatron logprobs currently require context_parallel_size=1"
            )
        if "draft" in self.cfg and self.cfg["draft"]["enabled"]:
            raise NotImplementedError(
                "JustGRPO Megatron training does not support draft model training"
            )

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
    def _megatron_attention_context(self, attention_mode: str):
        """Switch JustGRPO reveal rows to diffusion inference attention when requested."""
        if attention_mode == "training":
            yield
            return
        if attention_mode not in {
            "inference_bidirectional",
            "inference_block_bidirectional",
        }:
            raise ValueError(f"Unsupported megatron_attention_mode: {attention_mode}")

        attention_modules = self._diffusion_attention_modules(self.model)
        for module in attention_modules:
            module.clear_kv_cache()
            module.set_inference_mode(True)
            module.set_inference_params(causal=False, cache_enabled=False)
        try:
            yield
        finally:
            for module in attention_modules:
                module.set_inference_mode(False)
                module.clear_kv_cache()
                if hasattr(module, "clear_block_bidirectional_mask"):
                    module.clear_block_bidirectional_mask()

    def _set_block_bidirectional_mask(
        self,
        attention_modules: list[Any],
        data: BatchedDataDict[Any],
        seq_len: int,
        block_size: int,
    ) -> None:
        target_positions = data["just_grpo_target_positions"]
        response_starts = data["just_grpo_response_starts"]
        device = target_positions.device
        relative_positions = (target_positions - response_starts).clamp_min(0)
        block_starts = response_starts + (relative_positions // block_size) * block_size
        block_ends = (block_starts + block_size).clamp(max=seq_len)
        block_starts = block_starts.to(device=device, dtype=torch.long)
        block_ends = block_ends.to(device=device, dtype=torch.long)

        for module in attention_modules:
            if not hasattr(module, "set_block_bidirectional_mask"):
                raise RuntimeError(
                    "Megatron diffusion attention does not support "
                    "set_block_bidirectional_mask. Ensure the Megatron-Bridge "
                    "block-bidirectional attention patch is on PYTHONPATH."
                )
            module.set_block_bidirectional_mask(block_starts, block_ends)

    def _with_block_bidirectional_attention(
        self,
        data_iterator: Iterator[Any],
        block_size: int,
    ) -> Iterator[Any]:
        attention_modules = self._diffusion_attention_modules(self.model)
        for processed_mb in data_iterator:
            self._set_block_bidirectional_mask(
                attention_modules=attention_modules,
                data=processed_mb.data_dict,
                seq_len=processed_mb.input_ids.shape[1],
                block_size=block_size,
            )
            yield processed_mb

    def _diffusion_block_size(self) -> int:
        config = getattr(self.model, "config", None)
        return int(getattr(config, "block_size", 32))

    def _drop_non_sequence_reveal_metadata(
        self, data: BatchedDataDict[Any]
    ) -> BatchedDataDict[Any]:
        """Drop reveal metadata that does not follow Megatron's [B, S] convention."""
        return BatchedDataDict[Any](
            {
                key: value
                for key, value in data.items()
                if key != "just_grpo_output_shape"
            }
        )

    def report_node_ip_and_gpu_id(self) -> tuple[str, int]:
        """Report manually pinned GPU IDs when Ray GPU resources are disabled."""
        ip = ray._private.services.get_node_ip_address()
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            gpu_id = int(visible_devices.split(",")[0])
            return (ip, gpu_id)
        return super().report_node_ip_and_gpu_id()

    @torch.no_grad()
    @wrap_with_nvtx_name("just_grpo_megatron_policy_worker/stream_weights_via_http")
    def stream_weights_via_http(
        self,
        sglang_url_to_gpu_uuids: dict[str, list[str]],
    ) -> None:
        """Stream exported Megatron weights to colocated SGLang servers."""
        from nemo_rl.models.policy.utils import stream_weights_via_http_impl

        stream_weights_via_http_impl(
            params_generator=self._iter_params_with_optional_kv_scales(),
            sglang_url_to_gpu_uuids=sglang_url_to_gpu_uuids,
            rank=self.rank,
            worker_name=str(self),
            current_device_uuid=self.report_device_id(),
        )

    @wrap_with_nvtx_name("just_grpo_megatron_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        self._validate_just_grpo_support()
        if hasattr(loss_fn, "loss_type") and loss_fn.loss_type != LossType.TOKEN_LEVEL:
            raise NotImplementedError(
                "JustGRPO Megatron training currently supports token-level loss only"
            )

        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None

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
        else:
            ctx = nullcontext()
            self.model.train()

        logprob_estimation_cfg = get_leftmost_reveal_logprob_estimation_cfg(self.cfg)
        cfg_for_training = self._cfg_without_sequence_packing()
        train_reveal_batch_size = logprob_estimation_cfg["train_reveal_batch_size"]
        reveal_schedule = logprob_estimation_cfg["reveal_schedule"]
        attention_mode = logprob_estimation_cfg["megatron_attention_mode"]
        max_reveal_positions = (
            logprob_estimation_cfg["max_reveal_positions"]
            if reveal_schedule == "fixed_response_window"
            else None
        )

        with ctx, self._megatron_attention_context(attention_mode):
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
                batch = build_leftmost_reveal_loss_batch(
                    gb_result["batch"],
                    mask_token_id=logprob_estimation_cfg["mask_token_id"],
                    reveal_schedule=reveal_schedule,
                    max_reveal_positions=max_reveal_positions,
                )
                pad_reveal_batch_to_multiple(batch, train_reveal_batch_size)
                megatron_batch = self._drop_non_sequence_reveal_metadata(batch)
                global_valid_seqs = gb_result["global_valid_seqs"]
                global_valid_toks = gb_result["global_valid_toks"]

                (
                    data_iterator,
                    num_microbatches,
                    micro_batch_size,
                    _seq_length,
                    padded_seq_length,
                ) = get_microbatch_iterator(
                    megatron_batch,
                    cfg_for_training,
                    train_reveal_batch_size,
                    straggler_timer=self.mcore_state.straggler_timer,
                )
                if attention_mode == "inference_block_bidirectional":
                    data_iterator = self._with_block_bidirectional_attention(
                        data_iterator,
                        block_size=self._diffusion_block_size(),
                    )
                total_num_microbatches += int(num_microbatches)

                loss_post_processor = JustGRPOLossPostProcessor(
                    loss_fn=loss_fn,
                    cfg=cfg_for_training,
                    num_microbatches=num_microbatches,
                    sampling_params=self.sampling_params,
                    draft_model=None,
                )

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()
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
                        draft_model=None,
                        enable_hidden_capture=False,
                        use_linear_ce_fusion_loss=False,
                    )

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
        num_moe_experts = getattr(self.model.config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"]["moe_per_layer_logging"],
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics
        return metrics

    @wrap_with_nvtx_name("just_grpo_megatron_policy_worker/get_logprobs")
    def get_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        self._validate_just_grpo_support()
        no_grad = torch.no_grad()
        no_grad.__enter__()
        logprob_estimation_cfg = get_leftmost_reveal_logprob_estimation_cfg(self.cfg)
        reveal_schedule = logprob_estimation_cfg["reveal_schedule"]
        attention_mode = logprob_estimation_cfg["megatron_attention_mode"]
        max_reveal_positions = (
            logprob_estimation_cfg["max_reveal_positions"]
            if reveal_schedule == "fixed_response_window"
            else None
        )
        reveal_data = build_leftmost_reveal_batch(
            input_ids=data["input_ids"],
            input_lengths=data["input_lengths"],
            token_mask=data["token_mask"],
            sample_mask=data.get("sample_mask", None),
            mask_token_id=logprob_estimation_cfg["mask_token_id"],
            reveal_schedule=reveal_schedule,
            max_reveal_positions=max_reveal_positions,
        )
        reveal_batch_size = logprob_estimation_cfg["reveal_batch_size"]
        if reveal_data["input_ids"].shape[0] == 0:
            no_grad.__exit__(None, None, None)
            return BatchedDataDict[LogprobOutputSpec](
                logprobs=torch.zeros_like(data["input_ids"], dtype=torch.float32)
            )
        reveal_count = pad_reveal_batch_to_multiple(reveal_data, reveal_batch_size)
        megatron_reveal_data = self._drop_non_sequence_reveal_metadata(reveal_data)
        cfg_for_logprobs = self._cfg_without_sequence_packing()

        self.model.eval()
        (
            mb_iterator,
            num_microbatches,
            micro_batch_size,
            _seq_length,
            padded_seq_length,
        ) = get_microbatch_iterator(
            megatron_reveal_data,
            cfg_for_logprobs,
            reveal_batch_size,
            straggler_timer=self.mcore_state.straggler_timer,
        )
        if attention_mode == "inference_block_bidirectional":
            mb_iterator = self._with_block_bidirectional_attention(
                mb_iterator,
                block_size=self._diffusion_block_size(),
            )

        logprobs_post_processor = JustGRPOLogprobsPostProcessor(
            cfg=cfg_for_logprobs,
            sampling_params=self.sampling_params,
            use_linear_ce_fusion=False,
        )

        with self._megatron_attention_context(attention_mode):
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
                use_linear_ce_fusion_loss=False,
            )

        if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            flat_logprobs = torch.cat([l["logprobs"] for l in list_of_logprobs], dim=0)
            flat_logprobs = flat_logprobs[:reveal_count]
            logprobs = scatter_leftmost_reveal_logprobs(
                flat_logprobs=flat_logprobs,
                batch_indices=reveal_data["just_grpo_batch_indices"][:reveal_count],
                target_positions=reveal_data["just_grpo_target_positions"][
                    :reveal_count
                ],
                output_shape=reveal_data["just_grpo_output_shape"][:reveal_count],
                row_mask=reveal_data["just_grpo_row_mask"][:reveal_count],
            )
            tensors = {"logprobs": logprobs}
        else:
            tensors = {"logprobs": None}
        logprobs = broadcast_tensors_from_last_stage(tensors)["logprobs"]

        no_grad.__exit__(None, None, None)
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs).to("cpu")


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class JustGRPOMegatronPolicyWorker(JustGRPOMegatronPolicyWorkerImpl):
    @staticmethod
    def configure_worker(
        num_gpus: float,
        bundle_indices: Optional[tuple[int, list[int]]] = None,
    ) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
        """Keep tensor-parallel ranks on their placement-group GPU.

        Colocated generation uses fractional Ray GPU resources. For Megatron tensor
        parallelism, allowing Ray to rewrite CUDA_VISIBLE_DEVICES can assign
        multiple ranks to the same physical GPU. When the placement-group bundle
        gives us a single physical GPU, pin that rank explicitly and avoid Ray's
        accelerator-id rewrite path by not requesting an additional GPU resource.
        """
        resources: dict[str, Any] = {"num_gpus": num_gpus}
        env_vars = {
            "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
            "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        }
        if bundle_indices is not None and len(bundle_indices[1]) == 1:
            resources["num_gpus"] = 0
            env_vars["CUDA_VISIBLE_DEVICES"] = str(bundle_indices[1][0])
            env_vars["LOCAL_RANK"] = "0"
        return resources, env_vars, {}
