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

"""Research worker. Native batching/loss logic pinned to main 26e2b802."""

import os
import warnings
from contextlib import AbstractContextManager, nullcontext
from functools import partial
from typing import Any, Iterable, Iterator, Optional, cast

import ray
import torch
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

from logra.compression import local_weight_rows
from logra.config import LoGRAConfig
from logra.cua import CuaConfig, CuaUpdate
from logra.optimizer import LoGRAOptimizer
from logra.setup import build_scheduler
from nemo_rl.algorithms.logits_sampling_utils import need_top_k_or_top_p_filtering
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.algorithms.metric_utils import LEARNING_RATE_KEY
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import ChunkedDistributedLogprob
from nemo_rl.models.automodel.data import (
    ProcessedMicrobatch,
    check_sequence_dim,
    get_microbatch_iterator,
    process_global_batch,
)
from nemo_rl.models.automodel.train import (
    LossPostProcessor,
    aggregate_training_statistics,
    automodel_forward_backward,
)
from nemo_rl.models.policy.draft_config import coerce_draft_config
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.automodel_policy_worker import (
    AutomodelPolicyWorkerImpl,
)
from nemo_rl.utils.grad_norm import warn_if_inf_grad_norm


class CuaLossPostProcessor(LossPostProcessor):
    """Reuse native chunked log probabilities to bound the FP32 working set."""

    def __init__(self, vocab_group, **kwargs):
        super().__init__(**kwargs)
        if (
            vocab_group.size() != 1
            or self.cp_size != 1
            or self.enable_seq_packing
            or need_top_k_or_top_p_filtering(self.sampling_params)
        ):
            raise ValueError(
                "CUA requires TP=CP=1, unpacked input and unfiltered training"
            )
        self.vocab_group = vocab_group

    def __call__(
        self,
        logits,
        data_dict,
        processed_inputs,
        global_valid_seqs,
        global_valid_toks,
        **kwargs,
    ):
        # Temperature was already applied by the native forward helper. Keep
        # logits in BF16; the native kernel upcasts and recomputes 256 positions
        # at a time instead of retaining a full [sequence, vocabulary] FP32 tensor.
        logprobs = ChunkedDistributedLogprob.apply(
            logits[:, :-1],
            data_dict["input_ids"][:, 1:],
            0,
            logits.shape[-1],
            256,
            self.vocab_group,
            not logits.requires_grad,
        )
        return self.loss_fn(
            next_token_logprobs=logprobs,
            data=data_dict,
            global_valid_seqs=global_valid_seqs,
            global_valid_toks=global_valid_toks,
        )


class LoGRAWorkerImpl(AutomodelPolicyWorkerImpl):
    optimizer: torch.optim.Optimizer | None

    def __init__(
        self,
        config,
        weights_path=None,
        optimizer_path=None,
        init_optimizer=True,
        init_reference_model=True,
        **kwargs,
    ):
        draft = coerce_draft_config(config.get("draft"))
        if draft is not None and draft.enabled:
            raise ValueError("LoGRA does not support draft co-training")
        self.logra_config = LoGRAConfig.model_validate(config["logra"])
        self.cua_config = (
            CuaConfig.model_validate(config["cua"])
            if config.get("cua") is not None
            else None
        )
        self.cua_update: CuaUpdate | None = None
        automodel = config["automodel_cfg"]
        if self.cua_config is not None:
            if (
                not self.logra_config.enabled
                or self.logra_config.optimizer != "sgd"
                or not self.logra_config.freeze_non_target
            ):
                raise ValueError(
                    "CUA requires LoGRA SGD with frozen non-target parameters"
                )
            if config["optimizer"]["kwargs"]["weight_decay"] != 0:
                raise ValueError("CUA requires zero weight decay")
            if (
                config["train_micro_batch_size"] != 1
                or config["sequence_packing"]["enabled"]
                or config["dynamic_batching"]["enabled"]
            ):
                raise ValueError(
                    "CUA requires un-packed fixed microbatches of size one"
                )
        if (
            automodel["tensor_parallel_size"] != 1
            or automodel["context_parallel_size"] != 1
        ):
            raise ValueError("LoGRA initially supports TP=CP=1")
        if config.get("cpu_offload") or automodel.get("cpu_offload"):
            raise ValueError("CPU offload is not validated for this research worker")
        # Delay all restore until the project optimizer exists. Reference capture
        # therefore remains anchored to the pretrained model, including on resume.
        super().__init__(
            config,
            weights_path=None,
            optimizer_path=None,
            init_optimizer=init_optimizer,
            init_reference_model=init_reference_model,
            **kwargs,
        )
        if self.logra_config.enabled and init_optimizer:
            if self.lora_enabled:
                raise ValueError("LoGRA does not use PEFT adapters")
            if self.cua_config is None:
                if self.is_moe_model or not automodel["automodel_kwargs"].get(
                    "force_hf"
                ):
                    raise ValueError("The GRPO recipe requires dense HF linear layers")
            elif self.is_moe_model and (
                self.moe_mesh is None or self.moe_mesh["ep"].size() != self.dp_size
            ):
                raise ValueError("CUA requires EP across all training ranks (TP=CP=1)")
            if self.optimizer is None:
                raise RuntimeError("The native worker did not initialize an optimizer")
            # Native AdamW is lazy: no full moments have been allocated yet.
            # Undo scheduler initialization before building the replacement schedule.
            for group in self.optimizer.param_groups:
                group["lr"] = config["optimizer"]["kwargs"]["lr"]
                group.pop("initial_lr", None)
            self.optimizer = LoGRAOptimizer(
                self.model, self.optimizer, self.logra_config
            )
            self.scheduler = build_scheduler(self.optimizer, config.get("scheduler"))
        if weights_path:
            self.load_checkpoint(weights_path, optimizer_path)
        if self.cua_config is not None:
            self.cua_update = CuaUpdate(self, self.cua_config)

    def logra_manifest(self, sample_weights=False):
        """Record targets; optional tiny expert samples support checkpoint audits."""
        if not isinstance(self.optimizer, LoGRAOptimizer):
            raise ValueError("No LoGRA optimizer is installed")
        entries = []
        for layer in self.optimizer.layers:
            entry = {
                "name": layer.name,
                "shape": list(layer.weight.shape),
                "dtype": str(layer.weight.dtype),
                "rank": layer.projection.shape[0],
            }
            if sample_weights and layer.weight.ndim == 3:
                local, offset = local_weight_rows(layer.weight)
                rows = [0, local.shape[0] // 2, local.shape[0] - 1]
                columns = [0, local.shape[1] // 2, local.shape[1] - 1]
                entry["sample_rows"] = [offset + row for row in rows]
                entry["sample_columns"] = columns
                entry["sample_values"] = (
                    local.detach()[rows][:, columns].float().tolist()
                )
            entries.append(entry)
        return entries

    def record_memory(self, metrics):
        # GRPO sums unknown all_mb_metrics fields across workers. Divide by DP
        # size explicitly to log the mean per-GPU peak, not their sum.
        values = {
            "mean_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "mean_allocated_after_update_gib": torch.cuda.memory_allocated() / 2**30,
            "mean_peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        for name, value in values.items():
            metrics["all_mb_metrics"]["memory/" + name] = [value / self.dp_size]
        worst = torch.tensor(values["mean_peak_allocated_gib"], device="cuda")
        torch.distributed.all_reduce(
            worst, op=torch.distributed.ReduceOp.MAX, group=self.dp_mesh.get_group()
        )
        metrics["all_mb_metrics"]["memory/max_peak_allocated_gib"] = [
            worst.item() / self.dp_size
        ]

    def move_optimizer_to_device(self, device):
        super().move_optimizer_to_device(device)
        if isinstance(self.optimizer, LoGRAOptimizer):
            for state in self.optimizer.native.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)

    def save_checkpoint(
        self,
        weights_path,
        optimizer_path=None,
        tokenizer_path=None,
        is_final_checkpoint=False,
    ):
        if not isinstance(self.optimizer, LoGRAOptimizer):
            return super().save_checkpoint(
                weights_path,
                optimizer_path,
                tokenizer_path,
                is_final_checkpoint=is_final_checkpoint,
            )
        super().save_checkpoint(
            weights_path, None, tokenizer_path, is_final_checkpoint=is_final_checkpoint
        )
        if optimizer_path:
            os.makedirs(optimizer_path, exist_ok=True)
            torch.save(
                {
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "world_size": self.dp_size,
                },
                os.path.join(optimizer_path, f"logra-rank{self.rank}.pt"),
            )

    def load_checkpoint(self, weights_path, optimizer_path=None):
        if not isinstance(self.optimizer, LoGRAOptimizer):
            return super().load_checkpoint(weights_path, optimizer_path)
        super().load_checkpoint(weights_path, None)
        if optimizer_path:
            state = torch.load(
                os.path.join(optimizer_path, f"logra-rank{self.rank}.pt"),
                map_location="cpu",
                weights_only=False,
            )
            if state["world_size"] != self.dp_size:
                raise ValueError(
                    "LoGRA resume currently requires the same DP world size"
                )
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])
        else:
            self.optimizer.reset_master_weights()

    # The native NVTX decorator erases the parent signature for static checkers.
    def train(  # pyrefly: ignore [bad-override]
        self,
        data: BatchedDataDict[Any],
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
        check_dim_skip_keys: Optional[Iterable[str]] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        if not self.logra_config.enabled:
            torch.cuda.reset_peak_memory_stats()
            metrics = super().train(
                data, loss_fn, eval_mode, gbs, mbs, check_dim_skip_keys
            )
            self.record_memory(metrics)
            return metrics
        self.timer.start("train")
        torch.cuda.reset_peak_memory_stats()
        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=self.dp_mesh.get_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        # Validate sequence dimension
        sequence_dim, _ = check_sequence_dim(data, skip_keys=check_dim_skip_keys)

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        # Create loss post-processor
        post_processor = (
            partial(CuaLossPostProcessor, vocab_group=self.tp_mesh.get_group())
            if self.cua_update is not None
            else LossPostProcessor
        )
        loss_post_processor = post_processor(
            loss_fn=loss_fn,
            cfg=self.cfg,
            cp_mesh=self.cp_mesh,
            cp_size=self.cp_size,
            dp_size=self.dp_size,
            enable_seq_packing=self.enable_seq_packing,
            sampling_params=self.sampling_params,
        )

        # Setup cache clearing callback if configured
        empty_cache_steps = self.cfg.get("automodel_cfg", {}).get(
            "clear_cache_every_n_steps"
        )
        if empty_cache_steps:
            warnings.warn(
                f"Emptying cache every {empty_cache_steps} microbatches; doing so unnecessarily would incur a large performance overhead.",
            )

        def on_microbatch_start(mb_idx):
            if empty_cache_steps and mb_idx % empty_cache_steps == 0:
                torch.cuda.empty_cache()

        with ctx:
            # Get data from batch and move to device
            data = data.to("cuda")

            losses = []
            all_mb_metrics = []
            for gb_idx in range(num_global_batches):
                # Process global batch and compute normalization factors
                gb_result = process_global_batch(
                    data,
                    loss_fn,
                    self.dp_mesh.get_group(),
                    batch_idx=gb_idx,
                    batch_size=local_gbs,
                )
                batch = gb_result["batch"]
                global_valid_seqs = gb_result["global_valid_seqs"]
                global_valid_toks = gb_result["global_valid_toks"]

                self.optimizer.zero_grad()

                # Get microbatch iterator based on batching strategy
                processed_iterator, iterator_len = get_microbatch_iterator(
                    batch,
                    cast(dict[str, Any], self.cfg),
                    mbs,
                    self.dp_mesh,
                    tokenizer=self.tokenizer,
                )

                batches_context: AbstractContextManager[
                    Iterator[ProcessedMicrobatch]
                ] = (
                    self.cua_update.training_batches(processed_iterator)
                    if self.cua_update is not None
                    else nullcontext[Iterator[ProcessedMicrobatch]](processed_iterator)
                )
                with batches_context as training_iterator:
                    mb_results = automodel_forward_backward(
                        model=self.model,
                        data_iterator=training_iterator,
                        post_processing_fn=loss_post_processor,
                        device_mesh=self.device_mesh,
                        padding_token_id=self.tokenizer.pad_token_id or 0,
                        autocast_context_factory=self._autocast_context,
                        forward_only=eval_mode,
                        is_reward_model=self._is_reward_model,
                        allow_flash_attn_args=self.allow_flash_attn_args,
                        global_valid_seqs=global_valid_seqs,
                        global_valid_toks=global_valid_toks,
                        sampling_params=self.sampling_params,
                        sequence_dim=sequence_dim,
                        dp_size=self.dp_size,
                        cp_size=self.cp_size,
                        num_global_batches=num_global_batches,
                        num_valid_microbatches=iterator_len,
                        on_microbatch_start=on_microbatch_start,
                        router_replay_gates=(
                            self._router_replay_gates
                            if self.cua_update is None
                            else None
                        ),
                    )

                # Extract losses and metrics from results
                mb_losses = []
                for mb_idx, (loss, loss_metrics) in enumerate(mb_results):
                    # Only process valid (non-dummy) batches for metrics
                    if mb_idx < iterator_len:
                        num_valid_samples = loss_metrics["num_valid_samples"]
                        loss_metrics[LEARNING_RATE_KEY] = self.optimizer.param_groups[
                            0
                        ]["lr"]
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()

                        if num_valid_samples > 0:
                            mb_losses.append(loss_metrics["loss"] * num_global_batches)
                            all_mb_metrics.append(loss_metrics)

                grad_norm: torch.Tensor | None = None
                if not eval_mode:
                    if isinstance(self.optimizer, LoGRAOptimizer):
                        raw_grad_norm = self.optimizer.synchronize_and_clip(
                            self.dp_mesh.get_group(),
                            None if self.cua_update is not None else self.max_grad_norm,
                        )
                    else:
                        raw_grad_norm = scale_grads_and_clip_grad_norm(
                            self.max_grad_norm,
                            [self.model],
                            norm_type=2.0,
                            pp_enabled=False,
                            device_mesh=self.device_mesh,
                            moe_mesh=None,
                            ep_axis_name=None,
                            pp_axis_name=None,
                            foreach=True,
                            num_label_tokens=1,
                            dp_group_size=self.dp_size,
                        )
                    grad_norm = torch.as_tensor(raw_grad_norm, device="cpu")
                    warn_if_inf_grad_norm(grad_norm)
                    if self.cua_update is not None:
                        assert isinstance(self.optimizer, LoGRAOptimizer)
                        step_metrics = self.cua_update.step(
                            self.optimizer, raw_grad_norm
                        )
                        all_mb_metrics.append(
                            {
                                "cua/" + key: value
                                / (self.dp_size * num_global_batches)
                                for key, value in step_metrics.items()
                            }
                        )
                    else:
                        self.optimizer.step()
                    self._update_moe_gate_bias_if_supported()

                losses.append(torch.tensor(mb_losses).sum().item())

            # release gradient memory before rollouts
            self.optimizer.zero_grad()
            # increment scheduler after all batches in rollout are processed
            if not eval_mode:
                self.scheduler.step()
            # dynamic batch and sequence dims causes alot of fragmentation, so clear
            # the memory allocator before moving on
            torch.cuda.empty_cache()

            # Aggregate training statistics across microbatches and ranks
            metrics = aggregate_training_statistics(
                losses=losses,
                all_mb_metrics=all_mb_metrics,
                grad_norm=grad_norm,
                dp_group=self.dp_mesh.get_group(),
                dtype=self.dtype,
            )

            self.record_memory(metrics)
            self.timer.stop("train")
            return metrics


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("automodel_policy_worker")
)  # pragma: no cover
class LoGRAWorker(LoGRAWorkerImpl):
    pass
