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
"""On-policy distillation trainer using TransferQueue for teacher top-k."""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    from nemo_rl.models.policy.tq_policy import TQPolicy

import numpy as np
import ray
import torch
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.distillation import (
    DistillationSaveState,
    MasterConfig,
    TokenizerType,
    validate,
)
from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.loss import DistillationLossFn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.data.multimodal_utils import (
    PackedTensor,
    get_multimodal_keys_from_processor,
)
from nemo_rl.data_plane.column_io import stamp_global_forward_pad_seqlen
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.preshard import shard_meta_for_dp
from nemo_rl.data_plane.schema import (
    DISTILLATION_TRAIN_FIELDS,
    TEACHER_TOPK_SEED_FIELDS,
)
from nemo_rl.data_plane.transport_metrics import add_byte_metric_derivatives
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.distillation_rollout_actor import DistillationRolloutActor
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer
from nemo_rl.utils.venvs import make_actor_runtime_env


def _dedupe_fields(*field_groups: list[str] | tuple[str, ...]) -> list[str]:
    """Concatenate field groups while preserving first occurrence order."""
    out: list[str] = []
    seen: set[str] = set()
    for group in field_groups:
        for field in group:
            if field in seen:
                continue
            seen.add(field)
            out.append(field)
    return out


def _as_row_aligned_tensor(value: Any, expected_rows: int) -> Optional[torch.Tensor]:
    """Return a tensor only when TQ can store it under row-addressed keys."""
    if isinstance(value, PackedTensor):
        value = value.as_tensor()
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 0 or value.shape[0] != expected_rows:
        return None
    return value


def derive_model_input_fields(
    prompt_batch: BatchedDataDict[DatumSpec],
    tokenizer: TokenizerType,
    master_config: MasterConfig,
) -> list[str]:
    """Derive positive model-input extras before partition registration."""
    flat, _ = batched_message_log_to_flat_message(
        prompt_batch["message_log"],
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
        make_sequence_length_divisible_by=master_config.policy[
            "make_sequence_length_divisible_by"
        ],
    )
    expected_rows = int(prompt_batch.size)
    processor_fields = set(get_multimodal_keys_from_processor(tokenizer))
    optional_tensor_fields = set(BatchedDataDict.ADDITIONAL_OPTIONAL_KEY_TENSORS)
    fields: list[str] = []
    for field, value in flat.items():
        if (
            field not in processor_fields
            and field not in optional_tensor_fields
            and not isinstance(value, PackedTensor)
        ):
            continue
        if _as_row_aligned_tensor(value, expected_rows) is not None:
            fields.append(field)
    return fields


def _packing_args_for_policy(
    policy: ColocatablePolicyInterface,
    mb_tokens_key: str,
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Resolve packing args for a plain Policy-like object."""
    if getattr(policy, "use_dynamic_batches", False):
        args = dict(policy.dynamic_batching_args)  # type: ignore[attr-defined]
        args["max_tokens_per_microbatch"] = policy.cfg["dynamic_batching"][  # type: ignore[attr-defined]
            mb_tokens_key
        ]
        return None, args
    if getattr(policy, "use_sequence_packing", False):
        args = dict(policy.sequence_packing_args)  # type: ignore[attr-defined]
        args["max_tokens_per_microbatch"] = policy.cfg["sequence_packing"][  # type: ignore[attr-defined]
            mb_tokens_key
        ]
        return args, None
    return None, None


def _stamp_policy_pad_seqlen(
    policy: ColocatablePolicyInterface,
    meta: KVBatchMeta,
    *,
    mb_tokens_key: str,
) -> None:
    """Stamp the forward pad target for a plain Policy-like dispatch."""
    _, dba = _packing_args_for_policy(policy, mb_tokens_key)
    sequence_length_round = int(dba["sequence_length_round"]) if dba is not None else 1
    stamp_global_forward_pad_seqlen(
        meta,
        sequence_length_round=sequence_length_round,
    )


def attach_policy_workers_to_data_plane(
    policy: ColocatablePolicyInterface,
    dp_cfg: dict[str, Any],
) -> None:
    """Attach plain policy workers to an already bootstrapped data plane."""
    ray.get(
        policy.worker_group.run_all_workers_single_data(  # type: ignore[attr-defined]
            "setup_data_plane", cfg=dp_cfg
        )
    )


def _aggregate_teacher_topk_transport_results(
    results: list[Any],
) -> dict[str, float | int]:
    """Aggregate scalar teacher top-k transport acks from workers."""
    transport_metrics: dict[str, float | int] = {
        "teacher_topk_payload_bytes": 0,
        "teacher_topk_valid_payload_bytes": 0,
        "teacher_topk_padding_overhead_bytes": 0,
        "driver_rx_teacher_topk_bytes": 0,
        "driver_tx_teacher_topk_bytes": 0,
        "driver_teacher_topk_bytes": 0,
        "driver_teacher_topk_bytes_avoided": 0,
        "tq_teacher_topk_write_bytes": 0,
        "tq_teacher_topk_write_num_samples": 0,
        "tq_teacher_topk_write_ms_sum": 0.0,
        "tq_teacher_topk_write_ms_max": 0.0,
    }
    for result in results:
        if result in ({}, None):
            continue
        if not isinstance(result, Mapping):
            raise RuntimeError(
                "teacher top-k writeback path must return scalar metric acks; "
                f"got {type(result).__name__}."
            )
        tensor_keys = [k for k, v in result.items() if isinstance(v, torch.Tensor)]
        if tensor_keys:
            raise RuntimeError(
                "teacher top-k writeback path returned tensor payloads for "
                f"{tensor_keys}; expected scalar metric acks only."
            )
        transport_metrics["teacher_topk_payload_bytes"] += int(
            result.get("teacher_topk_payload_bytes", 0)
        )
        transport_metrics["teacher_topk_valid_payload_bytes"] += int(
            result.get("teacher_topk_valid_payload_bytes", 0)
        )
        transport_metrics["teacher_topk_padding_overhead_bytes"] += int(
            result.get("teacher_topk_padding_overhead_bytes", 0)
        )
        transport_metrics["tq_teacher_topk_write_bytes"] += int(
            result.get("tq_teacher_topk_write_bytes", 0)
        )
        transport_metrics["tq_teacher_topk_write_num_samples"] += int(
            result.get("tq_teacher_topk_write_num_samples", 0)
        )
        write_ms = float(result.get("tq_teacher_topk_write_ms", 0.0))
        transport_metrics["tq_teacher_topk_write_ms_sum"] += write_ms
        transport_metrics["tq_teacher_topk_write_ms_max"] = max(
            float(transport_metrics["tq_teacher_topk_write_ms_max"]),
            write_ms,
        )

    payload_bytes = int(transport_metrics["teacher_topk_payload_bytes"])
    transport_metrics["driver_teacher_topk_bytes_avoided"] = 2 * payload_bytes
    return transport_metrics


def dispatch_teacher_topk_writeback(
    teacher_policy: ColocatablePolicyInterface,
    meta: KVBatchMeta,
    *,
    fields: list[str],
    k: int,
    timer: Optional[Timer] = None,
) -> dict[str, float | int]:
    """Compute teacher top-k on workers and write it back to TQ only."""
    _stamp_policy_pad_seqlen(
        teacher_policy,
        meta,
        mb_tokens_key="logprob_mb_tokens",
    )
    spa, dba = _packing_args_for_policy(teacher_policy, "logprob_mb_tokens")
    teacher_meta = replace(meta, fields=list(fields), task_name="teacher_topk")

    with timer.time("teacher_topk/shard_meta") if timer else nullcontext():
        dp_metas, _ = shard_meta_for_dp(
            teacher_meta,
            dp_world=teacher_policy.sharding_annotations.get_axis_size(  # type: ignore[attr-defined]
                "data_parallel"
            ),
            batch_size=None,
            sequence_packing_args=spa,
            dynamic_batching_args=dba,
        )

    with timer.time("teacher_topk/submit_futures") if timer else nullcontext():
        futures = teacher_policy.worker_group.run_all_workers_sharded_data(  # type: ignore[attr-defined]
            "get_topk_logits_presharded",
            meta=dp_metas,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            output_is_replicated=[
                "context_parallel",
                "tensor_parallel",
                "pipeline_parallel",
            ],
            common_kwargs={"k": k},
        )
    results = teacher_policy.worker_group.get_all_worker_results(futures)  # type: ignore[attr-defined]
    return _aggregate_teacher_topk_transport_results(results)


def distillation_train_sync(
    student_policy: ColocatablePolicyInterface,
    teacher_policy: ColocatablePolicyInterface,
    student_generation: Optional[GenerationInterface],
    dataloader: StatefulDataLoader,
    val_dataloader: Optional[StatefulDataLoader],
    tokenizer: TokenizerType,
    loss_fn: DistillationLossFn,
    task_to_env: dict[str, EnvironmentInterface],
    val_task_to_env: Optional[dict[str, EnvironmentInterface]],
    logger: Logger,
    checkpointer: CheckpointManager,
    distillation_save_state: DistillationSaveState,
    master_config: MasterConfig,
) -> None:
    """Run on-policy distillation with teacher top-k resident in TQ."""
    dp_cfg = master_config.data_plane
    if not dp_cfg or not dp_cfg["enabled"]:
        raise ValueError(
            "distillation_train_sync requires data_plane.enabled=true. "
            "Use nemo_rl.algorithms.distillation.distillation_train for the "
            "legacy driver-mediated path."
        )
    if not hasattr(student_policy, "dp_cfg"):
        raise ValueError(
            "distillation_train_sync requires the student policy to be a "
            "TQPolicy constructed by examples/run_distillation.py."
        )
    student_tq_policy = cast("TQPolicy", student_policy)
    attach_policy_workers_to_data_plane(teacher_policy, dp_cfg)

    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    need_refit = True
    if student_generation is None:
        student_generation = student_policy  # type: ignore[assignment]
        need_refit = False
    policy_generation_stale = True
    assert student_generation is not None

    current_epoch = distillation_save_state["current_epoch"]
    current_step = distillation_save_state["current_step"]
    total_steps = distillation_save_state["total_steps"]
    consumed_samples = distillation_save_state["consumed_samples"]
    total_valid_tokens = distillation_save_state["total_valid_tokens"]
    val_period = master_config.distillation["val_period"]
    val_at_start = master_config.distillation["val_at_start"]
    val_at_end = master_config.distillation["val_at_end"]
    colocated_inference = master_config.policy["generation"]["colocated"]["enabled"]
    max_epochs = master_config.distillation["max_num_epochs"]
    max_steps = master_config.distillation["max_num_steps"]

    rollout_actor = DistillationRolloutActor.options(
        runtime_env=make_actor_runtime_env(
            "nemo_rl.experience.distillation_rollout_actor.DistillationRolloutActor"
        ),
    ).remote(
        policy_generation=student_generation,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        master_config=master_config,
        dp_cfg=dp_cfg,
    )

    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...", flush=True)
        if need_refit and policy_generation_stale:
            refit_policy_generation(
                student_policy, student_generation, colocated_inference
            )
            policy_generation_stale = False
        else:
            student_generation.prepare_for_generation()
        val_metrics, validation_timings = validate(
            student_generation,
            val_dataloader,
            tokenizer,
            val_task_to_env,
            step=total_steps,
            master_config=master_config,
        )
        student_generation.finish_generation()
        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    batch: BatchedDataDict[DatumSpec]
    while total_steps < max_steps and current_epoch < max_epochs:
        print(
            f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_epochs} {'=' * 25}",
            flush=True,
        )

        for batch in dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(dataloader), max_steps)} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(student_policy, total_steps + 1)
            if student_policy != student_generation:
                maybe_gpu_profile_step(student_generation, total_steps + 1)
            val_metrics, validation_timings = None, None
            should_save_by_timeout = False

            with timer.time("total_step_time"):
                print("▶ Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    repeated_batch: BatchedDataDict[DatumSpec] = (
                        batch.repeat_interleave(
                            master_config.distillation["num_generations_per_prompt"]
                        )
                    )
                    model_input_fields = derive_model_input_fields(
                        repeated_batch,
                        tokenizer,
                        master_config,
                    )
                    distillation_train_fields = _dedupe_fields(
                        DISTILLATION_TRAIN_FIELDS,
                        model_input_fields,
                    )
                    teacher_fetch_fields = _dedupe_fields(
                        TEACHER_TOPK_SEED_FIELDS,
                        model_input_fields,
                    )

                student_tq_policy.prepare_step(
                    num_samples=int(repeated_batch.size),
                    fields=distillation_train_fields,
                    consumer_tasks=["teacher_topk", "train"],
                )

                print(
                    f"▶ Generating responses for batch of size {repeated_batch.size}...",
                    flush=True,
                )
                with timer.time("prepare_for_generation"):
                    if need_refit and policy_generation_stale:
                        refit_policy_generation(
                            student_policy,
                            student_generation,
                            colocated_inference,
                            timer=timer,
                        )
                        policy_generation_stale = False
                    else:
                        student_generation.prepare_for_generation()

                with timer.time("generation"):
                    meta, driver_carry, rollout_metrics, _ = ray.get(
                        rollout_actor.rollout_to_tq.remote(
                            repeated_batch,
                            partition_id=student_tq_policy.tq_partition_id,
                            model_input_fields=model_input_fields,
                        )
                    )

                print("▶ Preparing for teacher logprob inference...", flush=True)
                with timer.time("teacher_logprob_inference_prep"):
                    teacher_policy.prepare_for_lp_inference()

                print("▶ Computing teacher top-k writeback...", flush=True)
                with timer.time("teacher_logprob_inference"):
                    transport_metrics = dispatch_teacher_topk_writeback(
                        teacher_policy,
                        meta,
                        fields=teacher_fetch_fields,
                        k=master_config.distillation["topk_logits_k"],
                        timer=timer,
                    )

                print("▶ Preparing for training...", flush=True)
                with timer.time("training_prep"):
                    teacher_policy.offload_after_refit()
                    student_policy.prepare_for_training()
                    policy_generation_stale = True

                print("▶ Training policy...", flush=True)
                with timer.time("policy_training"):
                    train_results = student_tq_policy.train_from_meta(
                        meta,
                        loss_fn,
                        timer=timer,
                        fields=distillation_train_fields,
                    )

                log_content = driver_carry["content"]
                log_input_lengths = driver_carry["input_lengths"]
                mean_prompt_length = driver_carry["length"]
                student_tq_policy.finish_step(meta)

                is_last_step = (total_steps + 1 >= max_steps) or (
                    (current_epoch + 1 == max_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    if need_refit and policy_generation_stale:
                        refit_policy_generation(
                            student_policy, student_generation, colocated_inference
                        )
                        policy_generation_stale = False
                    else:
                        student_generation.prepare_for_generation()
                    val_metrics, validation_timings = validate(
                        student_generation,
                        val_dataloader,
                        tokenizer,
                        val_task_to_env,
                        step=total_steps + 1,
                        master_config=master_config,
                    )
                    student_generation.finish_generation()
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "mean_prompt_length": mean_prompt_length.numpy(),
                    "total_num_tokens": log_input_lengths.numpy(),
                }
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {
                        "lr",
                        "wd",
                        "global_valid_seqs",
                        "global_valid_toks",
                        "mean_prompt_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()
                metrics.update(rollout_metrics)
                total_valid_tokens += metrics["global_valid_toks"]

                consumed_samples += master_config.distillation["num_prompts_per_step"]
                timeout.mark_iteration()
                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config.checkpointing["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                if master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    student_policy.prepare_for_training()

                    distillation_save_state["current_epoch"] = current_epoch
                    distillation_save_state["current_step"] = current_step + 1
                    distillation_save_state["total_steps"] = total_steps + 1
                    distillation_save_state["total_valid_tokens"] = total_valid_tokens
                    if val_metrics is not None:
                        distillation_save_state["val_reward"] = val_metrics["accuracy"]
                    elif "val_reward" in distillation_save_state:
                        del distillation_save_state["val_reward"]
                    distillation_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config.checkpointing["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                            f"  If you are using an old config, please updated checkpointing.metric_name to the new format, "
                            f" e.g. 'val_reward --> 'val:accuracy'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in distillation_save_state:
                                del distillation_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            distillation_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(
                            f"Saving checkpoint for step {total_steps + 1}...",
                            flush=True,
                        )
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, distillation_save_state, master_config
                        )
                        student_policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            )
                            if checkpointer.save_optimizer
                            else None,
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                            checkpointing_cfg=master_config.checkpointing,
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            log_data = {"content": log_content.tolist()}
            log_data["input_lengths"] = log_input_lengths.tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{total_steps + 1}.jsonl"
            )

            timing_metrics: dict[str, float] = timer.get_timing_metrics(
                reduction_op="sum"
            )

            print("\n📊 Training Results:")
            print(f"  • Loss: {metrics['loss']:.4f}")
            print(
                f"  • Mean Generation Length: {rollout_metrics['mean_gen_tokens_per_sample']:.4f}"
            )
            if "total_flops" in train_results:
                total_tflops = (
                    train_results["total_flops"]
                    / timing_metrics["policy_training"]
                    / 1e12
                )
                num_ranks = train_results["num_ranks"]
                print(
                    f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)",
                    flush=True,
                )
                if "theoretical_tflops" in train_results:
                    theoretical_tflops = train_results["theoretical_tflops"]
                    print(
                        f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%",
                        flush=True,
                    )
                    metrics["train_fp_utilization"] = total_tflops / theoretical_tflops

            print("\n⏱️  Timing:", flush=True)
            total_time = timing_metrics.get("total_step_time", 0)
            total_num_gpus = (
                master_config.cluster["num_nodes"]
                * master_config.cluster["gpus_per_node"]
            )
            metrics.update(
                {
                    "tokens_per_sec_per_gpu": metrics["total_num_tokens"]
                    / total_time
                    / total_num_gpus
                }
            )
            print(f"  • Total step time: {total_time:.2f}s", flush=True)
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            add_byte_metric_derivatives(
                transport_metrics,
                token_count=metrics["total_num_tokens"],
            )
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")
            logger.log_metrics(transport_metrics, total_steps + 1, prefix="transport")

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= max_steps:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        current_epoch += 1
        current_step = 0
