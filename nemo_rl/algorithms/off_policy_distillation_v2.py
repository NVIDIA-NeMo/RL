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

"""Off-policy distillation v2: single dual-model Policy with GPU-local IPC transfer.

Unlike v1 (off_policy_distillation.py) which uses separate teacher/student Policies
and transfers logprobs through Ray's CPU object store, this version uses a single
Policy whose workers hold both teacher and student models. Teacher logprobs stay
on GPU via IPC buffers — no object store bottleneck.
"""

import os
import warnings
from typing import Any, Callable, Optional

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.loss_functions import DistillationLossFn
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.utils.checkpoint import CheckpointManager
from nemo_rl.utils.logger import Logger
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer


def off_policy_distillation_train_v2(
    policy: ColocatablePolicyInterface,
    dataloader: StatefulDataLoader,
    tokenizer: AutoTokenizer,
    loss_fn: DistillationLossFn,
    logger: Logger,
    checkpointer: CheckpointManager,
    distillation_save_state: dict,
    master_config: dict[str, Any],
    eval_hook: Optional[Callable] = None,
    eval_hook_period: int = 0,
    eval_hook_at_start: bool = False,
) -> None:
    """Off-policy distillation with a single dual-model Policy.

    The Policy's workers hold both teacher and student models. Each step:
    1. policy.teacher_forward(data, k) -- teacher logprobs stay on GPU via IPC
    2. policy.train(data, loss_fn, teacher_logits=handles) -- student reads GPU directly
    """
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    current_epoch = distillation_save_state["current_epoch"]
    current_step = distillation_save_state["current_step"]
    total_steps = distillation_save_state["total_steps"]
    consumed_samples = distillation_save_state["consumed_samples"]
    total_valid_tokens = distillation_save_state["total_valid_tokens"]
    max_epochs = master_config["distillation"]["max_num_epochs"]
    max_steps = master_config["distillation"]["max_num_steps"]

    eval_hook_metrics = None
    if eval_hook and eval_hook_at_start and total_steps == 0:
        print("\n Running initial eval hook...", flush=True)
        eval_hook_metrics = eval_hook(
            step=0, student_policy=policy, teacher_policy=policy, logger=logger,
        )
        if isinstance(eval_hook_metrics, dict):
            logger.log_metrics(eval_hook_metrics, 0, prefix="eval_hook")

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
            maybe_gpu_profile_step(policy, total_steps + 1)
            val_metrics = None

            with timer.time("total_step_time"):
                # ==== Data Processing ====
                print("Processing batch data (off-policy)...", flush=True)
                with timer.time("data_processing"):
                    for message_log in batch["message_log"]:
                        for message in message_log:
                            if "token_loss_mask" not in message:
                                if message["role"] == "assistant":
                                    message["token_loss_mask"] = torch.ones_like(
                                        message["token_ids"]
                                    )
                                else:
                                    message["token_loss_mask"] = torch.zeros_like(
                                        message["token_ids"]
                                    )

                    flat_messages, input_lengths = batched_message_log_to_flat_message(
                        batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    train_data = BatchedDataDict(
                        {
                            "input_ids": flat_messages["token_ids"],
                            "input_lengths": input_lengths,
                            "token_mask": flat_messages["token_loss_mask"],
                            "sample_mask": batch["loss_multiplier"],
                        }
                    )
                    train_data.update(
                        flat_messages.get_multimodal_dict(as_tensors=False)
                    )
                    train_data.to("cpu")

                # ==== Teacher Forward (GPU-local via IPC) ====
                # Each worker runs teacher forward and stores top-k logprobs
                # in GPU IPC buffers. No data leaves the GPU.
                print("Computing teacher logprobs (IPC)...", flush=True)
                with timer.time("teacher_forward"):
                    policy.teacher_forward(
                        train_data, k=master_config["distillation"]["topk_logits_k"]
                    )

                # ==== Student Training (reads from IPC buffers) ====
                # Each worker's train() reads self.teacher_logits from the
                # IPC buffer that teacher_forward() just populated.
                print("Training student policy...", flush=True)
                with timer.time("training_prep"):
                    policy.prepare_for_training()

                with timer.time("policy_training"):
                    train_results = policy.train(train_data, loss_fn)

                is_last_step = (total_steps + 1 >= max_steps) or (
                    (current_epoch + 1 == max_epochs)
                    and (current_step + 1 == len(dataloader))
                )

                # ==== Eval Hook ====
                if eval_hook and eval_hook_period > 0 and (total_steps + 1) % eval_hook_period == 0:
                    print(f"\n Running eval hook at step {total_steps + 1}...", flush=True)
                    with timer.time("eval_hook"):
                        eval_hook_metrics = eval_hook(
                            step=total_steps + 1,
                            student_policy=policy,
                            teacher_policy=policy,
                            logger=logger,
                        )
                    if isinstance(eval_hook_metrics, dict):
                        logger.log_metrics(eval_hook_metrics, total_steps + 1, prefix="eval_hook")
                    policy.prepare_for_training()

                # ==== Metrics ====
                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                    "mean_seq_length": batch["length"].numpy().mean(),
                    "total_num_tokens": input_lengths.numpy().sum(),
                }
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {
                        "lr", "wd", "global_valid_seqs",
                        "global_valid_toks", "mean_seq_length",
                    }:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()
                total_valid_tokens += metrics["global_valid_toks"]

                # ==== Checkpointing ====
                consumed_samples += master_config["distillation"]["num_prompts_per_step"]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"] == 0
                )
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    policy.prepare_for_training()

                    distillation_save_state["current_epoch"] = current_epoch
                    distillation_save_state["current_step"] = current_step + 1
                    distillation_save_state["total_steps"] = total_steps + 1
                    distillation_save_state["total_valid_tokens"] = total_valid_tokens
                    distillation_save_state["consumed_samples"] = consumed_samples

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith("train:") or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else eval_hook_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"Checkpointing metric {metric_name} requested but no {prefix} metrics collected.",
                                stacklevel=2,
                            )
                            if full_metric_name in distillation_save_state:
                                del distillation_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            distillation_save_state[full_metric_name] = metrics_source[metric_name]

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {total_steps + 1}...", flush=True)
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, distillation_save_state, master_config
                        )
                        policy.save_checkpoint(
                            weights_path=os.path.join(checkpoint_path, "policy", "weights"),
                            optimizer_path=os.path.join(checkpoint_path, "policy", "optimizer"),
                            tokenizer_path=os.path.join(checkpoint_path, "policy", "tokenizer"),
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            # ==== Logging ====
            log_data = {"content": flat_messages["content"]}
            log_data["input_lengths"] = input_lengths.tolist()
            logger.log_batched_dict_as_jsonl(
                log_data, f"train_data_step{total_steps + 1}.jsonl"
            )

            timing_metrics: dict[str, float] = timer.get_timing_metrics(reduction_op="sum")

            print("\n Training Results:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Grad Norm: {metrics['grad_norm']:.4f}")
            print(f"  Mean Sequence Length: {metrics['mean_seq_length']:.1f}")

            if "total_flops" in train_results:
                total_time = timing_metrics.get("total_step_time", 0)
                total_tflops = train_results["total_flops"] / timing_metrics["policy_training"] / 1e12
                num_ranks = train_results["num_ranks"]
                print(f"  Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} per rank)", flush=True)

            print("\n Timing:", flush=True)
            total_time = timing_metrics.get("total_step_time", 0)
            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            metrics["tokens_per_sec_per_gpu"] = metrics["total_num_tokens"] / total_time / total_num_gpus

            print(f"  Total step time: {total_time:.2f}s", flush=True)
            for k, v in sorted(timing_metrics.items(), key=lambda item: item[1], reverse=True):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  {k}: {v:.2f}s ({percent:.1f}%)", flush=True)

            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1
            if should_save_by_timeout:
                print("Timeout reached, stopping training early", flush=True)
                return
            if total_steps >= max_steps:
                print("Max steps reached, stopping training early", flush=True)
                return

        current_epoch += 1
        current_step = 0
