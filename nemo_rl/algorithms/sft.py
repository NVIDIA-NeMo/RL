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
import os
import warnings
from pathlib import Path
from typing import Optional, TypedDict

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.loss_functions import (
    LossFunction,
    NLLLoss,
    PreferenceLoss,
)
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import (
    AllTaskProcessedDataset,
    preference_collate_fn,
    rl_collate_fn,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import PolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import Timer


class SFTSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    val_loss: float
    consumed_samples: int


def _default_sft_save_state() -> SFTSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
    }


class SFTConfig(TypedDict):
    max_num_steps: int
    max_num_epochs: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int


class MasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    sft: SFTConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


class SFTValMetrics(TypedDict):
    val_loss: float


class RMValMetrics(TypedDict):
    val_loss: float
    accuracy: float
    rewards_chosen_mean: float
    rewards_rejected_mean: float
    num_valid_samples: float


# =======================================================
# Setup & Initialization
# =======================================================
def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: AllTaskProcessedDataset,
) -> tuple[
    Policy,
    RayVirtualCluster,
    StatefulDataLoader,
    StatefulDataLoader,
    LossFunction,
    MasterConfig,
    Logger,
    TaskDataSpec,
    SFTSaveState,
]:
    """Main entry point for running SFT algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, master_config, logger
    """
    model_type = "reward" if "reward_model_type" in master_config["policy"] else "lm"

    if model_type == "lm":
        collate_fn = rl_collate_fn
        loss_fn_class = NLLLoss
    elif model_type == "reward":
        collate_fn = preference_collate_fn
        loss_fn_class = PreferenceLoss
    else:
        raise NotImplementedError(
            f"Model type {model_type} not implemented for SFT training."
        )

    set_seed(master_config["sft"]["seed"])

    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    sft_config = master_config["sft"]

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config)

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(master_config["checkpointing"])
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    sft_save_state: Optional[SFTSaveState] = checkpointer.load_training_info(
        last_checkpoint_path
    )

    # ==========================
    #           Data
    # ==========================
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state_dict)

    val_dataloader = StatefulDataLoader(
        val_dataset,
        batch_size=sft_config["val_global_batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="sft_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]]
        * cluster_config["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
    )
    print(f"  ✓ Ray cluster initialized with {cluster_config['num_nodes']} nodes")

    # ==========================
    #   Training
    # ==========================
    print("\n▶ Setting up model...")
    policy = Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        weights_path=Path(last_checkpoint_path) / "policy" / "weights"
        if last_checkpoint_path
        else None,
        optimizer_path=Path(last_checkpoint_path) / "policy" / "optimizer"
        if last_checkpoint_path
        else None,
        init_optimizer=True,
        init_reference_model=False,
    )
    loss_fn = loss_fn_class()
    print("  ✓ Model initialized")

    print("\n" + "=" * 60)
    print(" " * 18 + "SETUP COMPLETE")
    print("=" * 60 + "\n")

    return (
        policy,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        sft_save_state,
        master_config,
    )


# =======================================================
# Training & Validation
# =======================================================
def validate(
    policy: PolicyInterface,
    val_dataloader: StatefulDataLoader,
    tokenizer,
    loss_fn,
    step: int,
    master_config: MasterConfig,
    sft_task_spec: TaskDataSpec,
    val_batches: int,
    val_batch_size: int,
    val_mbs: int,
    model_type: str,
):
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return

    timer = Timer()

    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        # Show a progress indicator for validation
        # val_total = len(val_dataloader)

        list_of_val_metrics = []

        num_valid_batches = 0

        policy.prepare_for_training()
        for batch_idx, val_batch in enumerate(val_dataloader):
            ## add loss mask based on role to every message
            add_loss_mask_to_message_log(
                val_batch["message_log"],
                roles_to_train_on=["assistant"],
            )

            cat_and_padded, input_lengths = batched_message_log_to_flat_message(
                val_batch["message_log"],
                pad_value_dict={"token_ids": tokenizer.pad_token_id},
                make_sequence_length_divisible_by=master_config["policy"][
                    "make_sequence_length_divisible_by"
                ],
            )

            val_data: BatchedDataDict = BatchedDataDict(
                {
                    "input_ids": cat_and_padded["token_ids"],
                    "input_lengths": input_lengths,
                    "token_mask": cat_and_padded["token_loss_mask"],
                    "sample_mask": val_batch["loss_multiplier"],
                }
            )

            ## just run model fwd

            if model_type == "lm":
                val_results = policy.train(
                    val_data,
                    loss_fn,
                    eval_mode=True,
                    gbs=val_batch_size,
                    mbs=val_mbs,
                )
            elif model_type == "reward":
                val_results = policy.train(
                    val_data,
                    loss_fn,
                    eval_mode=True,
                    ## NOTE: we double the batch size here because each preference example corresponds to a pair of
                    ## examples, chosen and rejected, and the pair needs to be processed as part of the same microbatch.
                    gbs=val_batch_size * 2,
                    mbs=val_mbs * 2,
                )
            else:
                raise NotImplementedError(
                    f"Model type {model_type} not implemented for SFT training."
                )

            if len(val_results["all_mb_metrics"]) == 0:
                warnings.warn(
                    "No validation metrics were collected for this batch."
                    " This is likely because there were no valid samples."
                )
            else:
                if model_type == "lm":
                    list_of_val_metrics.append(
                        SFTValMetrics(val_loss=float(val_results["loss"]))
                    )
                elif model_type == "reward":
                    list_of_val_metrics.append(
                        RMValMetrics(
                            val_loss=sum(val_results["all_mb_metrics"]["loss"]),
                            accuracy=sum(val_results["all_mb_metrics"]["accuracy"]),
                            rewards_chosen_mean=sum(
                                val_results["all_mb_metrics"]["rewards_chosen_mean"]
                            ),
                            rewards_rejected_mean=sum(
                                val_results["all_mb_metrics"]["rewards_rejected_mean"]
                            ),
                            num_valid_samples=sum(
                                val_results["all_mb_metrics"]["num_valid_samples"]
                            ),
                        )
                    )
                else:
                    raise NotImplementedError(
                        f"Model type {model_type} not implemented for SFT training."
                    )

                num_valid_batches += 1

            if val_batches > 0 and batch_idx >= val_batches - 1:
                break

        if num_valid_batches > 0:
            if model_type == "lm":
                val_metrics = SFTValMetrics(
                    val_loss=sum([m["val_loss"] for m in list_of_val_metrics])
                )
                val_metrics["val_loss"] /= num_valid_batches
            elif model_type == "reward":
                sum_num_valid_samples = sum(
                    [m["num_valid_samples"] for m in list_of_val_metrics]
                )
                val_metrics = RMValMetrics(
                    val_loss=sum(
                        [
                            m["val_loss"] * m["num_valid_samples"]
                            for m in list_of_val_metrics
                        ]
                    )
                    / sum_num_valid_samples,
                    accuracy=sum(
                        [
                            m["accuracy"] * m["num_valid_samples"]
                            for m in list_of_val_metrics
                        ]
                    )
                    / sum_num_valid_samples,
                    rewards_chosen_mean=sum(
                        [
                            m["rewards_chosen_mean"] * m["num_valid_samples"]
                            for m in list_of_val_metrics
                        ]
                    )
                    / sum_num_valid_samples,
                    rewards_rejected_mean=sum(
                        [
                            m["rewards_rejected_mean"] * m["num_valid_samples"]
                            for m in list_of_val_metrics
                        ]
                    )
                    / sum_num_valid_samples,
                    num_valid_samples=sum_num_valid_samples,
                )
        else:
            warnings.warn(
                "No validation metrics were collected."
                " This is likely because there were no valid samples in the validation set."
            )
            if model_type == "lm":
                val_metrics = SFTValMetrics(val_loss=0.0)
            elif model_type == "reward":
                val_metrics = RMValMetrics(
                    val_loss=0.0,
                    accuracy=0.0,
                    rewards_chosen_mean=0.0,
                    rewards_rejected_mean=0.0,
                    num_valid_samples=0.0,
                )

        # Calculate validation metrics
        policy.prepare_for_training()

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    if num_valid_batches > 0:
        # Print summary of validation results
        print("\n📊 Validation Results:")
        print(f"    • Validation loss: {val_metrics['val_loss']:.4f}")

        if model_type == "reward":
            print(f"    • Validation accuracy: {val_metrics['accuracy']:.4f}")
            print(
                f"    • Validation rewards chosen mean: {val_metrics['rewards_chosen_mean']:.4f}"
            )
            print(
                f"    • Validation rewards rejected mean: {val_metrics['rewards_rejected_mean']:.4f}"
            )
            print(
                f"    • Validation num valid samples: {val_metrics['num_valid_samples']:.0f}"
            )

        # Print timing information
        print("\n  ⏱️  Validation Timing:")
        validation_time = timing_metrics.get("total_validation_time", 0)
        print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics


def sft_train(
    policy,
    train_dataloader,
    val_dataloader,
    tokenizer,
    loss_fn,
    master_config,
    logger,
    sft_task_spec,
    checkpointer,
    sft_save_state,
):
    # Run basic sft training
    timer = Timer()

    if sft_save_state is None:
        sft_save_state = _default_sft_save_state()
        current_epoch = 0
        current_step = 0
        total_steps = 0
    else:
        current_epoch = sft_save_state["epoch"]
        current_step = sft_save_state["step"]
        total_steps = sft_save_state["total_steps"]

    sft_config = master_config["sft"]
    # Validation configuration
    val_period = sft_config["val_period"]
    val_at_start = sft_config["val_at_start"]
    max_num_epochs = sft_config["max_num_epochs"]

    model_type = "reward" if "reward_model_type" in master_config["policy"] else "lm"

    # Run validation at the start if configured
    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...")
        val_metrics, validation_timings = validate(
            policy,
            val_dataloader,
            tokenizer,
            loss_fn,
            step=0,
            master_config=master_config,
            sft_task_spec=sft_task_spec,
            val_batches=sft_config["val_batches"],
            val_batch_size=sft_config["val_global_batch_size"],
            val_mbs=sft_config["val_micro_batch_size"],
            model_type=model_type,
        )

        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    policy.prepare_for_training()

    while current_epoch < max_num_epochs and (
        master_config["sft"]["max_num_steps"] == -1
        or total_steps < master_config["sft"]["max_num_steps"]
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in train_dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), master_config['sft']['max_num_steps'] if master_config['sft']['max_num_steps'] != -1 else len(train_dataloader))} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # Prepare batch and generate responses
                print("▶ Preparing batch...")
                with timer.time("data_processing"):
                    ## add loss mask based on role to every message
                    add_loss_mask_to_message_log(
                        batch["message_log"],
                        roles_to_train_on=["assistant"],
                    )

                    cat_and_padded, input_lengths = batched_message_log_to_flat_message(
                        batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )

                    train_data: BatchedDataDict = BatchedDataDict(
                        {
                            "input_ids": cat_and_padded["token_ids"],
                            "input_lengths": input_lengths,
                            "token_mask": cat_and_padded["token_loss_mask"],
                            "sample_mask": batch["loss_multiplier"],
                        }
                    )

                print("▶ Taking a training step...")

                if model_type == "lm":
                    train_results = policy.train(train_data, loss_fn)
                elif model_type == "reward":
                    train_results = policy.train(
                        train_data,
                        loss_fn,
                        eval_mode=False,
                        ## NOTE: we double the batch size here because each preference example corresponds to a pair of
                        ## examples, chosen and rejected, and the pair needs to be processed as part of the same microbatch.
                        gbs=master_config["policy"]["train_global_batch_size"] * 2,
                        mbs=master_config["policy"]["train_micro_batch_size"] * 2,
                    )
                else:
                    raise NotImplementedError(
                        f"Model type {model_type} not implemented for SFT training."
                    )

                is_last_step = (
                    master_config["sft"]["max_num_steps"] != -1
                    and total_steps + 1 >= master_config["sft"]["max_num_steps"]
                ) or (
                    current_epoch + 1 == max_num_epochs
                    and current_step + 1 == len(train_dataloader)
                )

                # Run validation if it's a validation step
                if val_period > 0 and (total_steps + 1) % val_period == 0:
                    val_metrics, validation_timings = validate(
                        policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps + 1,
                        master_config=master_config,
                        sft_task_spec=sft_task_spec,
                        val_batches=sft_config["val_batches"],
                        val_batch_size=sft_config["val_global_batch_size"],
                        val_mbs=sft_config["val_micro_batch_size"],
                        model_type=model_type,
                    )
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )

                ## Checkpointing
                sft_save_state["consumed_samples"] += master_config["policy"][
                    "train_global_batch_size"
                ]
                if master_config["checkpointing"]["enabled"] and (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                ):
                    ## +1 because step is 0-indexed
                    sft_save_state["step"] = (current_step + 1) % len(train_dataloader)
                    sft_save_state["total_steps"] = total_steps + 1
                    sft_save_state["epoch"] = current_epoch
                    if val_metrics is not None:
                        sft_save_state["val_loss"] = val_metrics["val_loss"]
                    elif "val_loss" in sft_save_state:
                        del sft_save_state["val_loss"]

                    if master_config["checkpointing"]["metric_name"] is not None:
                        if (
                            master_config["checkpointing"]["metric_name"]
                            not in sft_save_state
                        ):
                            warnings.warn(
                                f"You asked to save checkpoints based on {master_config['checkpointing']['metric_name']} but the metric is not found in the save state. "
                                "Saving most recent k checkpoints instead."
                            )
                            master_config["checkpointing"]["metric_name"] = None

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {total_steps + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, sft_save_state, master_config
                        )

                        policy.save_checkpoint(
                            weights_path=os.path.join(
                                checkpoint_path, "policy", "weights"
                            ),
                            optimizer_path=os.path.join(
                                checkpoint_path, "policy", "optimizer"
                            ),
                            tokenizer_path=os.path.join(
                                checkpoint_path, "policy", "tokenizer"
                            ),
                        )
                        torch.save(
                            train_dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            losses = train_results["loss"]
            metrics = {
                "loss": train_results["loss"].numpy(),
                "grad_norm": train_results["grad_norm"].numpy(),
            }

            metrics.update(train_results["all_mb_metrics"])
            for k, v in metrics.items():
                if k in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                    metrics[k] = np.mean(v).item()
                else:
                    metrics[k] = np.sum(v).item()
            timing_metrics = timer.get_timing_metrics(reduction_op="sum")

            print("\n📊 Training Results:")
            print(f"  • Loss: {float(metrics['loss']):.4f}")

            if model_type == "reward":
                print(f"  • Accuracy: {float(metrics['accuracy']):.4f}")
                print(
                    f"  • Rewards chosen mean: {float(metrics['rewards_chosen_mean']):.4f}"
                )
                print(
                    f"  • Rewards rejected mean: {float(metrics['rewards_rejected_mean']):.4f}"
                )
                print(
                    f"  • Num valid samples: {float(metrics['num_valid_samples']):.0f}"
                )

            print("\n⏱️  Timing:")
            # Display total time first, separately
            total_time = timing_metrics.get("total_step_time", 0)
            print(f"  • Total step time: {total_time:.2f}s")

            # Display all other timing metrics (if any)
            for k, v in sorted(
                timing_metrics.items(), key=lambda item: item[1], reverse=True
            ):
                if k != "total_step_time":
                    percent = (v / total_time * 100) if total_time > 0 else 0
                    print(f"  • {k}: {v:.2f}s ({percent:.1f}%)")

            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1

            if (
                master_config["sft"]["max_num_steps"] != -1
                and total_steps >= master_config["sft"]["max_num_steps"]
            ):
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch
