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
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Optional, TypedDict, cast

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.loss_functions import DPOLossFn
from nemo_rl.algorithms.utils import maybe_pad_last_batch, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import preference_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import PolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer


class DPOSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    consumed_samples: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training


def _default_dpo_save_state() -> DPOSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
        "total_valid_tokens": 0,
    }


class DPOConfig(TypedDict):
    max_num_epochs: int
    max_num_steps: int
    val_period: int
    val_batches: int
    val_global_batch_size: int
    val_micro_batch_size: int
    val_at_start: bool
    seed: int

    reference_policy_kl_penalty: float
    preference_average_log_probs: bool
    sft_average_log_probs: bool
    ## TODO(@ashors) support other loss functions
    ## https://github.com/NVIDIA-NeMo/RL/issues/193
    # preference_loss: str
    # gt_reward_scale: float
    preference_loss_weight: float
    sft_loss_weight: float


class MasterConfig(TypedDict):
    policy: PolicyConfig
    data: DataConfig
    dpo: DPOConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


class DPOValMetrics(TypedDict):
    loss: float
    sft_loss: float
    preference_loss: float
    accuracy: float
    rewards_chosen_mean: float
    rewards_rejected_mean: float
    num_valid_samples: float
    global_valid_seqs: float
    global_valid_toks: float


# =======================================================
# Setup & Initialization
# =======================================================
def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: dict[str, AllTaskProcessedDataset],
) -> tuple[
    Policy,
    RayVirtualCluster,
    StatefulDataLoader,
    dict[str, StatefulDataLoader],
    DPOLossFn,
    Logger,
    CheckpointManager,
    DPOSaveState,
    MasterConfig,
]:
    """Main entry point for running DPO algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, master_config, logger
    """
    # Make sure we are not using dynamic batching or sequence packing.
    # Anything that changes the order of data within a batch is currently incompatible with DPO.
    assert not master_config["policy"]["dynamic_batching"]["enabled"], (
        "Dynamic batching is currently not supported with DPO. "
        "See https://github.com/NVIDIA-NeMo/RL/issues/719"
    )
    assert not master_config["policy"]["sequence_packing"]["enabled"], (
        "Sequence packing is currently not supported with DPO. "
        "See https://github.com/NVIDIA-NeMo/RL/issues/719"
    )

    set_seed(master_config["dpo"]["seed"])

    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    dpo_config = master_config["dpo"]

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
    dpo_save_state: Optional[DPOSaveState] = cast(
        Optional[DPOSaveState], checkpointer.load_training_info(last_checkpoint_path)
    )

    # ==========================
    #           Data
    # ==========================
    ## TODO(@ashors) reduce boilerplate and move reused code into utils
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=data_config["shuffle"],
        collate_fn=partial(
            preference_collate_fn,
            tokenizer=tokenizer,
            make_sequence_length_divisible_by=policy_config[
                "make_sequence_length_divisible_by"
            ],
            add_loss_mask=True,
        ),
        drop_last=True,
        num_workers=data_config["num_workers"],
    )

    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state_dict)

    val_dataloader = {
        k: StatefulDataLoader(
            v,
            batch_size=dpo_config["val_global_batch_size"],
            shuffle=False,
            collate_fn=partial(
                preference_collate_fn,
                tokenizer=tokenizer,
                make_sequence_length_divisible_by=policy_config[
                    "make_sequence_length_divisible_by"
                ],
                add_loss_mask=True,
            ),
            drop_last=False,
            num_workers=data_config["num_workers"],
        )
        for k, v in val_dataset.items()
    }

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="dpo_cluster",
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
    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        total_train_iters = min(
            dpo_config["max_num_steps"],
            dpo_config["max_num_epochs"] * len(train_dataloader),
        )
        ## NOTE: we double the train_iters because effective batch size is doubled
        ## for (chosen, rejected) pairs
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters * 2
        if "scheduler" in policy_config["megatron_cfg"]:
            for k in policy_config["megatron_cfg"]["scheduler"]:
                if "iters" in k:
                    policy_config["megatron_cfg"]["scheduler"][k] *= 2

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
        init_reference_model=True,
    )
    # print the node IP and GPU ID of the policy workers for debugging
    policy.print_node_ip_and_gpu_id()

    loss_fn = DPOLossFn(master_config["dpo"])
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
        dpo_save_state,
        master_config,
    )


def add_ref_logprobs_to_data(dataloader, policy, master_config, is_val=False):
    dataloader_iter = iter(dataloader)
    while True:
        try:
            batch = next(dataloader_iter)

            micro_batch_size = (
                master_config["dpo"]["val_micro_batch_size"] * 2
                if is_val
                else master_config["policy"]["train_micro_batch_size"] * 2
            )

            # when running validation with drop_last=False, we might end up with a partial batch.
            # In this case, we pad the batch to the next multiple of micro_batch_size * dp_size.
            dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
            if batch.size % (dp_size * micro_batch_size) != 0:
                assert is_val, (
                    "Partial batches should only happen during validation, but got a partial batch during training."
                )
                batch = maybe_pad_last_batch(batch, dp_size, micro_batch_size)

            ## append ref policy logprobs to batch
            logprobs = policy.get_reference_policy_logprobs(
                batch,
                micro_batch_size=micro_batch_size,
            )["reference_logprobs"]
            ## want logprobs for batch to correspond to the log probabilities of the next tokens
            ## so we roll the logprobs to the left by one
            batch["reference_policy_logprobs"] = torch.roll(logprobs, -1, dims=-1)

            yield batch

        except StopIteration:
            break


# =======================================================
# Training & Validation
# =======================================================
def validate(
    policy: PolicyInterface,
    val_dataloader: dict[str, StatefulDataLoader],
    tokenizer,
    loss_fn,
    step: int,
    master_config: MasterConfig,
    val_batches: int,
    val_batch_size: int,
    val_mbs: int,
    logger: Logger,
):
    val_metrics, validation_timings = {}, {}
    for val_dataset_name, v in val_dataloader.items():
        k_val_metrics, k_validation_timings = validate_one_dataset(
            policy=policy,
            val_dataloader=v,
            loss_fn=loss_fn,
            step=step,
            master_config=master_config,
            val_batches=val_batches,
            val_batch_size=val_batch_size,
            val_mbs=val_mbs,
            dataset_name=val_dataset_name,
        )
        prefix = f"validation-{val_dataset_name}"

        logger.log_metrics(k_val_metrics, step, prefix=prefix)
        logger.log_metrics(k_validation_timings, step, prefix=f"timing/{prefix}")

        for metric_name in DPOValMetrics.__annotations__.keys():
            val_metrics[f"{prefix}_{metric_name}"] = k_val_metrics[metric_name]
        validation_timings[prefix + "_total_validation_time"] = k_validation_timings[
            "total_validation_time"
        ]

    if len(validation_timings) > 0:
        total_validation_time = sum(validation_timings.values())
        logger.log_metrics(
            {"total_validation_time": total_validation_time},
            step,
            prefix="timing/validation",
        )
        validation_timings["total_validation_time"] = total_validation_time

    return val_metrics, validation_timings


def validate_one_dataset(
    policy: PolicyInterface,
    val_dataloader: StatefulDataLoader,
    loss_fn,
    step: int,
    master_config: MasterConfig,
    val_batches: int,
    val_batch_size: int,
    val_mbs: int,
    dataset_name: str,
):
    """Run validation on one validation dataset."""
    if val_dataloader is None:
        assert val_dataloader is not None or master_config["dpo"]["val_period"] == 0, (
            "val_dataloader is None, so dpo.val_period must be 0"
        )
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return

    timer = Timer()

    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step} for `{dataset_name}` set..")

        val_metrics = defaultdict(list)
        num_valid_batches = 0
        for batch_idx, val_batch in enumerate(
            add_ref_logprobs_to_data(val_dataloader, policy, master_config, is_val=True)
        ):
            ## just run model fwd
            val_results = policy.train(
                val_batch,
                loss_fn,
                eval_mode=True,
                gbs=val_batch.size,
                mbs=val_mbs * 2,
            )

            if len(val_results["all_mb_metrics"]) == 0:
                warnings.warn(
                    "No validation metrics were collected for this batch."
                    " This is likely because there were no valid samples."
                )
            else:
                for metric_name in DPOValMetrics.__annotations__.keys():
                    reduction = (
                        np.mean
                        if metric_name in {"global_valid_seqs", "global_valid_toks"}
                        else sum
                    )
                    val_metrics[metric_name] += [
                        reduction(val_results["all_mb_metrics"][metric_name])
                    ]

                num_valid_batches += 1

            if val_batches > 0 and batch_idx >= val_batches - 1:
                break

        if num_valid_batches > 0:
            sum_num_valid_samples = sum(val_metrics["num_valid_samples"])
            global_valid_toks = sum(val_metrics["global_valid_toks"])
            global_valid_seqs = sum(val_metrics["global_valid_seqs"])
            val_metrics = DPOValMetrics(
                num_valid_samples=sum_num_valid_samples,
                global_valid_seqs=global_valid_seqs,
                global_valid_toks=global_valid_toks,
                **{
                    metric_name: sum(
                        [
                            value * weight
                            for value, weight in zip(
                                val_metrics[metric_name],
                                val_metrics["num_valid_samples"],
                            )
                        ]
                    )
                    / sum_num_valid_samples
                    for metric_name in DPOValMetrics.__annotations__.keys()
                    if metric_name
                    not in {
                        "num_valid_samples",
                        "global_valid_seqs",
                        "global_valid_toks",
                    }
                },
            )
        else:
            warnings.warn(
                "No validation metrics were collected."
                " This is likely because there were no valid samples in the validation set."
            )
            val_metrics = DPOValMetrics(
                **{
                    metric_name: 0.0
                    for metric_name in DPOValMetrics.__annotations__.keys()
                }
            )

        # Calculate validation metrics
        policy.prepare_for_training()

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    if len(val_metrics) == 0:
        warnings.warn(
            "No validation metrics were collected."
            " This is likely because there were no valid samples in the validation set."
        )

    else:
        # Print summary of validation results
        print(f"\n📊 Validation Results for `{dataset_name}` set:")
        for metric_name in DPOValMetrics.__annotations__.keys():
            print(f"    • Validation {metric_name}: {val_metrics[metric_name]:.4f}")

        # Print timing information
        print(f"\n  ⏱️  Validation Timing for `{dataset_name}` set:")
        validation_time = timing_metrics.get("total_validation_time", 0)
        print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics


def dpo_train(
    policy,
    train_dataloader,
    val_dataloader,
    tokenizer,
    loss_fn,
    master_config,
    logger,
    checkpointer,
    dpo_save_state: DPOSaveState,
) -> None:
    # Run dpo training
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    if dpo_save_state is None:
        dpo_save_state = _default_dpo_save_state()
        current_epoch = 0
        current_step = 0
        total_steps = 0
        total_valid_tokens = 0
    else:
        current_epoch = dpo_save_state["epoch"]
        current_step = dpo_save_state["step"]
        total_steps = dpo_save_state["total_steps"]
        total_valid_tokens = dpo_save_state.get(
            "total_valid_tokens", 0
        )  # Default to 0 for backward compatibility with older checkpoints

    dpo_config = master_config["dpo"]
    # Validation configuration
    val_period = dpo_config["val_period"]
    val_at_start = dpo_config["val_at_start"]
    max_num_epochs = dpo_config["max_num_epochs"]

    # Run validation at the start if configured
    if val_at_start and total_steps == 0:
        print("\n🔍 Running initial validation...")
        validation_result = validate(
            policy,
            val_dataloader,
            tokenizer,
            loss_fn,
            step=0,
            master_config=master_config,
            val_batches=dpo_config["val_batches"],
            val_batch_size=dpo_config["val_global_batch_size"],
            val_mbs=dpo_config["val_micro_batch_size"],
            logger=logger,
        )
        if validation_result is not None:
            val_metrics, validation_timings = validation_result
        else:
            val_metrics, validation_timings = None, None

    policy.prepare_for_training()

    while (
        current_epoch < max_num_epochs
        and total_steps < master_config["dpo"]["max_num_steps"]
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in add_ref_logprobs_to_data(train_dataloader, policy, master_config):
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), master_config['dpo']['max_num_steps'])} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                print("▶ Taking a training step...")
                with timer.time("policy_training"):
                    train_results = policy.train(
                        batch,
                        loss_fn,
                        eval_mode=False,
                        ## NOTE: we double the batch size here because each preference example corresponds to a pair of
                        ## examples, chosen and rejected, and the pair needs to be processed as part of the same microbatch.
                        gbs=master_config["policy"]["train_global_batch_size"] * 2,
                        mbs=master_config["policy"]["train_micro_batch_size"] * 2,
                    )

                is_last_step = total_steps + 1 >= master_config["dpo"][
                    "max_num_steps"
                ] or (
                    current_epoch + 1 == max_num_epochs
                    and current_step + 1 == len(train_dataloader)
                )

                # Run validation if it's a validation step
                if val_period > 0 and (total_steps + 1) % val_period == 0:
                    validation_result = validate(
                        policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps + 1,
                        master_config=master_config,
                        val_batches=dpo_config["val_batches"],
                        val_batch_size=dpo_config["val_global_batch_size"],
                        val_mbs=dpo_config["val_micro_batch_size"],
                        logger=logger,
                    )
                    if validation_result is not None:
                        val_metrics, validation_timings = validation_result
                    else:
                        val_metrics, validation_timings = None, None
                metrics = {
                    "loss": train_results["loss"].numpy(),
                    "grad_norm": train_results["grad_norm"].numpy(),
                }
                if "moe_metrics" in train_results:
                    metrics.update(
                        {f"moe/{k}": v for k, v in train_results["moe_metrics"].items()}
                    )
                metrics.update(train_results["all_mb_metrics"])
                for k, v in metrics.items():
                    if k in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
                        metrics[k] = np.mean(v).item()
                    else:
                        metrics[k] = np.sum(v).item()
                total_valid_tokens += metrics["global_valid_toks"]

                ## Checkpointing
                dpo_save_state["consumed_samples"] += master_config["policy"][
                    "train_global_batch_size"
                ]
                timeout.mark_iteration()

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                # +1 because step is 0-indexed
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    dpo_save_state["step"] = (current_step + 1) % len(train_dataloader)
                    dpo_save_state["total_steps"] = total_steps + 1
                    dpo_save_state["epoch"] = current_epoch
                    dpo_save_state["total_valid_tokens"] = total_valid_tokens
                    # Remove outdated validation metrics
                    for key in list(dpo_save_state):
                        if (
                            key.startswith("val")
                            and any(
                                [
                                    key.endswith(f"_{metric_name}")
                                    for metric_name in DPOValMetrics.__annotations__.keys()
                                    if metric_name != "num_valid_samples"
                                ]
                            )
                            and (val_metrics is None or key not in val_metrics)
                        ):
                            del dpo_save_state[key]
                    if val_metrics is not None:
                        dpo_save_state.update(val_metrics)

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                            f"  If you are using an old config, please updated checkpointing.metric_name to the new format, "
                            f" e.g. 'val_loss --> 'val:validation-default_loss'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in dpo_save_state:
                                del dpo_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            dpo_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {total_steps + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, dpo_save_state, master_config
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
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            train_dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            timing_metrics = timer.get_timing_metrics(reduction_op="sum")

            print("\n📊 Training Results:")
            for metric_name in DPOValMetrics.__annotations__.keys():
                print(f"  • {metric_name}: {float(metrics[metric_name]):.4f}")
            if "total_flops" in train_results:
                total_tflops = (
                    train_results["total_flops"]
                    / timing_metrics["policy_training"]
                    / 1e12
                )
                num_ranks = train_results["num_ranks"]
                print(
                    f"  • Training FLOPS: {total_tflops:.2f} TFLOPS ({total_tflops / num_ranks:.2f} TFLOPS per rank)"
                )
                if "theoretical_tflops" in train_results:
                    theoretical_tflops = train_results["theoretical_tflops"]
                    print(
                        f"  • Training Model Floating Point Utilization: {100 * total_tflops / theoretical_tflops:.2f}%"
                    )
                    metrics["train_fp_utilization"] = total_tflops / theoretical_tflops
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

            total_num_gpus = (
                master_config["cluster"]["num_nodes"]
                * master_config["cluster"]["gpus_per_node"]
            )
            timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                metrics["global_valid_toks"] / total_time / total_num_gpus
            )
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1

            if should_save_by_timeout:
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= master_config["dpo"]["max_num_steps"]:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch
