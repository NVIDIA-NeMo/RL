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
from typing import Optional, TypedDict

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.loss_functions import (
    PreferenceLoss,
)
from nemo_rl.algorithms.utils import maybe_pad_last_batch, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import (
    AllTaskProcessedDataset,
    preference_collate_fn,
)
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import PolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer


class RMSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    consumed_samples: int


def _default_rm_save_state() -> RMSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
    }


class RMConfig(TypedDict):
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
    rm: RMConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


class RMValMetrics(TypedDict):
    loss: float
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
    val_dataset: dict[str, AllTaskProcessedDataset],
) -> tuple[
    Policy,
    RayVirtualCluster,
    StatefulDataLoader,
    dict[str, StatefulDataLoader],
    PreferenceLoss,
    MasterConfig,
    Logger,
    TaskDataSpec,
    RMSaveState,
]:
    """Main entry point for running RM algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, master_config, logger
    """
    set_seed(master_config["rm"]["seed"])

    # Extract individual configs for easier access
    policy_config = master_config["policy"]
    data_config = master_config["data"]
    logger_config = master_config["logger"]
    cluster_config = master_config["cluster"]
    rm_config = master_config["rm"]

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
    rm_save_state: Optional[RMSaveState] = checkpointer.load_training_info(
        last_checkpoint_path
    )

    # ==========================
    #           Data
    # ==========================
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
            add_loss_mask=False,
        ),
        drop_last=True,
    )

    if last_checkpoint_path is not None:
        dataloader_state_dict = torch.load(
            os.path.join(last_checkpoint_path, "train_dataloader.pt")
        )
        train_dataloader.load_state_dict(dataloader_state_dict)

    val_dataloader = {
        k: StatefulDataLoader(
            v,
            batch_size=rm_config["val_global_batch_size"],
            shuffle=False,
            collate_fn=partial(
                preference_collate_fn,
                tokenizer=tokenizer,
                make_sequence_length_divisible_by=policy_config[
                    "make_sequence_length_divisible_by"
                ],
                add_loss_mask=False,
            ),
            drop_last=False,
        )
        for k, v in val_dataset.items()
    }

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="rm_cluster",
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
            rm_config["max_num_steps"],
            rm_config["max_num_epochs"] * len(train_dataloader),
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
        init_reference_model=False,
    )
    loss_fn = PreferenceLoss()
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
        rm_save_state,
        master_config,
    )


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

        for metric_name in RMValMetrics.__annotations__.keys():
            if metric_name != "num_valid_samples":
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

        # Show a progress indicator for validation
        # val_total = len(val_dataloader)

        dict_val_metrics = defaultdict(list)
        num_valid_batches = 0
        # Prepare JSONL dump path
        dump_dir = Path(master_config["logger"]["log_dir"]) / "validation"
        dump_dir.mkdir(parents=True, exist_ok=True)
        dump_fp = dump_dir / f"{dataset_name}_pairs_step_{step}.jsonl"
        import json as _json
        fout = open(dump_fp, "w")
        for batch_idx, val_batch in enumerate(val_dataloader):
            # When running validation with drop_last=False, we might end up with a partial batch.
            # In this case, we pad the batch to the next multiple of micro_batch_size * dp_size.
            if val_batch.size < val_batch_size * 2:
                dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
                val_batch = maybe_pad_last_batch(val_batch, dp_size, val_mbs * 2)

            ## just run model fwd
            val_results = policy.train(
                val_batch,
                loss_fn,
                eval_mode=True,
                gbs=val_batch.size,
                # NOTE: we double the batch size because each preference example corresponds to a pair of
                # examples, chosen and rejected, and the pair needs to be processed as part of the same microbatch.
                mbs=val_mbs * 2,
            )

            if len(val_results["all_mb_metrics"]) == 0:
                warnings.warn(
                    "No validation metrics were collected for this batch."
                    " This is likely because there were no valid samples."
                )
            else:
                for metric_name in RMValMetrics.__annotations__.keys():
                    dict_val_metrics[metric_name] += [
                        sum(val_results["all_mb_metrics"][metric_name])
                    ]

                num_valid_batches += 1

                # Emit JSONL per-pair entries with explicit idx→reward alignment
                # Prefer worker-returned per-sample idx ordering when available
                per_pair_chosen = (
                    val_results["all_mb_metrics"].get("per_pair_reward_chosen") or []
                )
                per_pair_rejected = (
                    val_results["all_mb_metrics"].get("per_pair_reward_rejected") or []
                )
                per_sample_idx = val_results["all_mb_metrics"].get("per_sample_idx") or []
                per_sample_rewards = val_results["all_mb_metrics"].get("per_sample_rewards") or []
                per_sample_raw = val_results["all_mb_metrics"].get("per_sample_raw_logits") or []
                per_sample_is_chosen = val_results["all_mb_metrics"].get("per_sample_is_chosen") or []
                per_sample_mask = val_results["all_mb_metrics"].get("per_sample_mask") or []
                # Flatten helpers
                def _flat(x):
                    out = []
                    for v in x:
                        if hasattr(v, "tolist"):
                            out.extend(v.tolist())
                        elif isinstance(v, (list, tuple)):
                            out.extend(list(v))
                        else:
                            out.append(v)
                    return out
                per_pair_chosen = _flat(per_pair_chosen)
                per_pair_rejected = _flat(per_pair_rejected)
                per_sample_idx = _flat(per_sample_idx)
                per_sample_rewards = _flat(per_sample_rewards)
                per_sample_raw = _flat(per_sample_raw)
                per_sample_is_chosen = _flat(per_sample_is_chosen)
                per_sample_mask = _flat(per_sample_mask)

                # Build idx->(chosen,rejected) map using per_sample_idx order if available
                idx_to_rewards: dict[int, tuple[float,float]] = {}
                idx_to_source: dict[int, str] = {}
                if per_sample_idx and per_sample_rewards and len(per_sample_rewards) == len(per_sample_idx):
                    # Build per-pair rewards without assuming strict adjacency, using is_chosen flags when available
                    tmp: dict[int, dict[str, float]] = {}
                    for pos, (idx_val_raw, reward_val) in enumerate(zip(per_sample_idx, per_sample_rewards)):
                        # Skip masked/padded samples
                        if per_sample_mask and pos < len(per_sample_mask):
                            try:
                                if float(per_sample_mask[pos]) <= 0:
                                    continue
                            except Exception:
                                pass
                        try:
                            idx_val = int(idx_val_raw)
                        except Exception:
                            continue
                        is_chosen = None
                        if per_sample_is_chosen and pos < len(per_sample_is_chosen):
                            try:
                                is_chosen = int(per_sample_is_chosen[pos])
                            except Exception:
                                is_chosen = None
                        entry = tmp.setdefault(idx_val, {})
                        if is_chosen == 1:
                            entry["chosen"] = float(reward_val)
                        elif is_chosen == 0:
                            entry["rejected"] = float(reward_val)
                        else:
                            # Fallback to parity if flags missing
                            if (pos % 2) == 0:
                                entry.setdefault("chosen", float(reward_val))
                            else:
                                entry.setdefault("rejected", float(reward_val))
                    for idx_val, parts in tmp.items():
                        if "chosen" in parts and "rejected" in parts:
                            idx_to_rewards[idx_val] = (parts["chosen"], parts["rejected"])  # type: ignore[index]
                            idx_to_source[idx_val] = "per_sample"
                elif per_sample_idx and len(per_sample_idx) % 2 == 0 and len(per_pair_chosen) * 2 == len(per_sample_idx):
                    for p, i in enumerate(range(0, len(per_sample_idx), 2)):
                        idx_val = int(per_sample_idx[i])
                        rw_c = float(per_pair_chosen[p]) if p < len(per_pair_chosen) else None
                        rw_r = float(per_pair_rejected[p]) if p < len(per_pair_rejected) else None
                        if rw_c is not None and rw_r is not None:
                            idx_to_rewards[idx_val] = (rw_c, rw_r)
                            idx_to_source[idx_val] = "per_pair"
                else:
                    # Fallback: rely on current batch idx ordering
                    idx_seq = val_batch["idx"]
                    pair_indices = []
                    for i in range(0, len(idx_seq), 2):
                        try:
                            pair_indices.append(int(idx_seq[i]))
                        except Exception:
                            continue
                    for p, idx_val in enumerate(pair_indices):
                        rw_c = float(per_pair_chosen[p]) if p < len(per_pair_chosen) else None
                        rw_r = float(per_pair_rejected[p]) if p < len(per_pair_rejected) else None
                        if rw_c is not None and rw_r is not None:
                            idx_to_rewards[idx_val] = (rw_c, rw_r)
                            idx_to_source[idx_val] = "fallback"

                # Helper to extract formatted text and token ids stored by preprocessor in dataset items
                # These are not in the collated tensors; recover from message logs present in batch
                msg_logs = val_batch["message_log"]
                flat_ids_batch = val_batch.get("input_ids")
                flat_lens_batch = val_batch.get("input_lengths")
                # Recompute pair indices from current batch message logs
                batch_pair_indices = []
                idx_seq_batch = val_batch["idx"]
                for i in range(0, len(idx_seq_batch), 2):
                    try:
                        batch_pair_indices.append(int(idx_seq_batch[i]))
                    except Exception:
                        continue
                for pair_i, global_idx in enumerate(batch_pair_indices):
                    # For each pair, the interleaved positions are 2*pair_i and 2*pair_i+1
                    pos_c = 2 * pair_i
                    pos_r = 2 * pair_i + 1
                    # Extract rendered strings as concatenation of per-message content
                    def _concat_content(ml):
                        parts = []
                        for m in ml:
                            c = m.get("content")
                            if isinstance(c, str):
                                parts.append(c)
                            elif isinstance(c, list):
                                # multimodal: extract text items
                                for item in c:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        parts.append(item.get("text", ""))
                        return "".join(parts)
                    ml_c = msg_logs[pos_c]
                    ml_r = msg_logs[pos_r]
                    text_c = _concat_content(ml_c)
                    text_r = _concat_content(ml_r)
                    # Flatten token ids
                    ids_c = [int(t) for m in ml_c for t in m.get("token_ids", []).tolist()]
                    ids_r = [int(t) for m in ml_r for t in m.get("token_ids", []).tolist()]
                    # Rewards (aligned by idx when available)
                    rw_c, rw_r = idx_to_rewards.get(global_idx, (None, None))
                    # Extract exact flat input_ids slices used in forward for chosen/rejected
                    flat_ids_c = None
                    flat_ids_r = None
                    if flat_ids_batch is not None and flat_lens_batch is not None:
                        try:
                            Lc = int(flat_lens_batch[pos_c])
                            Lr = int(flat_lens_batch[pos_r])
                            flat_ids_c = [int(t) for t in flat_ids_batch[pos_c, :Lc].detach().cpu().tolist()]
                            flat_ids_r = [int(t) for t in flat_ids_batch[pos_r, :Lr].detach().cpu().tolist()]
                        except Exception:
                            flat_ids_c = None
                            flat_ids_r = None
                    rec = {
                        "dataset": dataset_name,
                        "batch_id": batch_idx,
                        "within_batch_pair_index": pair_i,
                        "pair_index": global_idx,
                        "applyChatTemplate_chosen": text_c,
                        "applyChatTemplate_rejected": text_r,
                        "token_ids_chosen": ids_c,
                        "token_ids_rejected": ids_r,
                        # Exact model inputs used (trimmed to true lengths)
                        "flat_input_ids_chosen": flat_ids_c,
                        "flat_input_ids_rejected": flat_ids_r,
                        "reward_chosen": rw_c,
                        "reward_rejected": rw_r,
                        "reward_delta": (None if (rw_c is None or rw_r is None) else float(rw_c - rw_r)),
                        "debug_reward_source": idx_to_source.get(global_idx, None),
                        # Optional diagnostics
                        "per_sample_rewards": per_sample_rewards if per_sample_rewards else None,
                        "per_sample_raw_logits": per_sample_raw if per_sample_raw else None,
                    }
                    fout.write(_json.dumps(rec) + "\n")

            if val_batches > 0 and batch_idx >= val_batches - 1:
                break

        try:
            fout.close()
            print(f"  ✓ Wrote validation dump to {dump_fp}")
        except Exception:
            pass

        if num_valid_batches > 0:
            sum_num_valid_samples = sum(dict_val_metrics["num_valid_samples"])
            val_metrics = RMValMetrics(
                num_valid_samples=sum_num_valid_samples,
                **{
                    metric_name: sum(
                        [
                            value * weight
                            for value, weight in zip(
                                dict_val_metrics[metric_name],
                                dict_val_metrics["num_valid_samples"],
                            )
                        ]
                    )
                    / sum_num_valid_samples
                    for metric_name in RMValMetrics.__annotations__.keys()
                    if metric_name != "num_valid_samples"
                },
            )
        else:
            warnings.warn(
                "No validation metrics were collected."
                " This is likely because there were no valid samples in the validation set."
            )
            val_metrics = RMValMetrics(
                **{
                    metric_name: 0.0
                    for metric_name in RMValMetrics.__annotations__.keys()
                }
            )

        # Calculate validation metrics
        policy.prepare_for_training()

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    if num_valid_batches > 0:
        # Print summary of validation results
        print(f"\n📊 Validation Results for `{dataset_name}` set:")
        for metric_name in RMValMetrics.__annotations__.keys():
            if metric_name != "num_valid_samples":
                print(f"    • Validation {metric_name}: {val_metrics[metric_name]:.4f}")
            else:
                print(
                    f"    • Validation num valid samples: {val_metrics['num_valid_samples']:.0f}"
                )

        # Print timing information
        print(f"\n  ⏱️  Validation Timing for `{dataset_name}` set:")
        validation_time = timing_metrics.get("total_validation_time", 0)
        print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics


def rm_train(
    policy,
    train_dataloader,
    val_dataloader,
    tokenizer,
    loss_fn,
    master_config,
    logger,
    rm_task_spec,
    checkpointer,
    rm_save_state,
):
    # Run basic rm training
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()
    if rm_save_state is None:
        rm_save_state = _default_rm_save_state()
        current_epoch = 0
        current_step = 0
        total_steps = 0
    else:
        current_epoch = rm_save_state["epoch"]
        current_step = rm_save_state["step"]
        total_steps = rm_save_state["total_steps"]

    rm_config = master_config["rm"]
    # Validation configuration
    val_period = rm_config["val_period"]
    val_at_start = rm_config["val_at_start"]
    max_num_epochs = rm_config["max_num_epochs"]

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
            val_batches=rm_config["val_batches"],
            val_batch_size=rm_config["val_global_batch_size"],
            val_mbs=rm_config["val_micro_batch_size"],
            logger=logger,
        )

    policy.prepare_for_training()

    while current_epoch < max_num_epochs and (
        master_config["rm"]["max_num_steps"] == -1
        or total_steps < master_config["rm"]["max_num_steps"]
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in train_dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), master_config['rm']['max_num_steps'] if master_config['rm']['max_num_steps'] != -1 else len(train_dataloader))} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # Prepare batch and generate responses
                print("▶ Taking a training step...")

                train_results = policy.train(
                    batch,
                    loss_fn,
                    eval_mode=False,
                    ## NOTE: we double the batch size here because each preference example corresponds to a pair of
                    ## examples, chosen and rejected, and the pair needs to be processed as part of the same microbatch.
                    gbs=master_config["policy"]["train_global_batch_size"] * 2,
                    mbs=master_config["policy"]["train_micro_batch_size"] * 2,
                )

                is_last_step = (
                    master_config["rm"]["max_num_steps"] != -1
                    and total_steps + 1 >= master_config["rm"]["max_num_steps"]
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
                        val_batches=rm_config["val_batches"],
                        val_batch_size=rm_config["val_global_batch_size"],
                        val_mbs=rm_config["val_micro_batch_size"],
                        logger=logger,
                    )

                ## Checkpointing
                timeout.mark_iteration()

                rm_save_state["consumed_samples"] += master_config["policy"][
                    "train_global_batch_size"
                ]

                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config["checkpointing"]["save_period"]
                    == 0
                )
                should_save_by_timeout = timeout.check_save()

                if master_config["checkpointing"]["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    ## +1 because step is 0-indexed
                    rm_save_state["step"] = (current_step + 1) % len(train_dataloader)
                    rm_save_state["total_steps"] = total_steps + 1
                    rm_save_state["epoch"] = current_epoch
                    # Remove outdated validation metrics
                    for key in list(rm_save_state):
                        if (
                            key.startswith("val")
                            and any(
                                [
                                    key.endswith(f"_{metric_name}")
                                    for metric_name in RMValMetrics.__annotations__.keys()
                                    if metric_name != "num_valid_samples"
                                ]
                            )
                            and (val_metrics is None or key not in val_metrics)
                        ):
                            del rm_save_state[key]
                    if val_metrics is not None:
                        rm_save_state.update(val_metrics)

                    if master_config["checkpointing"]["metric_name"] is not None:
                        if (
                            master_config["checkpointing"]["metric_name"]
                            not in rm_save_state
                        ):
                            warnings.warn(
                                f"You asked to save checkpoints based on {master_config['checkpointing']['metric_name']} but the metric is not found in the save state. "
                                "Saving most recent k checkpoints instead."
                            )
                            master_config["checkpointing"]["metric_name"] = None

                    with timer.time("checkpointing"):
                        print(f"Saving checkpoint for step {total_steps + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, rm_save_state, master_config
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
            for metric_name in RMValMetrics.__annotations__.keys():
                if metric_name != "num_valid_samples":
                    print(f"  • {metric_name}: {float(metrics[metric_name]):.4f}")
                else:
                    print(f"  • num valid samples: {float(metrics[metric_name]):.0f}")

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

            if should_save_by_timeout:
                return
            if (
                master_config["rm"]["max_num_steps"] != -1
                and total_steps >= master_config["rm"]["max_num_steps"]
            ):
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch
