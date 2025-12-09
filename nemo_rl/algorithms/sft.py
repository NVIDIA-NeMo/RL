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
from typing import NotRequired, Optional, TypedDict, cast

import numpy as np
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.loss_functions import (
    NLLLoss,
)
from nemo_rl.algorithms.utils import maybe_pad_last_batch, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
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
from nemo_rl.utils.timer import TimeoutChecker, Timer


class SFTSaveState(TypedDict):
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    val_loss: NotRequired[float]  # Optional field - may not be present during training
    consumed_samples: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training


def _default_sft_save_state() -> SFTSaveState:
    return {
        "epoch": 0,
        "step": 0,
        "total_steps": 0,
        "consumed_samples": 0,
        "total_valid_tokens": 0,
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
    NLLLoss,
    Logger,
    CheckpointManager,
    SFTSaveState,
    MasterConfig,
]:
    """Main entry point for running SFT algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, master_config, logger
    """
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
    sft_save_state: Optional[SFTSaveState] = cast(
        Optional[SFTSaveState], checkpointer.load_training_info(last_checkpoint_path)
    )

    # ==========================
    #           Data
    # ==========================
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=policy_config["train_global_batch_size"],
        shuffle=data_config["shuffle"],
        collate_fn=rl_collate_fn,
        drop_last=True,
        num_workers=data_config["num_workers"],
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
        collate_fn=rl_collate_fn,
        drop_last=False,
        num_workers=data_config["num_workers"],
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
    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        total_train_iters = min(
            sft_config["max_num_steps"],
            sft_config["max_num_epochs"] * len(train_dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters
    # check if tokenizer is a processor (e.g. for VLMs)
    processor = None
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        processor = tokenizer
        tokenizer = processor.tokenizer

    policy = Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        processor=processor,
        weights_path=Path(last_checkpoint_path) / "policy" / "weights"
        if last_checkpoint_path
        else None,
        optimizer_path=Path(last_checkpoint_path) / "policy" / "optimizer"
        if last_checkpoint_path
        else None,
        init_optimizer=True,
        init_reference_model=False,
    )
    # print the node IP and GPU ID of the policy workers for debugging
    policy.print_node_ip_and_gpu_id()

    loss_fn = NLLLoss()
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
):
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        assert val_dataloader is not None or master_config["dpo"]["val_period"] == 0, (
            "val_dataloader is None, so dpo.val_period must be 0"
        )
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return

    timer = Timer()

    with timer.time("total_validation_time"):
        print(f"▶ Starting validation at step {step}...")

        # Show a progress indicator for validation
        # val_total = len(val_dataloader)

        val_metrics = {"val_loss": 0.0}
        sum_num_valid_tokens = 0

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

            # update multimodal data
            val_data.update(cat_and_padded.get_multimodal_dict(as_tensors=False))
            # When running validation with drop_last=False, we might end up with a partial batch.
            # Check if we need to pad the final batch to make it divisible by micro_batch_size * dp_size.
            if val_data.size < val_batch_size:
                dp_size = policy.sharding_annotations.get_axis_size("data_parallel")
                val_data = maybe_pad_last_batch(val_data, dp_size, val_mbs)

            ## just run model fwd
            val_results = policy.train(
                val_data,
                loss_fn,
                eval_mode=True,
                gbs=val_data.size,
                mbs=val_mbs,
            )

            if len(val_results["all_mb_metrics"]) == 0:
                warnings.warn(
                    "No validation metrics were collected for this batch."
                    " This is likely because there were no valid samples."
                )
            else:
                num_valid_tokens = (
                    val_data["sample_mask"].unsqueeze(-1) * val_data["token_mask"]
                ).sum()
                val_metrics["val_loss"] += float(val_results["loss"]) * num_valid_tokens
                sum_num_valid_tokens += num_valid_tokens

            if val_batches > 0 and batch_idx >= val_batches - 1:
                break

        if sum_num_valid_tokens > 0:
            val_metrics["val_loss"] /= sum_num_valid_tokens
        else:
            warnings.warn(
                "No validation metrics were collected."
                " This is likely because there were no valid samples in the validation set."
            )

        # Calculate validation metrics
        policy.prepare_for_training()

    # Get timing metrics
    timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    validation_time = timing_metrics.get("total_validation_time", 0)

    if sum_num_valid_tokens > 0:
        # Print summary of validation results
        print("\n📊 Validation Results:")
        print(f"    • Validation loss: {val_metrics['val_loss']:.4f}")

        # Print timing information
        print("\n  ⏱️  Validation Timing:")
        validation_time = timing_metrics.get("total_validation_time", 0)
        print(f"    • Total validation time: {validation_time:.2f}s")

    # Make sure to reset the timer after validation
    timer.reset()

    return val_metrics, timing_metrics


def _warmup_compiled_graph(
    policy: PolicyInterface,
    loss_fn,
    tokenizer: PreTrainedTokenizerBase,
    master_config: dict,
) -> None:
    """Warmup Ray Compiled Graph with maximum sequence length.
    
    This creates fake data with max_total_sequence_length to ensure the
    compiled graph is built with the worst-case input shape, avoiding
    recompilation during actual training.
    
    Args:
        policy: The policy to warmup
        loss_fn: Loss function to use
        tokenizer: Tokenizer for creating fake tokens
        master_config: Master configuration dict
    """
    import os
    import torch
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    
    # Get configuration
    max_seq_len = master_config["policy"]["max_total_sequence_length"]
    gbs = master_config["policy"]["train_global_batch_size"]
    mbs = master_config["policy"]["train_micro_batch_size"]
    
    # Allow override via environment variable
    warmup_seq_len = int(os.environ.get("NEMO_RL_WARMUP_SEQ_LEN", max_seq_len))
    warmup_gbs = int(os.environ.get("NEMO_RL_WARMUP_GBS", gbs))
    
    print(f"  🔧 Warmup config: SEQ_LEN={warmup_seq_len}, GBS={warmup_gbs}, MBS={mbs}")
    print(f"  📦 Creating fake data with shape: ({warmup_gbs}, {warmup_seq_len})")
    
    # Create fake data with max sequence length
    # Use valid token IDs from the tokenizer's vocabulary
    vocab_size = len(tokenizer)
    fake_input_ids = torch.randint(
        low=0, 
        high=min(vocab_size, 32000),  # Use reasonable token range
        size=(warmup_gbs, warmup_seq_len),
        dtype=torch.long
    )
    
    # Create attention mask (all ones = no padding)
    fake_attention_mask = torch.ones(
        (warmup_gbs, warmup_seq_len),
        dtype=torch.long
    )
    
    # Create position IDs
    fake_position_ids = torch.arange(
        warmup_seq_len,
        dtype=torch.long
    ).unsqueeze(0).expand(warmup_gbs, -1)
    
    # Create labels (same as input_ids for SFT)
    fake_labels = fake_input_ids.clone()
    
    # Create loss mask (all ones = compute loss on all tokens)
    fake_loss_mask = torch.ones(
        (warmup_gbs, warmup_seq_len),
        dtype=torch.float32
    )
    
    # All sequences have the same length (max_seq_len)
    fake_input_lengths = torch.full(
        (warmup_gbs,),
        warmup_seq_len,
        dtype=torch.long
    )
    
    # Create sample mask (all sequences are valid, no padding/dummy sequences)
    fake_sample_mask = torch.ones(
        (warmup_gbs,),
        dtype=torch.float32
    )
    
    # Create token mask (all tokens contribute to loss, used for token-level loss)
    fake_token_mask = torch.ones(
        (warmup_gbs, warmup_seq_len),
        dtype=torch.float32
    )
    
    # Create microbatch indices and lengths for sequence packing
    # For warmup, all sequences have same length, so we create simple placeholders
    num_microbatches = warmup_gbs // mbs
    fake_micro_batch_indices = []
    fake_micro_batch_lengths = []
    
    for mb_idx in range(num_microbatches):
        start_idx = mb_idx * mbs
        end_idx = start_idx + mbs
        # Each microbatch contains mbs sequences, each of length warmup_seq_len
        fake_micro_batch_indices.append(list(range(start_idx, end_idx)))
        fake_micro_batch_lengths.append([warmup_seq_len] * mbs)
    
    # Create BatchedDataDict with fake data
    warmup_data = BatchedDataDict({
        "input_ids": fake_input_ids,
        "attention_mask": fake_attention_mask,
        "position_ids": fake_position_ids,
        "labels": fake_labels,
        "loss_mask": fake_loss_mask,
        "input_lengths": fake_input_lengths,
        "sample_mask": fake_sample_mask,
        "token_mask": fake_token_mask,
        "micro_batch_indices": fake_micro_batch_indices,
        "micro_batch_lengths": fake_micro_batch_lengths,
    })
    
    # Store warmup data in Ray object store for efficient sharing across workers
    print(f"  🚀 Running warmup training step...")
    
    # Run one training step to trigger graph compilation
    # Use eval_mode=True to skip optimizer step (we don't care about gradients)
    try:
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        _ = policy.train(
            data=warmup_data,
            loss_fn=loss_fn,
            eval_mode=True,  # Skip optimizer step
            gbs=warmup_gbs,
            mbs=mbs,
        )
        end_time.record()
        
        # Wait for completion
        torch.cuda.synchronize()
        warmup_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        
        print(f"  ✅ Warmup step completed in {warmup_time:.2f}s")
        print(f"  💾 Compiled graph is now cached and ready for training")
        
    except Exception as e:
        print(f"  ⚠️  Warmup failed: {e}")
        print(f"  ℹ️  Continuing with normal training (graph will compile on first real step)")


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
    sft_save_state: SFTSaveState,
) -> None:
    # Run basic sft training
    timer = Timer()
    timeout = TimeoutChecker(
        timeout=master_config["checkpointing"]["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    if sft_save_state is None:
        sft_save_state = _default_sft_save_state()
        current_epoch = 0
        current_step = 0
        total_steps = 0
        total_valid_tokens = 0
    else:
        current_epoch = sft_save_state["epoch"]
        current_step = sft_save_state["step"]
        total_steps = sft_save_state["total_steps"]
        total_valid_tokens = sft_save_state.get(
            "total_valid_tokens", 0
        )  # Default to 0 for backward compatibility with older checkpoints

    sft_config = master_config["sft"]
    # Validation configuration
    val_period = sft_config["val_period"]
    val_at_start = sft_config["val_at_start"]
    max_num_epochs = sft_config["max_num_epochs"]

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
        )

        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    # Warmup compiled graph with max sequence length if enabled
    import os
    from nemo_rl.distributed.ray_compiled_graph import should_use_compiled_graph
    
    # Only warmup if BOTH warmup flag AND RCG are enabled
    warmup_enabled = os.environ.get("NEMO_RL_WARMUP_COMPILED_GRAPH", "0") == "1"
    rcg_enabled = should_use_compiled_graph()
    
    if warmup_enabled and rcg_enabled and total_steps == 0:
        print("\n🔥 Warming up Ray Compiled Graph with max sequence length...")
        _warmup_compiled_graph(
            policy=policy,
            loss_fn=loss_fn,
            tokenizer=tokenizer,
            master_config=master_config,
        )
        print("✅ Warmup complete! Proceeding with normal training...\n")

    policy.prepare_for_training()

    while (
        current_epoch < max_num_epochs
        and total_steps < master_config["sft"]["max_num_steps"]
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}", flush=True)

        for batch in train_dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), master_config['sft']['max_num_steps'])} {'=' * 25}",
                flush=True,
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            val_metrics, validation_timings = None, None

            with timer.time("total_step_time"):
                # Prepare batch and generate responses
                print("▶ Preparing batch...", flush=True)
                with timer.time("data_processing"):
                    import time as time_module
                    import torch
                    dp_start = time_module.time()
                    
                    torch.cuda.nvtx.range_push("data_processing/add_loss_mask")
                    mask_start = time_module.time()
                    ## add loss mask based on role to every message
                    add_loss_mask_to_message_log(
                        batch["message_log"],
                        roles_to_train_on=["assistant"],
                    )
                    mask_time = time_module.time() - mask_start
                    torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("data_processing/flatten_and_pad")
                    flat_start = time_module.time()
                    cat_and_padded, input_lengths = batched_message_log_to_flat_message(
                        batch["message_log"],
                        pad_value_dict={"token_ids": tokenizer.pad_token_id},
                        make_sequence_length_divisible_by=master_config["policy"][
                            "make_sequence_length_divisible_by"
                        ],
                    )
                    flat_time = time_module.time() - flat_start
                    torch.cuda.nvtx.range_pop()

                    torch.cuda.nvtx.range_push("data_processing/create_batch_dict")
                    batch_start = time_module.time()
                    train_data: BatchedDataDict = BatchedDataDict(
                        {
                            "input_ids": cat_and_padded["token_ids"],
                            "input_lengths": input_lengths,
                            "token_mask": cat_and_padded["token_loss_mask"],
                            "sample_mask": batch["loss_multiplier"],
                        }
                    )
                    train_data.update(
                        cat_and_padded.get_multimodal_dict(as_tensors=False)
                    )
                    batch_time = time_module.time() - batch_start
                    torch.cuda.nvtx.range_pop()
                    
                print("▶ Taking a training step...", flush=True)
                
                with timer.time("policy_training"):
                    train_results = policy.train(train_data, loss_fn)
                
                # Debug: write to file with SLURM job ID

                is_last_step = total_steps + 1 >= master_config["sft"][
                    "max_num_steps"
                ] or (
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
                    )
                    logger.log_metrics(
                        validation_timings, total_steps + 1, prefix="timing/validation"
                    )
                    logger.log_metrics(
                        val_metrics, total_steps + 1, prefix="validation"
                    )
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
                sft_save_state["consumed_samples"] += master_config["policy"][
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
                    sft_save_state["step"] = (current_step + 1) % len(train_dataloader)
                    sft_save_state["total_steps"] = total_steps + 1
                    sft_save_state["epoch"] = current_epoch
                    sft_save_state["total_valid_tokens"] = total_valid_tokens

                    full_metric_name = master_config["checkpointing"]["metric_name"]
                    if full_metric_name is not None:
                        assert full_metric_name.startswith(
                            "train:"
                        ) or full_metric_name.startswith("val:"), (
                            f"metric_name={full_metric_name} must start with 'val:' or 'train:',\n"
                            f'followed by the corresponding name in the "val" or "train" metrics dictionary.'
                            f"  If you are using an old config, please updated checkpointing.metric_name to the new format, "
                            f" e.g. 'val_loss --> 'val:val_loss'"
                        )
                        prefix, metric_name = full_metric_name.split(":", 1)
                        metrics_source = metrics if prefix == "train" else val_metrics
                        if not metrics_source:
                            warnings.warn(
                                f"You asked to save checkpoints based on {metric_name} but no {prefix} metrics were collected. "
                                "This checkpoint will not be saved as top-k.",
                                stacklevel=2,
                            )
                            if full_metric_name in sft_save_state:
                                del sft_save_state[full_metric_name]
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            sft_save_state[full_metric_name] = metrics_source[
                                metric_name
                            ]

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
                            checkpointing_cfg=master_config["checkpointing"],
                        )
                        torch.save(
                            train_dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.finalize_checkpoint(checkpoint_path)

            timing_metrics = timer.get_timing_metrics(reduction_op="sum")
            
            # Add worker timing metrics from train_results
            if "worker_computation_time_max" in train_results:
                timing_metrics["worker_computation_time_max"] = train_results["worker_computation_time_max"]
            if "worker_computation_time_min" in train_results:
                timing_metrics["worker_computation_time_min"] = train_results["worker_computation_time_min"]
            if "worker_imbalance" in train_results:
                timing_metrics["worker_imbalance"] = train_results["worker_imbalance"]
            
            # Debug: write timing_metrics to file
            import time as time_module
            log_dir = os.environ.get('WORKER_LOG_DIR', '/tmp')
            job_id = os.environ.get('SLURM_JOB_ID', 'unknown')
            debug_file = os.path.join(log_dir, f'{job_id}_sft_timing_debug.log')
            
            try:
                with open(debug_file, "a") as f:
                    f.write(f"timing_metrics after get_timing_metrics: {dict(timing_metrics)}\n")
                    f.write(f"{'='*80}\n\n")
                    f.flush()
            except Exception:
                pass

            print("\n📊 Training Results:")
            print(f"  • Loss: {float(metrics['loss']):.4f}")
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
            if total_steps >= master_config["sft"]["max_num_steps"]:
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch
