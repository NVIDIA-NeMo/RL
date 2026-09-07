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
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np
import torch
from pydantic import BaseModel
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from nemo_rl.algorithms.loss.loss_functions import NLLLossFn
from nemo_rl.algorithms.utils import maybe_pad_last_batch, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from nemo_rl.data.multimodal_utils import PER_TOKEN_MULTIMODAL_FIELDS, PackedTensor
from nemo_rl.data.utils import load_dataloader_state
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
    prepare_segment_topology,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import PolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.telemetry.config import TelemetryConfig
from nemo_rl.telemetry.instrumentation import managed_span, trace_fn
from nemo_rl.telemetry.setup import get_telemetry_handle
from nemo_rl.telemetry.span_groups import RLSpanGroup
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.nsys import maybe_gpu_profile_step
from nemo_rl.utils.timer import TimeoutChecker, Timer

if TYPE_CHECKING:
    from nemo_rl.data.energon.sft_dataloader import SFTDataLoader


@dataclass
class SFTSaveState:
    epoch: int  # Track current epoch
    step: int  # Track step within current epoch
    total_steps: int  # Track total number of steps across all epochs
    consumed_samples: int
    total_valid_tokens: int  # Track total number of non-padding tokens during training


def _initial_sft_save_state() -> SFTSaveState:
    return SFTSaveState(
        epoch=0, step=0, total_steps=0, consumed_samples=0, total_valid_tokens=0
    )


def _get_sft_save_state(
    loaded_state: Optional[dict[str, Any]],
) -> SFTSaveState:
    if loaded_state is None:
        return _initial_sft_save_state()

    # Start from current defaults so partial/legacy checkpoints remain loadable.
    known_fields = {field.name for field in fields(SFTSaveState)}
    state_values = vars(_initial_sft_save_state()).copy()
    state_values.update(
        {key: value for key, value in loaded_state.items() if key in known_fields}
    )
    return SFTSaveState(**state_values)


class SFTConfig(BaseModel, extra="allow"):
    max_num_steps: int = 60
    max_num_epochs: int = 1
    val_period: int = 10
    val_batches: int = 8
    val_global_batch_size: int = 32
    val_micro_batch_size: int = 1
    val_at_start: bool = True
    # Whether to run validation on the last training step. Setting this to True ensures the
    # final checkpoint has validation metrics, which is required for get_best_checkpoint_path().
    val_at_end: bool = False
    seed: int = 42
    only_unmask_final: bool = False


class MasterConfig(BaseModel, extra="allow"):
    policy: PolicyConfig
    data: DataConfig
    sft: SFTConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig
    telemetry: Optional[TelemetryConfig] = None


# =======================================================
# Setup & Initialization
# =======================================================
def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: Optional[AllTaskProcessedDataset],
    val_dataset: Optional[AllTaskProcessedDataset],
) -> tuple[
    Policy,
    RayVirtualCluster,
    SFTDataLoader,
    Optional[SFTDataLoader],
    NLLLossFn,
    Logger,
    CheckpointManager,
    SFTSaveState,
    MasterConfig,
]:
    """Main entry point for running SFT algorithm.

    Returns:
        Tuple of policy, cluster, dataloader, tokenizer, loss_fn, math_env, master_config, logger
    """
    set_seed(master_config.sft.seed)

    # Extract individual configs for easier access
    policy_config = master_config.policy
    data_config = master_config.data
    sft_config = master_config.sft
    logger_config = master_config.logger
    cluster_config = master_config.cluster
    checkpointing_config = master_config.checkpointing

    # VLM entry points pass a processor in the tokenizer position.
    processor = None
    if not isinstance(tokenizer, PreTrainedTokenizerBase):
        processor = tokenizer
        tokenizer = processor.tokenizer

    checkpointing_pretrained = checkpointing_config.get("pretrained_checkpoint")
    if checkpointing_pretrained is not None:
        policy_config["pretrained_checkpoint"] = checkpointing_pretrained

    # ==========================
    #         Logger
    # ==========================
    logger = Logger(logger_config)
    logger.log_hyperparams(master_config.model_dump())

    # ==========================
    #      Checkpointing
    # ==========================
    checkpointer = CheckpointManager(checkpointing_config)
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    loaded_state = checkpointer.load_training_info(last_checkpoint_path)
    sft_save_state = _get_sft_save_state(loaded_state)

    # ==========================
    #           Data
    # ==========================
    if data_config.get("backend") == "energon":
        raise ValueError("Energon SFT is not supported yet.")
    else:
        if train_dataset is None:
            raise ValueError("The Hugging Face SFT backend requires a train dataset.")
        train_dataloader = StatefulDataLoader(
            train_dataset,
            batch_size=policy_config["train_global_batch_size"],
            shuffle=data_config["shuffle"],
            collate_fn=rl_collate_fn,
            drop_last=True,
            num_workers=data_config["num_workers"],
        )

        if val_dataset is not None:
            val_dataloader = StatefulDataLoader(
                val_dataset,
                batch_size=sft_config.val_global_batch_size,
                shuffle=False,
                collate_fn=rl_collate_fn,
                drop_last=False,
                num_workers=data_config["num_workers"],
            )
        else:
            val_dataloader = None

    if last_checkpoint_path is not None:
        load_dataloader_state(train_dataloader, last_checkpoint_path, data_config)

    # ==========================
    #          Cluster
    # ==========================
    print("\n▶ Setting up compute cluster...")
    num_nodes = cluster_config["num_nodes"]
    segment_size = cluster_config.get("segment_size")
    node_resource_constraints, _, _ = prepare_segment_topology(segment_size, num_nodes)
    cluster = RayVirtualCluster(
        name="sft_cluster",
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]] * num_nodes,
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
        port_range_low=cluster_config.get("master_port_range_low"),
        port_range_high=cluster_config.get("master_port_range_high"),
        segment_size=segment_size,
        node_resource_constraints=node_resource_constraints,
    )
    print(f"  ✓ Ray cluster initialized with {num_nodes} nodes")

    # ==========================
    #   Training
    # ==========================
    print("\n▶ Setting up model...")
    if policy_config.get("megatron_cfg", {}).get("enabled", False):
        total_train_iters = min(
            sft_config.max_num_steps,
            sft_config.max_num_epochs * len(train_dataloader),
        )
        policy_config["megatron_cfg"]["train_iters"] = total_train_iters
    weights_path, optimizer_path = checkpointer.get_resume_paths(last_checkpoint_path)

    policy = Policy(
        cluster=cluster,
        config=policy_config,
        tokenizer=tokenizer,
        processor=processor,
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
        init_reference_model=False,
    )
    # print the node IP and GPU ID of the policy workers for debugging
    policy.print_node_ip_and_gpu_id()

    loss_fn = NLLLossFn(
        use_fused_linear_logprobs=policy_config["megatron_cfg"]["enabled"]
        and policy_config["megatron_cfg"]["use_fused_linear_logprobs"]
    )
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
def _validate_prepared_sft_batch(batch: BatchedDataDict) -> None:
    required_keys = {"input_ids", "input_lengths", "token_mask", "sample_mask"}
    missing_keys = required_keys.difference(batch)
    if missing_keys:
        raise ValueError(f"Prepared SFT batch is missing keys: {sorted(missing_keys)}.")

    input_ids = batch["input_ids"]
    input_lengths = batch["input_lengths"]
    token_mask = batch["token_mask"]
    sample_mask = batch["sample_mask"]
    if not all(
        isinstance(value, torch.Tensor)
        for value in (input_ids, input_lengths, token_mask, sample_mask)
    ):
        raise TypeError("Prepared SFT core fields must be torch tensors.")
    if input_ids.ndim != 2 or token_mask.shape != input_ids.shape:
        raise ValueError("Prepared input_ids and token_mask must have matching [B, S].")
    batch_size, sequence_length = input_ids.shape
    if input_lengths.shape != (batch_size,) or sample_mask.shape != (batch_size,):
        raise ValueError("Prepared input_lengths and sample_mask must have shape [B].")
    if bool(torch.any(input_lengths < 0)) or bool(
        torch.any(input_lengths > sequence_length)
    ):
        raise ValueError("Prepared input_lengths must be within the padded sequence.")
    trainable_by_row = token_mask.bool().any(dim=-1)
    degenerate = sample_mask.bool() & ~trainable_by_row
    if bool(torch.any(degenerate)):
        # Such a row contributed exactly 0 to the loss before this validation
        # existed, so mask it out rather than killing a run hours in over one
        # bad sample (e.g. a conversation with no assistant turn).
        warnings.warn(
            f"{int(degenerate.sum())} active SFT rows have no trainable token; "
            "masking them out of this batch.",
            stacklevel=2,
        )
        batch["sample_mask"] = sample_mask * (~degenerate).to(sample_mask.dtype)

    for key, value in batch.items():
        if isinstance(value, PackedTensor) and len(value) != batch_size:
            raise ValueError(
                f"Prepared PackedTensor {key!r} has {len(value)} rows; "
                f"expected {batch_size}."
            )
    for key in PER_TOKEN_MULTIMODAL_FIELDS:
        value = batch.get(key)
        if isinstance(value, torch.Tensor) and value.shape != input_ids.shape:
            raise ValueError(
                f"Prepared sequence side tensor {key!r} must match input_ids."
            )


def prepare_sft_batch(
    batch: Mapping[str, Any],
    *,
    tokenizer: PreTrainedTokenizerBase,
    only_unmask_final: bool,
    make_sequence_length_divisible_by: int,
) -> BatchedDataDict:
    """Normalize a tokenized message-log batch or an already prepared batch."""
    if "message_log" in batch:
        loss_mask_mode = batch.get("loss_mask_mode")
        if loss_mask_mode == "precomputed":
            if only_unmask_final:
                raise ValueError(
                    "only_unmask_final cannot override precomputed SFT loss masks."
                )
            for message_log in batch["message_log"]:
                for message in message_log:
                    tokens = message.get("token_ids")
                    mask = message.get("token_loss_mask")
                    if (
                        not isinstance(tokens, torch.Tensor)
                        or not isinstance(mask, torch.Tensor)
                        or tokens.ndim != 1
                        or mask.ndim != 1
                        or tokens.shape != mask.shape
                        or bool(((mask != 0) & (mask != 1)).any())
                    ):
                        raise ValueError(
                            "Precomputed SFT masks must be binary vectors matching "
                            "each token vector."
                        )
        elif loss_mask_mode is None:
            add_loss_mask_to_message_log(
                batch["message_log"],
                roles_to_train_on=["assistant"],
                only_unmask_final=only_unmask_final,
            )
        else:
            raise ValueError(f"Unsupported SFT loss_mask_mode={loss_mask_mode!r}.")
        cat_and_padded, input_lengths = batched_message_log_to_flat_message(
            batch["message_log"],
            pad_value_dict={"token_ids": tokenizer.pad_token_id},
            make_sequence_length_divisible_by=make_sequence_length_divisible_by,
        )
        sample_mask = batch["loss_multiplier"]
        if isinstance(sample_mask, torch.Tensor) and sample_mask.ndim == 0:
            sample_mask = sample_mask.unsqueeze(0)
        prepared = BatchedDataDict(
            {
                "input_ids": cat_and_padded["token_ids"],
                "input_lengths": input_lengths,
                "token_mask": cat_and_padded["token_loss_mask"],
                "sample_mask": sample_mask,
            }
        )
        prepared.update(cat_and_padded.get_multimodal_dict(as_tensors=False))
        # get_multimodal_dict only carries PackedTensor leaves and the optional
        # sequence-aligned tensors, so per-sample provenance has to come across
        # explicitly or the data plane tags every row "unknown:<row>".
        if "source_ids" in batch:
            prepared["source_ids"] = batch["source_ids"]
    else:
        prepared = (
            batch if isinstance(batch, BatchedDataDict) else BatchedDataDict(batch)
        )

    _validate_prepared_sft_batch(prepared)
    return prepared


def validate(
    policy: PolicyInterface,
    val_dataloader: Optional[SFTDataLoader],
    tokenizer,
    loss_fn,
    step: int,
    master_config: MasterConfig,
    val_batches: int,
    val_batch_size: int,
    val_mbs: int,
):
    """Run validation on the validation dataset."""
    if val_dataloader is None:
        assert master_config.sft.val_period <= 0, (
            "val_dataloader is None, so sft.val_period must be <= 0"
        )
        print("  ⚠️ No validation dataloader provided, skipping validation")
        return {}, {}

    timer = Timer()
    _telemetry = get_telemetry_handle()
    _tracer = _telemetry.tracer if _telemetry is not None else None

    with (
        timer.time("total_validation_time"),
        managed_span(
            RLSpanGroup.EVALUATE,
            "rl.sft.evaluate",
            tracer=_tracer,
            **{"rl.step": step},
        ),
    ):
        print(f"▶ Starting validation at step {step}...")

        # Show a progress indicator for validation
        # val_total = len(val_dataloader)

        val_metrics = {"val_loss": 0.0}
        sum_num_valid_tokens = 0

        policy.prepare_for_training()
        for batch_idx, val_batch in enumerate(val_dataloader):
            val_data = prepare_sft_batch(
                val_batch,
                tokenizer=tokenizer,
                only_unmask_final=master_config.sft.only_unmask_final,
                make_sequence_length_divisible_by=master_config.policy[
                    "make_sequence_length_divisible_by"
                ],
            )
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


@trace_fn(RLSpanGroup.JOB, "rl.sft.job")
def sft_train(
    policy,
    train_dataloader,
    val_dataloader,
    tokenizer,
    loss_fn,
    master_config,
    logger,
    checkpointer,
    sft_save_state: SFTSaveState,
) -> None:
    # Run basic sft training
    timer = Timer()
    _telemetry = get_telemetry_handle()
    _tracer = _telemetry.tracer if _telemetry is not None else None
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()

    current_epoch = sft_save_state.epoch
    current_step = sft_save_state.step
    total_steps = sft_save_state.total_steps
    total_valid_tokens = sft_save_state.total_valid_tokens

    sft_config = master_config.sft
    # Validation configuration
    val_period = sft_config.val_period
    val_at_start = sft_config.val_at_start
    val_at_end = sft_config.val_at_end
    max_num_epochs = sft_config.max_num_epochs

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
            val_batches=sft_config.val_batches,
            val_batch_size=sft_config.val_global_batch_size,
            val_mbs=sft_config.val_micro_batch_size,
        )

        logger.log_metrics(val_metrics, total_steps, prefix="validation")
        logger.log_metrics(validation_timings, total_steps, prefix="timing/validation")

    policy.prepare_for_training()

    ft_save_period = master_config.checkpointing.get("ft_save_period")

    while (
        current_epoch < max_num_epochs and total_steps < master_config.sft.max_num_steps
    ):
        print(f"\n{'=' * 25} Epoch {current_epoch + 1}/{max_num_epochs} {'=' * 25}")

        for batch in train_dataloader:
            print(
                f"\n{'=' * 25} Step {current_step + 1}/{min(len(train_dataloader), master_config.sft.max_num_steps)} {'=' * 25}"
            )
            maybe_gpu_profile_step(policy, total_steps + 1)
            val_metrics, validation_timings = None, None

            with (
                timer.time("total_step_time"),
                managed_span(
                    RLSpanGroup.STEP,
                    "rl.sft.step",
                    tracer=_tracer,
                    **{"rl.iteration": total_steps + 1, "rl.epoch": current_epoch + 1},
                ),
            ):
                # Prepare batch and generate responses
                print("▶ Preparing batch...")
                with (
                    timer.time("data_processing"),
                    managed_span(
                        RLSpanGroup.DATA_PROCESSING,
                        "rl.sft.data_processing",
                        tracer=_tracer,
                    ),
                ):
                    train_data = prepare_sft_batch(
                        batch,
                        tokenizer=tokenizer,
                        only_unmask_final=master_config.sft.only_unmask_final,
                        make_sequence_length_divisible_by=master_config.policy[
                            "make_sequence_length_divisible_by"
                        ],
                    )

                print("▶ Taking a training step...")
                with (
                    timer.time("policy_training"),
                    managed_span(
                        RLSpanGroup.POLICY_UPDATE,
                        "rl.sft.policy_training",
                        tracer=_tracer,
                        **{"rl.iteration": total_steps + 1},
                    ),
                ):
                    train_results = policy.train(
                        train_data,
                        loss_fn,
                        timer=timer,
                    )

                is_last_step = total_steps + 1 >= master_config.sft.max_num_steps or (
                    current_epoch + 1 == max_num_epochs
                    and current_step + 1 == len(train_dataloader)
                )

                # Run validation if it's a validation step or last step with val_at_end
                if (val_period > 0 and (total_steps + 1) % val_period == 0) or (
                    val_at_end and is_last_step
                ):
                    val_metrics, validation_timings = validate(
                        policy,
                        val_dataloader,
                        tokenizer,
                        loss_fn,
                        step=total_steps + 1,
                        master_config=master_config,
                        val_batches=sft_config.val_batches,
                        val_batch_size=sft_config.val_global_batch_size,
                        val_mbs=sft_config.val_micro_batch_size,
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
                total_valid_tokens += metrics.get("global_valid_toks", 0)

                ## Checkpointing
                sft_save_state.consumed_samples += master_config.policy[
                    "train_global_batch_size"
                ]
                timeout.mark_iteration()
                should_save_by_step = (
                    is_last_step
                    or (total_steps + 1) % master_config.checkpointing["save_period"]
                    == 0
                    or (
                        ft_save_period is not None
                        and (total_steps + 1) % ft_save_period == 0
                    )
                )
                # +1 because step is 0-indexed
                # Check if timeout-based checkpointing is enabled in config.
                should_save_by_timeout = timeout.check_save()

                if master_config.checkpointing["enabled"] and (
                    should_save_by_step or should_save_by_timeout
                ):
                    sft_save_state.step = (current_step + 1) % len(train_dataloader)
                    sft_save_state.total_steps = total_steps + 1
                    sft_save_state.epoch = current_epoch
                    sft_save_state.total_valid_tokens = total_valid_tokens

                    full_metric_name = master_config.checkpointing["metric_name"]
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
                            if hasattr(sft_save_state, full_metric_name):
                                delattr(sft_save_state, full_metric_name)
                        elif metric_name not in metrics_source:
                            raise ValueError(
                                f"Metric {metric_name} not found in {prefix} metrics"
                            )
                        else:
                            setattr(
                                sft_save_state,
                                full_metric_name,
                                metrics_source[metric_name],
                            )

                    with (
                        timer.time("checkpointing"),
                        managed_span(
                            RLSpanGroup.CHECKPOINT,
                            "rl.sft.checkpointing",
                            tracer=_tracer,
                        ),
                    ):
                        print(f"Saving checkpoint for step {total_steps + 1}...")
                        checkpoint_path = checkpointer.init_tmp_checkpoint(
                            total_steps + 1, vars(sft_save_state), master_config
                        )
                        policy.save_checkpoint(
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
                            train_dataloader.state_dict(),
                            os.path.join(checkpoint_path, "train_dataloader.pt"),
                        )
                        checkpointer.begin_finalization(
                            checkpoint_path,
                            wait_fn=policy.finalize_async_save,
                        )

            timing_metrics = timer.get_timing_metrics(reduction_op="sum")

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
                master_config.cluster["num_nodes"]
                * master_config.cluster["gpus_per_node"]
            )
            if total_time > 0:
                timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                    metrics.get("global_valid_toks", 0) / total_time / total_num_gpus
                )
            else:
                timing_metrics["valid_tokens_per_sec_per_gpu"] = 0.0
            logger.log_metrics(metrics, total_steps + 1, prefix="train")
            logger.log_metrics(timing_metrics, total_steps + 1, prefix="timing/train")

            timer.reset()
            current_step += 1
            total_steps += 1

            if should_save_by_timeout:
                checkpointer.shutdown()
                print("Timeout has been reached, stopping training early", flush=True)
                return
            if total_steps >= master_config.sft.max_num_steps:
                checkpointer.shutdown()
                print(
                    "Max number of steps has been reached, stopping training early",
                    flush=True,
                )
                return

        current_epoch += 1
        current_step = 0  # Reset step counter for new epoch

    # Flush the last checkpoint's background finalization on an epoch-bounded
    # exit. Reaching max_num_epochs falls through the while loop and bypasses
    # the inline shutdown() calls at the max_num_steps / timeout early returns,
    # so without this the daemon finalization thread could be killed before the
    # final tmp_step_N is renamed.
    checkpointer.shutdown()
