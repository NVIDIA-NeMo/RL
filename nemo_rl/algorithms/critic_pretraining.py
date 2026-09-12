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

"""Offline scalar critic pretraining with NeMo RL's upstream Value model."""

import json
import math
import os
import re
import warnings
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from functools import partial
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import torch
from pydantic import BaseModel
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import AutoTokenizer

from nemo_rl.algorithms.critic_metrics import compute_critic_evaluation_suites
from nemo_rl.algorithms.loss import MseValueLossConfig, MseValueLossFn
from nemo_rl.algorithms.utils import set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import trajectory_value_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.utils import load_dataloader_state
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    ClusterConfig,
    RayVirtualCluster,
    prepare_segment_topology,
)
from nemo_rl.models.value import Value, ValueConfig, ValueInterface
from nemo_rl.utils.checkpoint import CheckpointingConfig, CheckpointManager
from nemo_rl.utils.logger import Logger, LoggerConfig
from nemo_rl.utils.timer import TimeoutChecker, Timer


@dataclass
class CriticSaveState:
    epoch: int
    step: int
    total_steps: int
    consumed_samples: int
    total_valid_tokens: int


@dataclass(frozen=True)
class CriticCheckpoint:
    """Resolved critic checkpoint used by standalone evaluation."""

    checkpoint_path: Path
    weights_path: Path
    step: int
    training_info: dict[str, Any]


def _initial_save_state() -> CriticSaveState:
    return CriticSaveState(
        epoch=0,
        step=0,
        total_steps=0,
        consumed_samples=0,
        total_valid_tokens=0,
    )


def resolve_critic_checkpoint(checkpoint_path: str | Path) -> CriticCheckpoint:
    """Resolve either a critic step directory or its ``value/weights`` path."""
    requested_path = Path(checkpoint_path).expanduser().resolve()
    if not requested_path.is_dir():
        raise FileNotFoundError(
            f"critic checkpoint path is not a directory: {requested_path}"
        )

    if requested_path.name == "weights" and requested_path.parent.name == "value":
        resolved_checkpoint_path = requested_path.parent.parent
        weights_path = requested_path
    else:
        resolved_checkpoint_path = requested_path
        weights_path = resolved_checkpoint_path / "value" / "weights"
    if not weights_path.is_dir():
        raise FileNotFoundError(
            f"critic checkpoint has no value weights directory: {weights_path}"
        )

    training_info_path = resolved_checkpoint_path / "training_info.json"
    training_info: dict[str, Any] = {}
    if training_info_path.is_file():
        loaded_training_info = json.loads(
            training_info_path.read_text(encoding="utf-8")
        )
        if not isinstance(loaded_training_info, dict):
            raise TypeError(
                f"checkpoint training info must be a mapping: {training_info_path}"
            )
        training_info = loaded_training_info

    name_match = re.fullmatch(r"step_(\d+)", resolved_checkpoint_path.name)
    name_step = int(name_match.group(1)) if name_match is not None else None
    info_step = training_info.get("total_steps")
    if info_step is not None and (
        isinstance(info_step, bool) or not isinstance(info_step, int)
    ):
        raise TypeError(
            f"checkpoint total_steps must be an integer: {training_info_path}"
        )
    if name_step is not None and info_step is not None and name_step != info_step:
        raise ValueError(
            "checkpoint directory step disagrees with training_info.json: "
            f"{name_step} != {info_step}"
        )
    step = info_step if info_step is not None else name_step
    if step is None:
        raise ValueError(
            "cannot determine checkpoint step; use a step_N directory or include "
            f"total_steps in {training_info_path}"
        )

    return CriticCheckpoint(
        checkpoint_path=resolved_checkpoint_path,
        weights_path=weights_path,
        step=step,
        training_info=training_info,
    )


class CriticPretrainingConfig(BaseModel, extra="allow"):
    max_num_steps: int = -1
    max_num_epochs: int = 1
    val_period: int = 0
    val_batches: int = -1
    val_global_batch_size: int = 512
    val_micro_batch_size: int = 1
    val_at_start: bool = False
    val_at_end: bool = True
    val_max_span_targets_per_definition: int = 256
    resume_dataloader: bool = True
    seed: int = 42


class MasterConfig(BaseModel, extra="allow"):
    critic_pretraining: CriticPretrainingConfig
    value_loss_fn: MseValueLossConfig
    value: ValueConfig
    data: DataConfig
    logger: LoggerConfig
    cluster: ClusterConfig
    checkpointing: CheckpointingConfig


def _minimum_megatron_sequence_divisor(megatron_config: dict[str, Any]) -> int:
    """Return the per-sequence padding factor required by Megatron packing."""
    divisor = 1
    context_parallel_size = int(megatron_config["context_parallel_size"])
    tensor_parallel_size = int(megatron_config["tensor_model_parallel_size"])
    if context_parallel_size > 1:
        divisor *= 2 * context_parallel_size
    if tensor_parallel_size > 1 and megatron_config["sequence_parallel"]:
        divisor *= tensor_parallel_size
    return divisor


def validate_config(master_config: MasterConfig) -> None:
    """Reject configurations that silently change offline critic semantics."""
    value_config = master_config.value
    critic_config = master_config.critic_pretraining
    megatron_config = cast(dict[str, Any], value_config["megatron_cfg"])
    dtensor_config = cast(dict[str, Any], value_config["dtensor_cfg"])
    megatron_enabled = bool(megatron_config["enabled"])
    dtensor_enabled = bool(dtensor_config["enabled"])
    if megatron_enabled == dtensor_enabled:
        raise ValueError(
            "exactly one of value.megatron_cfg.enabled and "
            "value.dtensor_cfg.enabled must be true"
        )
    if master_config.value_loss_fn.cliprange is not None:
        raise ValueError(
            "offline critic pretraining requires value_loss_fn.cliprange=null; "
            "PPO value clipping depends on old online value predictions"
        )
    if critic_config.max_num_steps == 0 or critic_config.max_num_steps < -1:
        raise ValueError("critic_pretraining.max_num_steps must be -1 or positive")
    if critic_config.max_num_epochs <= 0:
        raise ValueError("critic_pretraining.max_num_epochs must be positive")
    if critic_config.val_global_batch_size <= 0:
        raise ValueError("critic_pretraining.val_global_batch_size must be positive")
    if critic_config.val_micro_batch_size <= 0:
        raise ValueError("critic_pretraining.val_micro_batch_size must be positive")
    if critic_config.val_max_span_targets_per_definition == 0 or (
        critic_config.val_max_span_targets_per_definition < -1
    ):
        raise ValueError(
            "critic_pretraining.val_max_span_targets_per_definition must be -1 "
            "or positive"
        )
    if (
        master_config.checkpointing["enabled"]
        and master_config.checkpointing["save_period"] <= 0
    ):
        raise ValueError("checkpointing.save_period must be positive when enabled")
    if "num_workers" not in master_config.data:
        raise ValueError("data.num_workers must be specified explicitly")
    sequence_divisor = value_config["make_sequence_length_divisible_by"]
    if sequence_divisor <= 0:
        raise ValueError("value.make_sequence_length_divisible_by must be positive")

    sequence_packing = value_config["sequence_packing"]
    if megatron_enabled:
        minimum_divisor = _minimum_megatron_sequence_divisor(megatron_config)
        if sequence_divisor % minimum_divisor != 0:
            raise ValueError(
                "value.make_sequence_length_divisible_by "
                f"({sequence_divisor}) must be a multiple of {minimum_divisor} "
                "for the configured Megatron TP, CP, and sequence parallelism"
            )
        if (
            megatron_config["context_parallel_size"] > 1
            and not sequence_packing["enabled"]
        ):
            raise ValueError(
                "Megatron value context parallelism requires sequence packing"
            )
        if sequence_packing["enabled"] and sequence_packing.get("fuse_loss", False):
            raise ValueError(
                "value.sequence_packing.fuse_loss must be false because the value "
                "head requires a custom right-shift loss preparation function"
            )
    else:
        if not dtensor_config["_v2"]:
            raise ValueError(
                "DTensor value training requires value.dtensor_cfg._v2=true"
            )
        if sequence_packing["enabled"]:
            raise ValueError("DTensor value training does not support sequence packing")
        if dtensor_config["context_parallel_size"] != 1:
            raise ValueError("DTensor value training requires context_parallel_size=1")
        if value_config["dynamic_batching"]["enabled"]:
            raise ValueError("DTensor value training does not support dynamic batching")
        reward_model_cfg = value_config["reward_model_cfg"]
        if not reward_model_cfg["enabled"]:
            raise ValueError(
                "DTensor value training requires reward_model_cfg.enabled=true"
            )
        if reward_model_cfg["reward_model_type"] != "regression":
            raise ValueError(
                "DTensor value training requires reward_model_type=regression"
            )


def _planned_train_steps(
    config: CriticPretrainingConfig, train_dataloader: StatefulDataLoader
) -> int:
    epoch_steps = config.max_num_epochs * len(train_dataloader)
    if config.max_num_steps == -1:
        return epoch_steps
    return min(config.max_num_steps, epoch_steps)


def _resolve_value_resume_paths(
    last_checkpoint_path: Optional[Path], value_config: ValueConfig
) -> tuple[Optional[Path], Optional[Path]]:
    if last_checkpoint_path is None:
        return None, None
    weights_path = last_checkpoint_path / "value" / "weights"
    if not weights_path.exists():
        warnings.warn(
            f"Value weights are missing from checkpoint {last_checkpoint_path}; "
            "initializing from value.model_name",
            stacklevel=2,
        )
        return None, None
    optimizer_path = last_checkpoint_path / "value" / "optimizer"
    if optimizer_path.exists():
        return weights_path, optimizer_path
    if value_config["megatron_cfg"]["enabled"]:
        # Megatron stores optimizer and scheduler state inside weights/; this
        # argument is the load-optimizer flag rather than a separate directory.
        return weights_path, weights_path
    warnings.warn(
        f"Optimizer state is missing from {optimizer_path}; using a fresh optimizer",
        stacklevel=2,
    )
    return weights_path, None


def _load_save_state(
    checkpointer: CheckpointManager, checkpoint: Optional[Path]
) -> CriticSaveState:
    loaded = checkpointer.load_training_info(checkpoint)
    if loaded is None:
        return _initial_save_state()
    known_fields = {field.name for field in fields(CriticSaveState)}
    defaults = asdict(_initial_save_state())
    defaults.update(
        {key: value for key, value in loaded.items() if key in known_fields}
    )
    return CriticSaveState(**defaults)


def _trajectory_value_collate(
    master_config: MasterConfig, tokenizer: AutoTokenizer
) -> Any:
    return partial(
        trajectory_value_collate_fn,
        tokenizer=tokenizer,
        make_sequence_length_divisible_by=master_config.value[
            "make_sequence_length_divisible_by"
        ],
    )


def _validation_dataloaders(
    master_config: MasterConfig,
    val_dataset: dict[str, AllTaskProcessedDataset],
    collate_fn: Any,
) -> dict[str, StatefulDataLoader]:
    critic_config = master_config.critic_pretraining
    return {
        name: StatefulDataLoader(
            dataset,
            batch_size=critic_config.val_global_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False,
            num_workers=master_config.data["num_workers"],
        )
        for name, dataset in val_dataset.items()
    }


def _critic_cluster(master_config: MasterConfig, name: str) -> RayVirtualCluster:
    cluster_config = master_config.cluster
    num_nodes = cluster_config["num_nodes"]
    segment_size = cluster_config.get("segment_size")
    node_constraints, _, _ = prepare_segment_topology(segment_size, num_nodes)
    return RayVirtualCluster(
        name=name,
        bundle_ct_per_node_list=[cluster_config["gpus_per_node"]] * num_nodes,
        use_gpus=True,
        num_gpus_per_node=cluster_config["gpus_per_node"],
        max_colocated_worker_groups=1,
        port_range_low=cluster_config.get("master_port_range_low"),
        port_range_high=cluster_config.get("master_port_range_high"),
        segment_size=segment_size,
        node_resource_constraints=node_constraints,
    )


def setup(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    train_dataset: AllTaskProcessedDataset,
    val_dataset: dict[str, AllTaskProcessedDataset],
) -> tuple[
    Value,
    RayVirtualCluster,
    StatefulDataLoader,
    dict[str, StatefulDataLoader],
    MseValueLossFn,
    Logger,
    CheckpointManager,
    CriticSaveState,
    MasterConfig,
]:
    """Construct data, distributed Value workers, loss, logging, and resume state."""
    validate_config(master_config)
    set_seed(master_config.critic_pretraining.seed)
    value_config = master_config.value
    data_config = master_config.data
    critic_config = master_config.critic_pretraining

    logger = Logger(master_config.logger)
    logger.log_hyperparams(master_config.model_dump())
    checkpointer = CheckpointManager(master_config.checkpointing)
    last_checkpoint_path = checkpointer.get_latest_checkpoint_path()
    checkpoint_path = (
        Path(last_checkpoint_path) if last_checkpoint_path is not None else None
    )
    save_state = _load_save_state(checkpointer, checkpoint_path)

    collate_fn = _trajectory_value_collate(master_config, tokenizer)
    train_dataloader = StatefulDataLoader(
        train_dataset,
        batch_size=value_config["train_global_batch_size"],
        shuffle=data_config["shuffle"],
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=data_config["num_workers"],
    )
    if last_checkpoint_path is not None and critic_config.resume_dataloader:
        load_dataloader_state(train_dataloader, last_checkpoint_path, data_config)
    elif last_checkpoint_path is not None:
        print(
            "Skipping dataloader-state restore; model and optimizer resume while "
            "the dataset cursor restarts at the beginning",
            flush=True,
        )
        save_state.step = 0

    val_dataloader = _validation_dataloaders(master_config, val_dataset, collate_fn)
    validation_enabled = (
        critic_config.val_period > 0
        or critic_config.val_at_start
        or critic_config.val_at_end
    )
    if validation_enabled and not val_dataloader:
        raise ValueError("a validation dataset is required when validation is enabled")

    print("\nSetting up critic training cluster...", flush=True)
    cluster = _critic_cluster(master_config, "critic_pretraining_cluster")

    if value_config["megatron_cfg"]["enabled"]:
        megatron_config = cast(dict[str, Any], value_config["megatron_cfg"])
        megatron_config["train_iters"] = _planned_train_steps(
            critic_config, train_dataloader
        )
    weights_path, optimizer_path = _resolve_value_resume_paths(
        checkpoint_path, value_config
    )
    value_model = Value(
        cluster=cluster,
        config=value_config,
        tokenizer=tokenizer,
        name_prefix="offline_critic",
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        init_optimizer=True,
    )
    loss_fn = MseValueLossFn(master_config.value_loss_fn)
    print(
        f"Critic setup complete: {len(train_dataset)} train samples, "
        f"{sum(len(dataset) for dataset in val_dataset.values())} validation samples",
        flush=True,
    )
    return (
        value_model,
        cluster,
        train_dataloader,
        val_dataloader,
        loss_fn,
        logger,
        checkpointer,
        save_state,
        master_config,
    )


def setup_evaluation(
    master_config: MasterConfig,
    tokenizer: AutoTokenizer,
    val_dataset: dict[str, AllTaskProcessedDataset],
    checkpoint: str | Path | CriticCheckpoint,
) -> tuple[
    Value,
    RayVirtualCluster,
    dict[str, StatefulDataLoader],
    Logger,
    CriticCheckpoint,
    MasterConfig,
]:
    """Construct a critic for checkpoint-only evaluation without an optimizer."""
    validate_config(master_config)
    set_seed(master_config.critic_pretraining.seed)
    if not val_dataset:
        raise ValueError("standalone critic evaluation requires a validation dataset")

    resolved_checkpoint = (
        checkpoint
        if isinstance(checkpoint, CriticCheckpoint)
        else resolve_critic_checkpoint(checkpoint)
    )
    logger = Logger(master_config.logger)
    hyperparameters = master_config.model_dump()
    hyperparameters["evaluation_checkpoint"] = {
        "checkpoint_path": str(resolved_checkpoint.checkpoint_path),
        "weights_path": str(resolved_checkpoint.weights_path),
        "step": resolved_checkpoint.step,
        "training_info": resolved_checkpoint.training_info,
    }
    logger.log_hyperparams(hyperparameters)

    collate_fn = _trajectory_value_collate(master_config, tokenizer)
    val_dataloader = _validation_dataloaders(master_config, val_dataset, collate_fn)
    print("\nSetting up critic evaluation cluster...", flush=True)
    cluster = _critic_cluster(master_config, "critic_evaluation_cluster")
    value_config = master_config.value
    if value_config.get("megatron_cfg", {}).get("enabled", False):
        # Megatron requires this runtime field during model construction even
        # when evaluation initializes no optimizer and performs no train step.
        value_config["megatron_cfg"]["train_iters"] = 1
    value_model = Value(
        cluster=cluster,
        config=value_config,
        tokenizer=tokenizer,
        name_prefix="offline_critic_eval",
        weights_path=resolved_checkpoint.weights_path,
        optimizer_path=None,
        init_optimizer=False,
    )
    print(
        "Critic evaluation setup complete: "
        f"checkpoint step {resolved_checkpoint.step}, "
        f"{sum(len(dataset) for dataset in val_dataset.values())} validation samples",
        flush=True,
    )
    return (
        value_model,
        cluster,
        val_dataloader,
        logger,
        resolved_checkpoint,
        master_config,
    )


def _pad_batch(
    batch: BatchedDataDict[Any], divisor: int
) -> tuple[BatchedDataDict[Any], int]:
    padding = math.ceil(batch.size / divisor) * divisor - batch.size
    if padding == 0:
        return batch, batch.size
    original_size = batch.size
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            if key == "sample_mask":
                pad = torch.zeros(
                    (padding, *value.shape[1:]), dtype=value.dtype, device=value.device
                )
            else:
                repetitions = (padding,) + (1,) * (value.ndim - 1)
                pad = value[-1:].repeat(repetitions)
            batch[key] = torch.cat([value, pad], dim=0)
        elif isinstance(value, list):
            batch[key] = value + [deepcopy(value[-1]) for _ in range(padding)]
        else:
            raise TypeError(f"cannot pad batch field {key} of type {type(value)}")
    return batch, original_size


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
    os.replace(temporary, path)


def _write_validation_artifacts(
    log_dir: str,
    dataset_name: str,
    step: int,
    records: list[dict[str, Any]],
    metrics: dict[str, float],
    details: dict[str, Any],
) -> None:
    safe_name = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in dataset_name
    )
    artifact_dir = Path(log_dir) / "critic_validation" / f"step_{step:08d}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = artifact_dir / f"{safe_name}_predictions.jsonl"
    temporary = prediction_path.with_name(f".{prediction_path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for record in sorted(
            records,
            key=lambda item: (
                item["trajectory_id"],
                item["token_position"],
                item["target_id"],
            ),
        ):
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, prediction_path)
    _atomic_write_json(
        artifact_dir / f"{safe_name}_metrics.json",
        {"metrics": metrics, "details": details},
    )


def validate_one_dataset(
    value_model: ValueInterface,
    val_dataloader: StatefulDataLoader,
    step: int,
    master_config: MasterConfig,
    dataset_name: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Run value inference and retain per-target predictions for diagnostics."""
    timer = Timer(context={"worker": "critic_validator"})
    records: list[dict[str, Any]] = []
    masked_samples = 0
    critic_config = master_config.critic_pretraining
    dp_size = value_model.sharding_annotations.get_axis_size("data_parallel")  # type: ignore[attr-defined]
    divisor = dp_size * critic_config.val_micro_batch_size

    value_model.prepare_for_inference()
    try:
        with timer.time("total_validation_time"):
            for batch_index, batch in enumerate(val_dataloader):
                masked_samples += int((batch["sample_mask"] == 0).sum().item())
                batch, original_size = _pad_batch(batch, divisor)
                inference_batch = BatchedDataDict(
                    input_ids=batch["input_ids"],
                    input_lengths=batch["input_lengths"],
                    sample_mask=batch["sample_mask"],
                )
                outputs = value_model.get_values(inference_batch, timer=timer)
                values = outputs["values"]
                if values.ndim == 3 and values.shape[-1] == 1:
                    values = values.squeeze(-1)
                for sample_index in range(original_size):
                    if float(batch["sample_mask"][sample_index]) == 0:
                        continue
                    positions = batch["target_positions"][sample_index]
                    targets = batch["target_values"][sample_index]
                    points = batch["target_is_point"][sample_index]
                    definition_indices = batch["target_definition_indices"][
                        sample_index
                    ]
                    definitions = batch["target_definitions"][sample_index]

                    selected_indices = [
                        index for index, is_point in enumerate(points) if is_point
                    ]
                    spans_by_definition: dict[int, list[int]] = {}
                    for target_index, is_point in enumerate(points):
                        if not is_point:
                            spans_by_definition.setdefault(
                                definition_indices[target_index], []
                            ).append(target_index)
                    span_cap = critic_config.val_max_span_targets_per_definition
                    for span_indices in spans_by_definition.values():
                        if span_cap == -1 or len(span_indices) <= span_cap:
                            selected_indices.extend(span_indices)
                        else:
                            evenly_spaced = np.linspace(
                                0, len(span_indices) - 1, span_cap, dtype=np.int64
                            )
                            selected_indices.extend(
                                span_indices[int(index)] for index in evenly_spaced
                            )

                    for target_index in sorted(selected_indices):
                        token_position = int(positions[target_index])
                        definition_index = definition_indices[target_index]
                        definition = definitions[definition_index]
                        target_id = str(
                            definition.get(
                                "target_id",
                                f"target-{definition_index}",
                            )
                        )
                        group_metadata = dict(batch["group_metadata"][sample_index])
                        target_metadata = definition.get("metadata")
                        if isinstance(target_metadata, dict):
                            group_metadata.update(target_metadata)
                        group_metadata.update(
                            {
                                "target_type": definition.get(
                                    "target_type", "unspecified"
                                ),
                            }
                        )
                        record: dict[str, Any] = {
                            "prediction": float(values[sample_index, token_position]),
                            "target": float(targets[target_index]),
                            "target_id": target_id,
                            "target_type": definition.get("target_type", "unspecified"),
                            "target_is_point": bool(points[target_index]),
                            "token_position": token_position,
                            "pivot_id": str(definition.get("pivot_id", target_id)),
                            "trajectory_id": batch["trajectory_id"][sample_index],
                            "experiment_id": batch["experiment_id"][sample_index],
                            "instance_id": batch["instance_id"][sample_index],
                            "label_source": definition.get(
                                "label_source",
                                batch["label_source"][sample_index],
                            ),
                            "evaluation_suite": "dense_exp",
                            "group_metadata": group_metadata,
                            "untruncated_length": int(
                                batch["untruncated_length"][sample_index]
                            ),
                        }
                        pass_count = definition.get(
                            "pass_count", batch["pass_count"][sample_index]
                        )
                        rollout_count = definition.get(
                            "rollout_count", batch["rollout_count"][sample_index]
                        )
                        if isinstance(pass_count, int) and isinstance(
                            rollout_count, int
                        ):
                            record["pass_count"] = pass_count
                            record["rollout_count"] = rollout_count
                        records.append(record)

                    evaluation_positions = batch["evaluation_positions"][sample_index]
                    evaluation_targets = batch["evaluation_values"][sample_index]
                    evaluation_definition_indices = batch[
                        "evaluation_definition_indices"
                    ][sample_index]
                    evaluation_definitions = batch["evaluation_definitions"][
                        sample_index
                    ]
                    for evaluation_index, token_position in enumerate(
                        evaluation_positions
                    ):
                        definition_index = evaluation_definition_indices[
                            evaluation_index
                        ]
                        definition = evaluation_definitions[definition_index]
                        target_id = str(definition["target_id"])
                        group_metadata = dict(batch["group_metadata"][sample_index])
                        target_metadata = definition.get("metadata")
                        if isinstance(target_metadata, dict):
                            group_metadata.update(target_metadata)
                        group_metadata["target_type"] = definition.get(
                            "target_type", "unspecified"
                        )
                        anchor_kind = definition.get("anchor_kind")
                        record = {
                            "prediction": float(
                                values[sample_index, int(token_position)]
                            ),
                            "target": float(evaluation_targets[evaluation_index]),
                            "target_id": target_id,
                            "target_type": definition.get("target_type", "unspecified"),
                            "target_is_point": True,
                            "token_position": int(token_position),
                            "pivot_id": str(definition.get("pivot_id") or target_id),
                            "trajectory_id": batch["trajectory_id"][sample_index],
                            "experiment_id": batch["experiment_id"][sample_index],
                            "instance_id": batch["instance_id"][sample_index],
                            "label_source": definition.get(
                                "label_source", "evaluation_target"
                            ),
                            "evaluation_suite": definition["evaluation_suite"],
                            "anchor_kind": anchor_kind,
                            "state_type": definition.get("state_type"),
                            "group_metadata": group_metadata,
                            "untruncated_length": int(
                                batch["untruncated_length"][sample_index]
                            ),
                        }
                        pass_count = definition.get("pass_count")
                        rollout_count = definition.get("rollout_count")
                        if isinstance(pass_count, int) and isinstance(
                            rollout_count, int
                        ):
                            record["pass_count"] = pass_count
                            record["rollout_count"] = rollout_count
                        records.append(record)
                if (
                    critic_config.val_batches > 0
                    and batch_index >= critic_config.val_batches - 1
                ):
                    break
    finally:
        value_model.finish_inference()  # type: ignore[attr-defined]

    metrics, details = compute_critic_evaluation_suites(records)
    metrics["masked_overlength_samples"] = float(masked_samples)
    _write_validation_artifacts(
        master_config.logger["log_dir"],
        dataset_name,
        step,
        records,
        metrics,
        details,
    )
    raw_timing_metrics = timer.get_timing_metrics(reduction_op="sum")
    timing_metrics = {
        key: float(sum(value)) if isinstance(value, list) else float(value)
        for key, value in raw_timing_metrics.items()
    }
    timer.reset()
    return metrics, timing_metrics


def validate(
    value_model: ValueInterface,
    val_dataloader: dict[str, StatefulDataLoader],
    step: int,
    master_config: MasterConfig,
    logger: Logger,
) -> tuple[dict[str, float], dict[str, float]]:
    all_metrics: dict[str, float] = {}
    all_timings: dict[str, float] = {}
    for dataset_name, dataloader in val_dataloader.items():
        metrics, timings = validate_one_dataset(
            value_model, dataloader, step, master_config, dataset_name
        )
        prefix = f"validation-{dataset_name}"
        logger.log_metrics(metrics, step, prefix=prefix)
        logger.log_metrics(timings, step, prefix=f"timing/{prefix}")
        all_metrics.update(
            {
                f"{prefix}_{metric_name}": metric_value
                for metric_name, metric_value in metrics.items()
            }
        )
        all_timings[f"{prefix}_total_validation_time"] = timings.get(
            "total_validation_time", 0.0
        )
    if all_timings:
        all_timings["total_validation_time"] = sum(all_timings.values())
    return all_metrics, all_timings


def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        flat_values = value.detach().float().cpu().reshape(-1).tolist()
    else:
        flat_values = np.reshape(np.asarray(value, dtype=np.float64), (-1,)).tolist()
    if not isinstance(flat_values, list):
        return float(flat_values)
    return float(sum(float(item) for item in flat_values))


def _aggregate_training_metrics(
    train_results: dict[str, Any], loss_scale: float
) -> dict[str, float]:
    microbatch_metrics = train_results["all_mb_metrics"]
    if not microbatch_metrics:
        raise RuntimeError("no valid samples were present in the training batch")
    metrics: dict[str, float] = {
        "loss": _as_float(train_results["loss"]),
        "grad_norm": _as_float(train_results["grad_norm"]),
    }
    for name, values in microbatch_metrics.items():
        numeric = [_as_float(value) for value in values]
        if name == "values_min":
            metrics[name] = min(numeric)
        elif name == "values_max":
            metrics[name] = max(numeric)
        elif name in {"lr", "wd", "global_valid_seqs", "global_valid_toks"}:
            metrics[name] = sum(numeric) / len(numeric)
        else:
            metrics[name] = sum(numeric)

    returns_mean = metrics.get("returns_mean", 0.0)
    values_mean = metrics.get("values_mean", 0.0)
    returns_sq_mean = metrics.get("returns_sq_mean", 0.0)
    residual_sq_mean = metrics.get("residual_sq_mean", 0.0)
    variance_returns = returns_sq_mean - returns_mean**2
    variance_residual = residual_sq_mean - (returns_mean - values_mean) ** 2
    metrics["explained_variance"] = 1.0 - variance_residual / max(
        variance_returns, 1e-8
    )
    metrics["mse"] = 2.0 * metrics["loss"] / loss_scale
    metrics["rmse"] = math.sqrt(max(metrics["mse"], 0.0))
    return metrics


def _valid_sample_count(batch: BatchedDataDict[Any]) -> int:
    return int(batch["sample_mask"].sum().item())


def _save_checkpoint(
    value_model: ValueInterface,
    train_dataloader: StatefulDataLoader,
    checkpointer: CheckpointManager,
    master_config: MasterConfig,
    save_state: CriticSaveState,
    train_metrics: Optional[dict[str, float]],
    val_metrics: Optional[dict[str, float]],
) -> None:
    training_info: dict[str, Any] = asdict(save_state)
    full_metric_name = master_config.checkpointing["metric_name"]
    if full_metric_name is not None:
        if not (
            full_metric_name.startswith("train:") or full_metric_name.startswith("val:")
        ):
            raise ValueError("checkpoint metric_name must start with train: or val:")
        source_name, metric_name = full_metric_name.split(":", 1)
        source = train_metrics if source_name == "train" else val_metrics
        if source is None:
            warnings.warn(
                f"No {source_name} metrics are available for {metric_name}; "
                "this checkpoint cannot participate in top-k selection",
                stacklevel=2,
            )
        elif metric_name not in source:
            raise ValueError(f"checkpoint metric {metric_name} was not produced")
        else:
            training_info[full_metric_name] = source[metric_name]

    checkpoint_path = Path(
        checkpointer.init_tmp_checkpoint(
            save_state.total_steps, training_info, master_config
        )
    )
    value_path = checkpoint_path / "value"
    value_model.save_checkpoint(
        weights_path=value_path / "weights",
        optimizer_path=(
            value_path / "optimizer" if checkpointer.save_optimizer else None
        ),
        tokenizer_path=value_path / "tokenizer",
        checkpointing_cfg=master_config.checkpointing,
    )
    torch.save(
        train_dataloader.state_dict(),
        checkpoint_path / "train_dataloader.pt",
    )
    checkpointer.finalize_checkpoint(checkpoint_path)


def critic_pretrain(
    value_model: ValueInterface,
    train_dataloader: StatefulDataLoader,
    val_dataloader: dict[str, StatefulDataLoader],
    loss_fn: MseValueLossFn,
    master_config: MasterConfig,
    logger: Logger,
    checkpointer: CheckpointManager,
    save_state: CriticSaveState,
) -> None:
    """Train the upstream token-level Value model on scalar prefix targets."""
    timer = Timer(context={"worker": "critic_driver"})
    timeout = TimeoutChecker(
        timeout=master_config.checkpointing["checkpoint_must_save_by"],
        fit_last_save_time=True,
    )
    timeout.start_iterations()
    critic_config = master_config.critic_pretraining
    current_epoch = save_state.epoch
    current_step = save_state.step
    total_steps = save_state.total_steps
    total_valid_tokens = save_state.total_valid_tokens
    last_checkpointed_step = total_steps if total_steps > 0 else -1
    last_validated_step = -1
    last_train_metrics: Optional[dict[str, float]] = None
    last_val_metrics: Optional[dict[str, float]] = None

    training_active = False
    try:
        if critic_config.val_at_start and total_steps == 0:
            last_val_metrics, _ = validate(
                value_model, val_dataloader, 0, master_config, logger
            )
            last_validated_step = 0

        value_model.prepare_for_training()
        training_active = True
        stop_requested = False
        while current_epoch < critic_config.max_num_epochs and (
            critic_config.max_num_steps == -1
            or total_steps < critic_config.max_num_steps
        ):
            print(
                f"Epoch {current_epoch + 1}/{critic_config.max_num_epochs} "
                f"starting at dataloader step {current_step}",
                flush=True,
            )
            for batch in train_dataloader:
                current_step += 1
                valid_samples = _valid_sample_count(batch)
                if valid_samples == 0:
                    if current_step <= 5 or current_step % 100 == 0:
                        print(
                            f"Skipping dataloader batch {current_step}: all examples "
                            "were masked by max sequence length",
                            flush=True,
                        )
                    continue

                next_total_step = total_steps + 1
                with timer.time("total_step_time"):
                    train_results = value_model.train(
                        batch,
                        loss_fn,
                        eval_mode=False,
                        gbs=master_config.value["train_global_batch_size"],
                        mbs=master_config.value["train_micro_batch_size"],
                        timer=timer,
                    )

                    last_train_metrics = _aggregate_training_metrics(
                        train_results, master_config.value_loss_fn.scale
                    )
                    total_valid_tokens += int(
                        last_train_metrics.get("global_valid_toks", 0.0)
                    )
                    total_steps = next_total_step
                    save_state.epoch = current_epoch
                    save_state.step = current_step % len(train_dataloader)
                    save_state.total_steps = total_steps
                    save_state.consumed_samples += master_config.value[
                        "train_global_batch_size"
                    ]
                    save_state.total_valid_tokens = total_valid_tokens

                    should_validate = critic_config.val_period > 0 and (
                        total_steps % critic_config.val_period == 0
                    )
                    if should_validate:
                        value_model.finish_training()
                        training_active = False
                        last_val_metrics, _ = validate(
                            value_model,
                            val_dataloader,
                            total_steps,
                            master_config,
                            logger,
                        )
                        last_validated_step = total_steps
                        value_model.prepare_for_training()
                        training_active = True

                    timeout.mark_iteration()
                    checkpointing_enabled = master_config.checkpointing["enabled"]
                    should_save_by_step = checkpointing_enabled and (
                        total_steps % master_config.checkpointing["save_period"] == 0
                    )
                    should_save_by_timeout = (
                        checkpointing_enabled and timeout.check_save()
                    )
                    if should_save_by_step or should_save_by_timeout:
                        _save_checkpoint(
                            value_model,
                            train_dataloader,
                            checkpointer,
                            master_config,
                            save_state,
                            last_train_metrics,
                            last_val_metrics,
                        )
                        last_checkpointed_step = total_steps

                raw_timing_metrics = timer.get_timing_metrics(reduction_op="sum")
                timing_metrics = {
                    key: float(sum(value)) if isinstance(value, list) else float(value)
                    for key, value in raw_timing_metrics.items()
                }
                total_time = timing_metrics.get("total_step_time", 0.0)
                total_gpus = (
                    master_config.cluster["num_nodes"]
                    * master_config.cluster["gpus_per_node"]
                )
                if total_time > 0 and total_gpus > 0:
                    timing_metrics["valid_tokens_per_sec_per_gpu"] = (
                        last_train_metrics.get("global_valid_toks", 0.0)
                        / total_time
                        / total_gpus
                    )
                print(
                    f"Step {total_steps}: mse={last_train_metrics['mse']:.6f}, "
                    f"value_mean={last_train_metrics.get('values_mean', float('nan')):.4f}, "
                    f"target_mean={last_train_metrics.get('returns_mean', float('nan')):.4f}",
                    flush=True,
                )
                logger.log_metrics(last_train_metrics, total_steps, prefix="train")
                logger.log_metrics(timing_metrics, total_steps, prefix="timing/train")
                timer.reset()

                if should_save_by_timeout:
                    print(
                        "Checkpoint deadline reached; stopping after save", flush=True
                    )
                    stop_requested = True
                    break
                if (
                    critic_config.max_num_steps != -1
                    and total_steps >= critic_config.max_num_steps
                ):
                    stop_requested = True
                    break

            if stop_requested:
                break
            current_epoch += 1
            current_step = 0
            save_state.epoch = current_epoch
            save_state.step = 0

        if total_steps == 0:
            raise RuntimeError(
                "training produced no optimizer steps; every full batch was masked "
                "or the dataset was smaller than the global batch size"
            )
        ran_final_validation = False
        if critic_config.val_at_end and last_validated_step != total_steps:
            value_model.finish_training()
            training_active = False
            last_val_metrics, _ = validate(
                value_model, val_dataloader, total_steps, master_config, logger
            )
            ran_final_validation = True
        if master_config.checkpointing["enabled"] and (
            last_checkpointed_step != total_steps or ran_final_validation
        ):
            if not training_active:
                value_model.prepare_for_training()
                training_active = True
            _save_checkpoint(
                value_model,
                train_dataloader,
                checkpointer,
                master_config,
                save_state,
                last_train_metrics,
                last_val_metrics,
            )
    finally:
        if training_active:
            value_model.finish_training()
