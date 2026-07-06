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
"""TransferQueue rollout actor for on-policy distillation."""

from __future__ import annotations

import uuid
from typing import Any, Optional

import numpy as np
import ray
import torch

from nemo_rl.algorithms.grpo import _should_use_async_rollouts
from nemo_rl.data.llm_message_utils import (
    add_loss_mask_to_message_log,
    batched_message_log_to_flat_message,
)
from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane.column_io import kv_first_write
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface


def _as_tq_tensor(value: Any, expected_rows: int) -> Optional[torch.Tensor]:
    """Return a TQ-storable tensor only when it is row-aligned."""
    if isinstance(value, PackedTensor):
        value = value.as_tensor()
    if not isinstance(value, torch.Tensor):
        return None
    if value.dim() == 0 or value.shape[0] != expected_rows:
        return None
    return value


@ray.remote  # pragma: no cover
class DistillationRolloutActor:
    """Runs distillation rollout processing and seeds the TQ partition."""

    def __init__(
        self,
        policy_generation: GenerationInterface,
        tokenizer: Any,
        task_to_env: dict[str, EnvironmentInterface],
        master_config: Any,
        dp_cfg: dict[str, Any],
    ) -> None:
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.master_config = master_config

        from nemo_rl.data_plane import build_data_plane_client

        self._dp_client = build_data_plane_client(dp_cfg, bootstrap=False)

    def rollout_to_tq(
        self,
        input_batch: BatchedDataDict[Any],
        *,
        partition_id: str,
        model_input_fields: list[str],
        first_iter: bool = True,
        finish_generation: bool = True,
        task_to_env_override: Optional[dict[str, EnvironmentInterface]] = None,
    ) -> tuple[
        KVBatchMeta,
        BatchedDataDict[Any],
        dict[str, Any],
        Optional[dict[str, Any]],
    ]:
        """Run rollout, flatten training tensors, and write seed columns to TQ."""
        if first_iter and hasattr(self.policy_generation, "snapshot_step_metrics"):
            self.policy_generation.snapshot_step_metrics()
        if hasattr(self.policy_generation, "clear_logger_metrics"):
            self.policy_generation.clear_logger_metrics()

        cfg = self.master_config
        task_to_env = (
            task_to_env_override
            if task_to_env_override is not None
            else self.task_to_env
        )
        rollout_kwargs = dict(
            policy_generation=self.policy_generation,
            input_batch=input_batch,
            tokenizer=self.tokenizer,
            task_to_env=task_to_env,
            max_seq_len=cfg.policy["max_total_sequence_length"],
            max_rollout_turns=cfg.distillation["max_rollout_turns"],
            greedy=False,
        )
        if _should_use_async_rollouts(cfg):
            final_batch, rollout_metrics = run_async_multi_turn_rollout(
                **rollout_kwargs
            )
        else:
            final_batch, rollout_metrics = run_multi_turn_rollout(**rollout_kwargs)

        final_batch = final_batch.to("cpu")
        add_loss_mask_to_message_log(final_batch["message_log"])
        flat, input_lengths = batched_message_log_to_flat_message(
            final_batch["message_log"],
            pad_value_dict={"token_ids": self.tokenizer.pad_token_id},
            make_sequence_length_divisible_by=cfg.policy[
                "make_sequence_length_divisible_by"
            ],
        )

        bulk_batch = BatchedDataDict(
            {
                "input_ids": flat["token_ids"],
                "input_lengths": input_lengths,
                "token_mask": flat["token_loss_mask"],
                "sample_mask": final_batch["loss_multiplier"],
            }
        )
        expected_rows = int(bulk_batch["sample_mask"].shape[0])
        for field in model_input_fields:
            if field not in flat:
                raise KeyError(f"model input field {field!r} was not produced")
            tensor = _as_tq_tensor(flat[field], expected_rows)
            if tensor is None:
                raise ValueError(
                    f"model input field {field!r} is not row-aligned for TQ"
                )
            bulk_batch[field] = tensor
        bulk_batch.to("cpu")

        sample_ids = [str(uuid.uuid4()) for _ in range(expected_rows)]
        meta = kv_first_write(
            bulk_batch,
            sample_ids=sample_ids,
            dp_client=self._dp_client,
            partition_id=partition_id,
            task_name=partition_id,
            pad_to_multiple=int(cfg.policy["make_sequence_length_divisible_by"]),
        )

        length = final_batch.get("length", input_lengths)
        if not isinstance(length, torch.Tensor):
            length = torch.tensor(length)
        content_values = flat.get("content", [""] * expected_rows)
        content = np.empty(expected_rows, dtype=object)
        for i, item in enumerate(content_values):
            content[i] = item
        driver_carry = BatchedDataDict(
            {
                "length": length,
                "input_lengths": input_lengths,
                "content": content,
            }
        )

        if finish_generation:
            self.policy_generation.finish_generation()
        generation_metrics = (
            self.policy_generation.get_logger_metrics()
            if hasattr(self.policy_generation, "get_logger_metrics")
            else None
        )
        return meta, driver_carry, rollout_metrics, generation_metrics

    def shutdown(self) -> None:
        """Close this actor's data-plane client."""
        try:
            self._dp_client.close()
        except Exception:
            pass
