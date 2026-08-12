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

"""Producer-side payload helpers for the async-RL TQ path."""

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.codec import pack_jagged_fields
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import PromptGroupRecord


def record_to_train_batch(
    record: PromptGroupRecord,
    *,
    pad_value_dict: Mapping[str, int],
) -> BatchedDataDict[Any]:
    """Convert one prompt group's record into a packed BatchedDataDict of N rows.

    Args:
        record: Rollout's PromptGroupRecord with N completions to flatten into rows.
        pad_value_dict: Field-name → pad value used by batched_message_log_to_flat_message.

    Returns:
        BatchedDataDict with input_ids, input_lengths, generation_logprobs, token_mask,
        sample_mask, prompt_ids_for_adv, and total_reward.
    """
    # Lazy imports: grpo and llm_message_utils transitively pull
    # experience.rollouts, so importing at module top risks a cycle.
    from nemo_rl.algorithms.grpo import (
        add_grpo_token_loss_masks_and_generation_logprobs,
        extract_initial_prompt_messages,
    )
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message

    completions = record.completions
    n = len(completions)
    assert n > 0, "PromptGroupRecord has no completions"

    message_logs = [c.message_log for c in completions]
    prompt_token_count = sum(len(m["token_ids"]) for m in record.prompt)
    prompt_lengths = torch.full((n,), prompt_token_count, dtype=torch.long)

    prompt_message_logs = extract_initial_prompt_messages(message_logs, prompt_lengths)
    prompt_flat, _ = batched_message_log_to_flat_message(
        prompt_message_logs,
        pad_value_dict=dict(pad_value_dict),  # type: ignore
    )

    add_grpo_token_loss_masks_and_generation_logprobs(message_logs)
    flat, input_lengths = batched_message_log_to_flat_message(
        message_logs,  # type: ignore
        pad_value_dict=dict(pad_value_dict),  # type: ignore
    )

    total_reward = torch.tensor(
        [float(c.reward) for c in completions], dtype=torch.float32
    )
    sample_mask = torch.ones(n, dtype=torch.float32)

    return BatchedDataDict[Any](
        {
            "input_ids": flat["token_ids"],
            "input_lengths": input_lengths,
            "generation_logprobs": flat["generation_logprobs"],
            "token_mask": flat["token_loss_mask"],
            "sample_mask": sample_mask,
            "prompt_ids_for_adv": prompt_flat["token_ids"],
            "total_reward": total_reward,
        }
    )




def compute_failure_reasons_from_record(
    record: PromptGroupRecord,
) -> tuple[list[str], list[bool]]:
    """Extract per-row failure_reason categorical + resolved flag from a record.

    Reads ``Completion.env_extras`` which is the full NeMo-Gym result dict
    (populated by ``AsyncNemoGymRolloutImpl._result_to_completion`` from
    ``full_result``). Handles both shapes seen in the wild:
    ``env_extras["metadata"]["eval_report"]`` and ``env_extras["eval_report"]``.

    Returns:
        (reasons, resolved_flags), each of length ``len(record.completions)``.

    Categoricals:
        - ``"resolved"``           — instance report says resolved==True
        - ``"tests_failed"``       — patch applied, tests ran, some failed
        - ``"patch_apply_failed"`` — patch_successfully_applied==False
        - ``"eval_timeout_or_no_tests"`` — tests_status missing/empty
        - ``"no_report"``          — eval_report missing or malformed
        - ``"exception"``          — outer catch fired (has "error"+"traceback")
    """
    reasons: list[str] = []
    resolved_flags: list[bool] = []
    for c in record.completions:
        env_extras = c.env_extras or {}

        eval_report: Any = None
        metadata_container = env_extras.get("metadata")
        if isinstance(metadata_container, dict):
            eval_report = metadata_container.get("eval_report")
        if eval_report is None:
            eval_report = env_extras.get("eval_report")

        if not eval_report or not isinstance(eval_report, dict):
            reasons.append("no_report")
            resolved_flags.append(False)
            continue

        if "error" in eval_report and "traceback" in eval_report:
            reasons.append("exception")
            resolved_flags.append(False)
            continue

        instance_report: Optional[dict[str, Any]] = None
        for _k, v in eval_report.items():
            if isinstance(v, dict) and "resolved" in v:
                instance_report = v
                break

        if instance_report is None:
            reasons.append("no_report")
            resolved_flags.append(False)
            continue

        if bool(instance_report.get("resolved")):
            reasons.append("resolved")
            resolved_flags.append(True)
            continue

        if not instance_report.get("patch_successfully_applied"):
            reasons.append("patch_apply_failed")
            resolved_flags.append(False)
            continue

        if not instance_report.get("tests_status"):
            reasons.append("eval_timeout_or_no_tests")
            resolved_flags.append(False)
            continue

        reasons.append("tests_failed")
        resolved_flags.append(False)

    return reasons, resolved_flags


def pack_payload(
    train_batch: Mapping[str, Any],
    *,
    weight_version: int,
    group_id: str,
    extra_tags: Optional[list[dict[str, Any]]] = None,
) -> tuple[list[str], TensorDict, list[dict[str, Any]]]:
    """Pack a producer batch into (sample_ids, fields, tags) for put_samples.

    Args:
        train_batch: Mapping with at least input_lengths plus the tensor/object fields to send.
        weight_version: Trainer weight version stamped on every row's tag.
        group_id: Per-group identifier used as the sample_id prefix; the caller owns uniqueness.
        extra_tags: Optional per-row dicts merged into each row's tag before returning.
            Length must equal the number of samples in ``train_batch``.

    Returns:
        sample_ids of the form {group_id}_g{i}, a jagged-packed TensorDict, and per-row tags.
    """
    lengths = train_batch["input_lengths"]
    n = int(lengths.shape[0])
    tensor_fields: dict[str, torch.Tensor | np.ndarray] = {
        k: v
        for k, v in train_batch.items()
        if isinstance(v, torch.Tensor)
        or (isinstance(v, np.ndarray) and v.dtype == object)
    }
    fields_td = pack_jagged_fields(tensor_fields, lengths=lengths)
    sample_ids = [f"{group_id}_g{i}" for i in range(n)]
    tags = [{"weight_version": weight_version} for _ in range(n)]
    if extra_tags is not None:
        if len(extra_tags) != n:
            raise ValueError(
                f"pack_payload: extra_tags length {len(extra_tags)} != n {n}"
            )
        for i, extra in enumerate(extra_tags):
            tags[i].update(extra)
    return sample_ids, fields_td, tags
