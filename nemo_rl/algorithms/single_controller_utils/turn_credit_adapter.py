# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Adapt turn-credit shaping to the replay branch's in-controller advantage stage."""

import json
import logging
from functools import lru_cache
from typing import Any, Mapping

import torch

from nemo_rl.algorithms.single_controller_utils.turn_credit import (
    MAX_TRAILING_RUN_TOKENS,
    RolloutTurnLabels,
    TurnCreditConfig,
    TurnCreditStats,
    shape_turn_credit,
    token_runs,
)
from nemo_rl.data_plane.async_utils import call_data_plane

logger = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def tool_call_token_id(tokenizer_json: str) -> int | None:
    # The tokenizer can load remote assets; resolve it only when extra trailing calls need checking.
    from nemo_rl.algorithms.utils import get_tokenizer

    tokenizer = get_tokenizer(json.loads(tokenizer_json))
    ids = tokenizer.encode("<tool_call>", add_special_tokens=False)
    return ids[0] if len(ids) == 1 else None


async def apply_turn_credit(
    data_plane: Any,
    meta: Any,
    advantages: torch.Tensor,
    *,
    train_mask: torch.Tensor,
    token_mask: torch.Tensor,
    labels: Mapping[str, RolloutTurnLabels],
    config: TurnCreditConfig,
    tokenizer_config: Mapping[str, Any],
) -> tuple[torch.Tensor, TurnCreditStats, list[dict[str, Any]]]:
    """Align canonical row labels and preserve upstream's conservative trailing-call checks."""
    row_labels = {
        i: labels[sid] for i, sid in enumerate(meta.sample_ids) if sid in labels
    }
    row_turns = {}
    candidates = {}
    for i, label in row_labels.items():
        try:
            raw = (meta.tags or [])[i].get("rollout_debug")
            debug = json.loads(raw) if isinstance(raw, str) else (raw or {})
            turns = (debug.get("trace_metadata") or {}).get("turns")
            row_turns[i] = int(turns) if turns is not None else None
        except (IndexError, AttributeError, TypeError, ValueError):
            row_turns[i] = None
        runs = token_runs(token_mask[i])
        extra = len(runs) - label.n_calls
        if (
            label.aligned is True
            and 0 < extra <= label.trailing_empty_calls
            and row_turns[i] == len(runs)
        ):
            candidates[i] = runs[label.n_calls :]
    trailing_ok = {}
    if candidates:
        try:
            tool_id = tool_call_token_id(
                json.dumps(dict(tokenizer_config), sort_keys=True)
            )
        except Exception as error:
            logger.warning(
                "Cannot identify tool-call token; extra-call rows keep GRPO: %s", error
            )
            tool_id = None
        if tool_id is not None:
            data = await call_data_plane(
                data_plane,
                "get_samples",
                sample_ids=[meta.sample_ids[i] for i in candidates],
                partition_id=meta.partition_id,
                select_fields=["input_ids"],
            )
            ids = data["input_ids"]
            rows = ids.unbind() if getattr(ids, "is_nested", False) else ids
            for (i, extra_runs), row in zip(candidates.items(), rows):
                row = row.detach().reshape(-1).cpu()
                trailing_ok[i] = all(
                    end - start <= MAX_TRAILING_RUN_TOKENS
                    and end <= row.numel()
                    and not bool((row[start:end] == tool_id).any())
                    for start, end in extra_runs
                )
    result = shape_turn_credit(
        advantages,
        train_mask=train_mask,
        token_mask=token_mask,
        row_labels=row_labels,
        config=config,
        sample_ids=list(meta.sample_ids),
        row_turns=row_turns,
        trailing_ok=trailing_ok,
    )
    for record in result[2]:
        if str(record.get("status", "")).startswith("drop:"):
            logger.warning("turn_credit mapping rejected: %s", record)
    return result
