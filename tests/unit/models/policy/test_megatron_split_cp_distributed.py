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
"""Real-collective normalization tests for split Megatron draft training."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("megatron.bridge")

from nemo_rl.algorithms.loss.draft import DraftLossStats  # noqa: E402
from nemo_rl.algorithms.loss.interfaces import LossType  # noqa: E402
from nemo_rl.models.megatron.draft.step_state import (  # noqa: E402
    DRAFT_STEP_PAYLOAD_KEY,
    DraftStepState,
)
from tests.unit.models.policy.test_megatron_split_state import (  # noqa: E402
    WORKER_MOD,
    _make_worker,
)

pytestmark = pytest.mark.mcore


def _run_context_parallel_finish_parity(rank: int, world_size: int) -> None:
    singleton_groups = [
        torch.distributed.new_group(ranks=[group_rank])
        for group_rank in range(world_size)
    ]
    dp_group = singleton_groups[rank]
    cp_group = torch.distributed.group.WORLD
    device = torch.device("cuda")

    local_numerators, local_counts = {
        1: ([40.0], [10.0]),
        2: ([0.0, 40.0], [0.0, 10.0]),
        4: ([0.0, 10.0, 0.0, 30.0], [0.0, 2.0, 0.0, 8.0]),
    }[world_size]
    local_numerator = local_numerators[rank]
    local_count = local_counts[rank]

    draft_step_state = DraftStepState()
    payload = DraftStepState.metric_payload(
        DraftLossStats(
            numerators=torch.tensor([local_numerator], device=device),
            counts=torch.tensor([local_count], device=device),
            weights=torch.ones(1, device=device),
        )
    )
    draft_step_state.accumulate(payload)

    worker = _make_worker(LossType.TOKEN_LEVEL)
    worker.rank = rank
    draft_param = torch.nn.Parameter(torch.tensor(1.0, device=device))
    draft_param.grad_norm_group = "draft"
    # MCore's CP loss scaling supplies 1/CP before this finish-time correction.
    draft_param.main_grad = torch.tensor(3.0 / world_size, device=device)
    worker.model.parameters.side_effect = lambda: iter([draft_param])
    worker.optimizer.grad_norms_by_group = {"draft": 0.25}

    state = {
        "draft_step_state": draft_step_state,
        "local_valid_seqs": torch.tensor(8.0, device=device),
        "local_valid_toks": torch.tensor(2048.0, device=device),
        "loss_type": LossType.TOKEN_LEVEL,
        "saved_finalize_model_grads_func": lambda _models, _tokens: None,
        "num_chunks": 1,
        "total_num_microbatches": 1,
        "gbs": 32,
        "metric_normalizations": {},
        "all_mb_metrics": [
            {
                "loss": 2048.0,
                "draft_loss": torch.tensor(local_numerator, device=device),
                DRAFT_STEP_PAYLOAD_KEY: payload,
            }
        ],
        "mb_losses": [2048.0],
    }

    def data_parallel_group(
        *, with_context_parallel: bool = False
    ) -> torch.distributed.ProcessGroup:
        return cp_group if with_context_parallel else dp_group

    def aggregate_training_statistics(
        *,
        all_mb_metrics: list[dict[str, Any]],
        losses: list[float],
        **_kwargs: Any,
    ) -> tuple[dict[str, list[Any]], torch.Tensor]:
        return (
            {
                "loss": [all_mb_metrics[0]["loss"]],
                "draft_loss": [all_mb_metrics[0]["draft_loss"]],
            },
            torch.tensor(sum(losses), device=device),
        )

    with (
        patch(
            f"{WORKER_MOD}.parallel_state.get_data_parallel_group",
            side_effect=data_parallel_group,
        ),
        patch(
            f"{WORKER_MOD}.parallel_state.get_context_parallel_group",
            return_value=cp_group,
        ),
        patch(
            f"{WORKER_MOD}.parallel_state.get_context_parallel_world_size",
            return_value=world_size,
        ),
        patch(
            f"{WORKER_MOD}.get_pg_collection",
            return_value=SimpleNamespace(mp=cp_group),
        ),
        patch(
            f"{WORKER_MOD}.logical_and_across_model_parallel_group",
            side_effect=lambda value, **_kwargs: value,
        ),
        patch(
            f"{WORKER_MOD}.reduce_max_stat_across_model_parallel_group",
            side_effect=lambda value, **_kwargs: value,
        ),
        patch(
            f"{WORKER_MOD}.aggregate_training_statistics",
            side_effect=aggregate_training_statistics,
        ),
    ):
        metrics = worker._finish_train_step_body(state)

    actual_grad = draft_param.main_grad.item()
    actual_metric = metrics["all_mb_metrics"]["draft_loss"][0].item()
    assert actual_grad == pytest.approx(
        614.4, rel=1e-5
    ) and actual_metric == pytest.approx(4.0), (
        f"CP{world_size} normalization mismatch: grad={actual_grad}, metric={actual_metric}"
    )
    assert metrics["all_mb_metrics"]["loss"][0] == pytest.approx(1.0)


@pytest.mark.parametrize("context_parallel_size", [1, 2, 4])
def test_context_parallel_finish_metric_and_draft_grad_parity(
    distributed_test_runner,
    context_parallel_size: int,
) -> None:
    distributed_test_runner(
        _run_context_parallel_finish_parity,
        world_size=context_parallel_size,
    )
